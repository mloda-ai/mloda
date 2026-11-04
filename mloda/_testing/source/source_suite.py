"""Contract tests every SourceFeatureGroup implementation can inherit; a host sets the attributes and builders.

Imports pytest at top on purpose (pyarrow only through a guarded plugin import): the pytest-dependent part of mloda._testing.
Not collected on its own (no Test prefix).
"""

import copy
import dataclasses
import gc
import importlib
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, cast

import pytest

from mloda._testing.suite import ConformanceSuite
from mloda._testing.source_applicability import NO_POINTER, SourceApplicability, source_applicability
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.source_feature_group import HANDLE_OPTION
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.match_rejection import (
    CoreDeclineCode,
    MatchRejection,
    collect_match_rejections,
)
from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.utils import get_all_subclasses, is_nominal_subclass
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    resolve_or_raise,
)
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.provider import CHAIN_SEPARATOR, COLUMN_SEPARATOR, ComputeFramework, SourceFeatureGroup
from mloda.steward import ReaderProbeScope, reader_probe_scope
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class LoadCapture(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None:
            self.contexts.append(context)
        return result


class MatchCapture(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        self.contexts.append(context)
        return func(*args, **kwargs)


class MatchRefuseAll(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.name = "suite_match_gate"
        self.never_fall_back = True

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("no clearance")


def expected_implementation_version(loader: str | None) -> str | None:
    """The version() of the implementation class a ``module:qualname`` loader names; None for neutral."""
    if loader is None or loader == "neutral":
        return None
    module_name, qualname = loader.split(":")
    return cast(str, getattr(importlib.import_module(module_name), qualname).version())


def _spy(name: str, original: Any, touched: list[str]) -> Any:
    def spy(klass: Any, *args: Any, **kwargs: Any) -> Any:
        touched.append(name)
        return original(*args, **kwargs)

    return classmethod(spy)


def _open_rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a rejection window around one matcher call."""
    with collect_match_rejections() as window:
        yield window


def _new_implementation_builder() -> Callable[..., type[Implementation[Any]]]:
    """make(group, framework, fn=None, base=Implementation, name=None) builds a throwaway Implementation."""

    def make(
        group: Any,
        framework: Any,
        fn: Callable[[Any, Any, Any], Any] | None = None,
        base: type[Implementation[Any]] = Implementation,
        name: str | None = None,
    ) -> type[Implementation[Any]]:
        class _Impl(base):  # type: ignore[misc, valid-type]
            feature_group = group
            compute_framework = framework

            @classmethod
            def calculate(cls, serving: Any, data: Any, features: Any, framework: Any) -> Any:
                return None if fn is None else fn(serving, data, features)

        if name is not None:
            _Impl.__name__ = name
            _Impl.__qualname__ = name
        return _Impl

    return make


def _implementation_scope() -> Iterator[Callable[..., type[Implementation[Any]]]]:
    """Yield the builder; teardown asserts no class it built is still loaded after collection."""
    before = set(get_all_subclasses(Implementation))
    yield _new_implementation_builder()
    gc.collect()
    loaded = loaded_implementations()
    leaked = sorted(cls.__name__ for cls in get_all_subclasses(Implementation) if cls not in before and cls in loaded)
    assert not leaked, f"leaked implementations: {leaked}"


class SourceFeatureGroupTestMixin(ConformanceSuite, suite_base=True):
    """Contract tests of a source group.

    Required: feature_group_class, present_column, missing_column, expected_source, own_dac, foreign_dac,
    other_pointer, foreign_handle_dac, known_handles_dac, and own_pointer (return NO_POINTER for a group that
    takes no pointer). Optional: data_access_handle (None derives it from the routes), pointed_source (None
    claims with own_pointer), load_framework, context_options. rejection_window and make_implementation are
    fixture methods so a host in any package gets them without this repo's conftest.
    """

    feature_group_class: type[SourceFeatureGroup]
    present_column: str
    missing_column: str
    expected_source: str
    load_framework: type[ComputeFramework] = PyArrowTable
    context_options: ClassVar[dict[str, Any]] = {}
    data_access_handle: ClassVar[bool | None] = None
    pointed_source: ClassVar[str | None] = None

    @pytest.fixture
    def rejection_window(self) -> Iterator[dict[str, MatchRejection]]:
        yield from _open_rejection_window()

    @pytest.fixture
    def make_implementation(self) -> Iterator[Callable[..., type[Implementation[Any]]]]:
        yield from _implementation_scope()

    def _applicability(self) -> SourceApplicability:
        derived = source_applicability(self.feature_group_class, self.own_pointer() is not NO_POINTER)
        if self.data_access_handle is None:
            return derived
        return dataclasses.replace(derived, data_access_handle=self.data_access_handle)

    @staticmethod
    def _require(applies: bool, fact: str) -> None:
        if not applies:
            pytest.skip(f"not applicable: {fact}")

    @property
    def unpointed_source(self) -> str:
        """What an unpointed missing-column reason names for the source (default: expected_source)."""
        return self.expected_source

    def own_dac(self) -> DataAccessCollection:
        """A fresh collection holding a source of the group's own kind with present_column."""
        raise NotImplementedError

    def foreign_dac(self) -> DataAccessCollection:
        """A fresh collection holding only sources of another kind."""
        raise NotImplementedError

    def own_pointer(self) -> Any:
        """A pointer value naming the group's own source (the one own_dac holds), or NO_POINTER."""
        raise NotImplementedError

    def other_pointer(self) -> tuple[Any, str]:
        """A pointer to a second source of the kind holding present_column, with its expected source."""
        raise NotImplementedError

    def foreign_handle_dac(self) -> tuple[DataAccessCollection, str]:
        """The own source plus a handle of another kind, and that handle's name."""
        raise NotImplementedError

    def known_handles_dac(self) -> tuple[DataAccessCollection, list[str]]:
        """A collection with own-kind handles, and the handle names an unknown-handle rejection must list."""
        raise NotImplementedError

    def _group_name(self) -> str:
        return self.feature_group_class.get_class_name()

    def _options(self, group: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> Options:
        return Options(group, context={**self.context_options, **(context or {})})

    def _feature(self, name: str, group: dict[str, Any] | None = None, **kwargs: Any) -> Feature:
        return Feature(name, self._options(group), **kwargs)

    def _with_context(self, feature: Feature) -> Feature:
        if all(feature.options.context.get(key) == value for key, value in self.context_options.items()):
            return feature
        clone = copy.copy(feature)
        clone.options = Options(
            dict(feature.options.group), context={**feature.options.context, **self.context_options}
        )
        return clone

    def _evaluate(self, feature: Feature, dac: DataAccessCollection | None) -> EvaluationResult:
        return IdentifyFeatureGroupClass.evaluate(self._with_context(feature), self._plugins(), None, dac)

    def _resolve(self, feature: Feature, dac: DataAccessCollection | None) -> EvaluationResult:
        """Like _evaluate but raises the typed error (a ValueError) on an abort or failure."""
        return resolve_or_raise(self._with_context(feature), self._plugins(), None, dac)

    def _claimed_match(self, feature: Feature) -> SourceMatch:
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is self.feature_group_class
        return cast(SourceMatch, pair[1])

    def _plugins(self) -> FeatureGroupEnvironmentMapping:
        return {self.feature_group_class: {self.load_framework}}

    def _pointed_source(self) -> str:
        """The source the group claims when pointed at its own pointer."""
        feature = self._feature(self.present_column, {self._group_name(): self.own_pointer()})
        assert self._claims(feature, None)
        return self._claimed_match(feature).source

    def _claims(self, feature: Feature, dac: DataAccessCollection | None) -> bool:
        return self.feature_group_class in self._evaluate(feature, dac).identified

    def _gated_subclass(self) -> Any:
        """A subclass that claims only for a pointer keyed on its parent, so it cannot leak into other tests."""
        parent = self.feature_group_class
        parent_key = parent.__name__

        def gated(cls: Any, feature_name: Any, options: Options, data_access_collection: Any = None) -> bool:
            if parent_key not in options:
                return False
            return bool(
                getattr(super(cls, cls), "match_feature_group_criteria")(feature_name, options, data_access_collection)
            )

        return type(f"{parent.__name__}SuiteTakeover", (parent,), {"match_feature_group_criteria": classmethod(gated)})

    def test_claims_its_own_source(self) -> None:
        assert self._claims(Feature(self.present_column), self.own_dac())

    def test_declines_foreign_sources(self) -> None:
        assert not self._claims(Feature(self.present_column), self.foreign_dac())

    def test_leaves_collection_and_options_unchanged(self) -> None:
        dac = self.own_dac()
        state = copy.deepcopy(dac.__dict__)
        feature = Feature(self.present_column, Options({"unrelated_option": 1}))
        group = copy.deepcopy(feature.options.group)
        context = copy.deepcopy(feature.options.context)

        self._claims(feature, dac)

        assert dac.__dict__ == state
        assert feature.options.group == group
        assert feature.options.context == context
        assert RESERVED_READER_OPTION_KEY not in feature.options.group
        assert RESERVED_READER_OPTION_KEY not in feature.options.context

    def test_declines_a_missing_column_on_checked_routes(self) -> None:
        self._require(self._applicability().checked_route, "no checked route")
        assert not self._claims(Feature(self.missing_column), self.own_dac())
        assert not self._claims(Feature(self.feature_group_class.get_class_name()), self.own_dac())

    def test_an_arbitrary_name_does_not_claim_with_the_own_source_present(self) -> None:
        assert not self._claims(Feature("mixin_arbitrary_undeclared_name"), self.own_dac())

    def test_declines_a_feature_named_after_the_group_without_a_source(self) -> None:
        name = self.feature_group_class.get_class_name()
        assert not self._claims(Feature(name), None)
        assert not self._claims(Feature(name), self.foreign_dac())

    def _expected_loader(self) -> str:
        """The loader the group itself selects for its own source: the implementation path, else neutral."""
        feature = self._with_context(Feature(self.present_column))
        IdentifyFeatureGroupClass.evaluate(feature, self._plugins(), None, self.own_dac())
        impl = self.feature_group_class.select_implementation(
            self.load_framework, self._claimed_match(feature), loaded_implementations()
        )
        if impl is None:
            return "neutral"
        assert is_nominal_subclass(self.feature_group_class, impl.feature_group)
        return f"{impl.__module__}:{impl.__qualname__}"

    def test_fires_input_data_load_naming_group_source_format_and_loader(self) -> None:
        capture = LoadCapture()

        mloda.run_all(
            [self._feature(self.present_column)],
            compute_frameworks=[self.load_framework],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=self.own_dac(),
            function_extender={capture},
        )

        assert len(capture.contexts) == 1
        context = capture.contexts[0]
        group_name = self.feature_group_class.get_class_name()
        assert context.data_access_format == group_name
        assert context.reader_class is self.feature_group_class
        assert context.data_access_identity == self.expected_source
        assert context.data_access_loader == self._expected_loader()
        assert context.implementation == context.data_access_loader
        assert context.implementation_version == expected_implementation_version(context.data_access_loader)

    def test_a_refusing_input_data_match_gate_keeps_the_source_untouched_and_reports_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        touched: list[str] = []
        for name in ("find_sources", "columns", "has_column"):
            monkeypatch.setattr(
                self.feature_group_class, name, _spy(name, getattr(self.feature_group_class, name), touched)
            )

        with reader_probe_scope(ReaderProbeScope(MatchRefuseAll(), None, None, None, None)):
            result = self._evaluate(Feature(self.present_column), self.own_dac())

        assert touched == []
        assert self.feature_group_class not in result.identified
        reason = result.eliminations[self.feature_group_class].reason
        assert "refused at INPUT_DATA_MATCH" in reason
        assert "no clearance" in reason
        assert result.eliminations[self.feature_group_class].code is CoreDeclineCode.EXTENDER_REFUSED

    def test_fires_input_data_match_naming_the_group_class_and_format(self) -> None:
        capture = MatchCapture()

        with reader_probe_scope(ReaderProbeScope(capture, None, None, None, None)):
            assert self._claims(Feature(self.present_column), self.own_dac())

        contexts = [c for c in capture.contexts if c.reader_class is self.feature_group_class]
        assert contexts
        assert all(c.hook is ExtenderHook.INPUT_DATA_MATCH for c in contexts)
        assert all(c.data_access_format == self.feature_group_class.data_access_name() for c in contexts)

    def test_claims_via_feature_group_scope(self) -> None:
        feature = self._feature(self.present_column, feature_group=self.feature_group_class)
        assert self._claims(feature, self.own_dac())

    def test_pointer_value_is_used_alone_and_ignores_the_collection(self) -> None:
        self._require(self._applicability().pointer, "the group takes no pointer")
        value, expected = self.other_pointer()
        feature = self._feature(self.present_column, {self._group_name(): value})
        assert self._claims(feature, self.own_dac())
        assert self._claimed_match(feature).source == expected

    def test_a_pointer_of_the_wrong_type_declines_naming_its_type_and_never_searches_the_collection(self) -> None:
        feature = self._feature(self.present_column, {self._group_name(): 5})
        result = self._evaluate(feature, self.own_dac())
        assert self.feature_group_class not in result.identified
        assert result.eliminations[self.feature_group_class].code is CoreDeclineCode.POINTER_INVALID
        if self._applicability().pointer:
            assert "int" in result.eliminations[self.feature_group_class].reason

    def test_a_handle_of_another_kind_yields_no_sources(self) -> None:
        self._require(self._applicability().data_access_handle, "the group selects by connection_handle")
        dac, handle = self.foreign_handle_dac()
        feature = Feature(self.present_column, self._options(context={HANDLE_OPTION: handle}))
        assert not self._claims(feature, dac)

    def test_an_unknown_handle_records_a_rejection_listing_the_known_handles(self) -> None:
        self._require(self._applicability().data_access_handle, "the group selects by connection_handle")
        dac, known = self.known_handles_dac()
        options = self._options(context={HANDLE_OPTION: "unknown_handle_for_suite"})
        outcome = probe_match_criteria(self.feature_group_class, self.present_column, options, dac)
        assert not outcome.matched
        assert outcome.rejection is not None
        assert outcome.rejection.code is CoreDeclineCode.HANDLE_UNKNOWN
        assert "unknown_handle_for_suite" in outcome.rejection.reason
        for handle in known:
            assert handle in outcome.rejection.reason

    def test_a_non_str_handle_is_no_narrowing_and_is_rejected_by_the_option_check(self) -> None:
        self._require(self._applicability().data_access_handle, "data_access_handle is not an option of the group")
        feature = Feature(self.present_column, self._options(context={HANDLE_OPTION: ["own_handle"]}))
        assert not self._claims(feature, self.own_dac())

    def test_pointed_with_a_missing_column_aborts_naming_source_and_columns(self) -> None:
        self._require(self._applicability().checked_route, "no checked route")
        self._require(self._applicability().pointer, "the group takes no pointer")
        feature = self._feature(self.missing_column, {self._group_name(): self.own_pointer()})
        with pytest.raises(FeatureResolutionError) as exc_info:
            self._resolve(feature, None)
        message = str(exc_info.value)
        assert (self.pointed_source or self._pointed_source()) in message
        assert self.present_column in message
        assert exc_info.value.result.eliminations[self.feature_group_class].code is CoreDeclineCode.COLUMN_NOT_FOUND

    def test_pointed_with_query_text_aborts_naming_the_query_route_when_the_group_lists_none(self) -> None:
        applicability = self._applicability()
        self._require(applicability.pointer, "the group takes no pointer")
        self._require(not applicability.query_route, "the group lists a query route, so query_text is served")
        feature = self._feature(self.present_column, {self._group_name(): self.own_pointer(), "query_text": "select 1"})
        with pytest.raises(FeatureResolutionError) as exc_info:
            self._resolve(feature, None)
        assert "QUERY_ROUTE" in str(exc_info.value)
        assert exc_info.value.result.eliminations[self.feature_group_class].code is CoreDeclineCode.NO_QUERY_ROUTE

    def test_unpointed_with_query_text_declines_naming_the_query_route_when_the_group_lists_none(self) -> None:
        self._require(not self._applicability().query_route, "the group lists a query route, so query_text is served")
        feature = self._feature(self.present_column, {"query_text": "select 1"})
        result = self._evaluate(feature, self.own_dac())
        assert self.feature_group_class not in result.identified
        assert "QUERY_ROUTE" in result.eliminations[self.feature_group_class].reason
        assert result.eliminations[self.feature_group_class].code is CoreDeclineCode.NO_QUERY_ROUTE

    def test_unpointed_missing_column_declines_with_a_rejection(self) -> None:
        self._require(self._applicability().checked_route, "no checked route")
        result = self._evaluate(Feature(self.missing_column), self.own_dac())
        assert self.feature_group_class not in result.identified
        reason = result.eliminations[self.feature_group_class].reason
        assert self.unpointed_source in reason
        assert self.missing_column in reason
        assert result.eliminations[self.feature_group_class].code is CoreDeclineCode.COLUMN_NOT_FOUND

    def test_missing_column_text_hides_the_columns_unless_a_handle_narrows_the_search(self) -> None:
        self._require(self._applicability().checked_route, "no checked route")
        self._require(self._applicability().data_access_handle, "narrowing is by connection_handle")
        unpointed = self._evaluate(Feature(self.missing_column), self.own_dac())
        reason = unpointed.eliminations[self.feature_group_class].reason
        assert self.unpointed_source in reason
        assert self.present_column not in reason

        dac = self.own_dac()
        handle = sorted(dac.handles())[0]
        narrowed = Feature(self.missing_column, self._options(context={HANDLE_OPTION: handle}))
        reason = self._evaluate(narrowed, dac).eliminations[self.feature_group_class].reason
        assert self.expected_source in reason
        assert self.present_column in reason

    def test_a_typo_of_a_known_column_suggests_it_only_where_the_columns_are_shown(self) -> None:
        self._require(self._applicability().checked_route, "no checked route")
        applicability = self._applicability()
        self._require(applicability.pointer or applicability.data_access_handle, "no pointer or data_access_handle")
        typo = f"{self.present_column}x"
        suggestion = f"did you mean '{self.present_column}'?"

        if applicability.pointer:
            pointed = self._feature(typo, {self._group_name(): self.own_pointer()})
            with pytest.raises(ValueError) as exc_info:
                self._resolve(pointed, None)
            assert suggestion in str(exc_info.value)

        unpointed = self._evaluate(Feature(typo), self.own_dac())
        assert "did you mean" not in unpointed.eliminations[self.feature_group_class].reason

        if applicability.data_access_handle:
            dac = self.own_dac()
            narrowed = Feature(typo, self._options(context={HANDLE_OPTION: sorted(dac.handles())[0]}))
            assert suggestion in self._evaluate(narrowed, dac).eliminations[self.feature_group_class].reason

    def test_chain_and_column_separated_names_that_are_not_columns_decline(self) -> None:
        self._require(self._applicability().checked_route, "no checked route")
        chained = f"{self.present_column}{CHAIN_SEPARATOR}suite_rebased"
        separated = f"{self.missing_column}{COLUMN_SEPARATOR}0"
        assert not self._claims(Feature(chained), self.own_dac())
        outcome = probe_match_criteria(self.feature_group_class, chained, self._options(), self.own_dac())
        assert not outcome.matched
        assert outcome.rejection is not None
        assert outcome.rejection.code is CoreDeclineCode.COLUMN_NOT_FOUND
        assert self.unpointed_source in outcome.rejection.reason
        assert chained in outcome.rejection.reason
        assert not self._claims(Feature(separated), self.own_dac())

    def test_a_subclass_takes_over_a_pointer_keyed_on_its_parent(self) -> None:
        self._require(self._applicability().pointer, "the group takes no pointer")
        sub = self._gated_subclass()
        mapping: FeatureGroupEnvironmentMapping = {cast(type[SourceFeatureGroup], sub): {self.load_framework}}
        feature = self._feature(self.present_column, {self.feature_group_class.__name__: self.own_pointer()})
        result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, None)
        assert sub in result.identified
        del sub, mapping, result, feature

    def test_a_pointer_key_of_another_concrete_group_makes_the_searched_routes_decline_with_a_reason(self) -> None:
        sub = self._gated_subclass()
        mapping: FeatureGroupEnvironmentMapping = {
            self.feature_group_class: {self.load_framework},
            cast(type[SourceFeatureGroup], sub): {self.load_framework},
        }
        pointer = self.own_pointer()
        feature = self._feature(self.present_column, {sub.__name__: "unused" if pointer is NO_POINTER else pointer})
        result = IdentifyFeatureGroupClass.evaluate(self._with_context(feature), mapping, None, self.own_dac())
        assert self.feature_group_class not in result.identified
        reason = result.eliminations[self.feature_group_class].reason
        assert f"options point at {sub.__name__}" in reason
        assert "does not search the collection" in reason
        assert result.eliminations[self.feature_group_class].code is CoreDeclineCode.POINTED_ELSEWHERE
        del sub, mapping, result, feature
