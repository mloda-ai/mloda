"""Shared contract tests for FormatFeatureGroup implementations.

A concrete test class sets the class attributes and implements the two collection builders. Not collected on its
own (no Test prefix).
"""

import copy
import gc
from typing import Any, ClassVar, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass, resolve_or_raise
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.provider import CHAIN_SEPARATOR, COLUMN_SEPARATOR, ComputeFramework, FormatFeatureGroup, NamePolicy
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

HANDLE_OPTION = "data_access_handle"


class _LoadCapture(Extender):
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


class FormatFeatureGroupTestMixin:
    """Contract of every format group; the concrete class supplies the hooks below."""

    feature_group_class: type[FormatFeatureGroup]
    present_column: str
    missing_column: str
    expected_source: str
    load_framework: type[ComputeFramework] = PyArrowTable
    load_loader_names: tuple[str, ...] = ("neutral", "PyArrowTable")
    context_options: ClassVar[dict[str, Any]] = {}

    def own_dac(self) -> DataAccessCollection:
        """A fresh collection holding a source of the group's own kind with present_column."""
        raise NotImplementedError

    def foreign_dac(self) -> DataAccessCollection:
        """A fresh collection holding only sources of another kind."""
        raise NotImplementedError

    def own_pointer(self) -> Any:
        """A pointer value naming the group's own source (the one own_dac holds)."""
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
        return {self.feature_group_class: {PyArrowTable}}

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

        return type(f"{parent.__name__}ToyfmtTakeover", (parent,), {"match_feature_group_criteria": classmethod(gated)})

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
        if not any(r.names is NamePolicy.CHECKED for r in self.feature_group_class.CLAIM_ROUTES):
            pytest.skip("no checked route")
        assert not self._claims(Feature(self.missing_column), self.own_dac())
        assert not self._claims(Feature(self.feature_group_class.get_class_name()), self.own_dac())

    def test_an_arbitrary_name_does_not_claim_with_the_own_source_present(self) -> None:
        assert not self._claims(Feature("mixin_arbitrary_undeclared_name"), self.own_dac())

    def test_declines_a_feature_named_after_the_group_without_a_source(self) -> None:
        name = self.feature_group_class.get_class_name()
        assert not self._claims(Feature(name), None)
        assert not self._claims(Feature(name), self.foreign_dac())

    def test_fires_input_data_load_naming_group_source_format_and_loader(self) -> None:
        capture = _LoadCapture()

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
        assert context.data_access_loader in self.load_loader_names

    def test_claims_via_feature_group_scope(self) -> None:
        feature = self._feature(self.present_column, feature_group=self.feature_group_class)
        assert self._claims(feature, self.own_dac())

    def test_pointer_value_is_used_alone_and_ignores_the_collection(self) -> None:
        value, expected = self.other_pointer()
        feature = self._feature(self.present_column, {self._group_name(): value})
        assert self._claims(feature, self.own_dac())
        assert self._claimed_match(feature).source == expected

    def test_a_pointer_of_the_wrong_type_declines_naming_its_type_and_never_searches_the_collection(self) -> None:
        feature = self._feature(self.present_column, {self._group_name(): 5})
        result = self._evaluate(feature, self.own_dac())
        assert self.feature_group_class not in result.identified
        assert "int" in result.eliminations[self.feature_group_class].reason

    def test_a_handle_of_another_kind_yields_no_sources(self) -> None:
        dac, handle = self.foreign_handle_dac()
        feature = Feature(self.present_column, self._options(context={HANDLE_OPTION: handle}))
        assert not self._claims(feature, dac)

    def test_an_unknown_handle_records_a_rejection_listing_the_known_handles(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        dac, known = self.known_handles_dac()
        options = self._options(context={HANDLE_OPTION: "toyfmt_missing_handle"})
        assert not self.feature_group_class.match_feature_group_criteria(self.present_column, options, dac)
        reason = rejection_window[self._group_name()].reason
        assert "toyfmt_missing_handle" in reason
        for handle in known:
            assert handle in reason

    def test_a_non_str_handle_is_no_narrowing_and_is_rejected_by_the_option_check(self) -> None:
        feature = Feature(self.present_column, self._options(context={HANDLE_OPTION: ["own_handle"]}))
        assert not self._claims(feature, self.own_dac())

    def test_pointed_with_a_missing_column_aborts_naming_source_and_columns(self) -> None:
        feature = self._feature(self.missing_column, {self._group_name(): self.own_pointer()})
        with pytest.raises(ValueError) as exc_info:
            self._resolve(feature, None)
        message = str(exc_info.value)
        assert self.expected_source in message
        assert self.present_column in message

    def test_unpointed_missing_column_declines_with_a_rejection(self) -> None:
        result = self._evaluate(Feature(self.missing_column), self.own_dac())
        assert self.feature_group_class not in result.identified
        reason = result.eliminations[self.feature_group_class].reason
        assert self.expected_source in reason
        assert self.missing_column in reason

    def test_chain_and_column_separated_names_that_are_not_columns_decline(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        chained = f"{self.present_column}{CHAIN_SEPARATOR}toyfmt_rebased"
        separated = f"{self.missing_column}{COLUMN_SEPARATOR}0"
        assert not self._claims(Feature(chained), self.own_dac())
        assert not self.feature_group_class.match_feature_group_criteria(chained, self._options(), self.own_dac())
        reason = rejection_window[self._group_name()].reason
        assert self.expected_source in reason
        assert chained in reason
        assert not self._claims(Feature(separated), self.own_dac())

    def test_a_subclass_takes_over_a_pointer_keyed_on_its_parent(self) -> None:
        sub = self._gated_subclass()
        mapping: FeatureGroupEnvironmentMapping = {cast(type[FormatFeatureGroup], sub): {PyArrowTable}}
        feature = self._feature(self.present_column, {self.feature_group_class.__name__: self.own_pointer()})
        result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, None)
        assert sub in result.identified
        # Drop every reference (the feature's input_data_match pins the class) so the subclass can be collected.
        del sub, mapping, result, feature
        gc.collect()
