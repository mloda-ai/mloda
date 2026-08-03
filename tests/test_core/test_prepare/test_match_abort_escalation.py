"""os-005: a framework-owned raise escapes the match seam; an unmarked raise stays contained (#845).

The provenance marker must preserve the exception object exactly, because callers assert on its original
type at the matcher boundary. ``resolve_feature`` keeps its never-raises contract: a marked raise reaches
``ResolvedFeature.error`` instead of propagating. Doubles are dropped per test.

A mark is only worth anything if every handler BETWEEN the raise and the seam re-raises it. The two on the
normal path are covered here: ``match_parser_criteria`` and ``FeatureGroup.is_root``. ``utils.safe_field``
is the deliberate exception: it degrades one field in a rendering path and swallows a marked exception.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Optional, TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import escalate_match_abort, is_match_abort
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.steward import resolve_feature


MARKED_MESSAGE = "boom_845e_framework_owned_raise"
UNMARKED_MESSAGE = "boom_845e_plugin_owned_raise"
MARKED_FEATURE = "match_abort_marked_feat_845e"
UNMARKED_FEATURE = "match_abort_unmarked_feat_845e"
MARKED_CLASS_NAME = "MarkedRaiseFG845e"
UNMARKED_CLASS_NAME = "UnmarkedRaiseFG845e"
RAISE_TYPE_NAME = "ValueError"
MATCHER_ERROR_STAGE = "matcher_error"

# Name-bound probe for the parser's own marked raise: the key is a named capture of the pattern.
BINDING_KEY = "op_845f_both_categories"
BINDING_PATTERN = r".*__(?P<op_845f_both_categories>\w+)_845f$"
BINDING_FEATURE = "src845f__alpha_845f"
# Same pattern, nothing before the separator: parse_name's own UNMARKED "no source feature" ValueError.
NO_SOURCE_FEATURE = "__alpha_845f"
BOTH_CATEGORIES_FRAGMENT = "exists in both group and context"
IS_ROOT_MARKED_MESSAGE = "boom_845f_marked_input_features"
IS_ROOT_UNMARKED_MESSAGE = "boom_845f_unmarked_input_features"

T = TypeVar("T")


class MatchAbortFw845e(ComputeFramework):
    """Dummy compute framework for the match-abort escalation tests."""


@dataclass(frozen=True)
class _RaiseReadout:
    """Plain-data readout of an escaping raise. Holds no exception object and no traceback."""

    marked: bool
    type_name: str
    message: str


def _outcome(call: Callable[[], T]) -> tuple[Optional[T], Optional[_RaiseReadout]]:
    """Run call, returning (value, None) or (None, readout of the escaping raise)."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, _RaiseReadout(is_match_abort(exc), type(exc).__name__, str(exc))


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    value, readout = _outcome(call)
    return value, None if readout is None else f"{readout.type_name}: {readout.message}"


def _make_marked_raise_fg() -> type[FeatureGroup]:
    """Candidate whose matcher raises a MARKED ValueError for its own feature name."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class MarkedRaiseFG845e(FeatureGroup):
        """Stands in for framework-owned code raising inside the match hook."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {MARKED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MatchAbortFw845e}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if str(feature_name) != MARKED_FEATURE:
                return False
            raise escalate_match_abort(ValueError(MARKED_MESSAGE))

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return MarkedRaiseFG845e


def _make_unmarked_raise_fg() -> type[FeatureGroup]:
    """Twin of the marked double, raising the same ValueError type WITHOUT the marker."""
    gc.collect()

    class UnmarkedRaiseFG845e(FeatureGroup):
        """Stands in for a plugin matcher that simply breaks: contained as a non-match (#845)."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {UNMARKED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MatchAbortFw845e}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if str(feature_name) != UNMARKED_FEATURE:
                return False
            raise ValueError(UNMARKED_MESSAGE)

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return UnmarkedRaiseFG845e


@dataclass(frozen=True)
class _ContainedSnapshot:
    """Plain-data readout of one contained evaluation. Holds no class and no exception object."""

    escaped: Optional[str]
    failure_kind: Optional[str]
    eliminated_names: tuple[str, ...]
    stage: Optional[str]
    reason: Optional[str]


def _evaluate_marked_raise() -> Optional[str]:
    """Evaluate the marked double's own feature; return the escaping raise as 'Type: message', else None."""
    broken_fg = _make_marked_raise_fg()
    try:
        feature = Feature(MARKED_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {broken_fg: {MatchAbortFw845e}}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        del result
        del plugins
        return escaped
    finally:
        del broken_fg
        gc.collect()


def _evaluate_unmarked_raise() -> _ContainedSnapshot:
    """Evaluate the unmarked double's own feature and read the containment out as plain data."""
    broken_fg = _make_unmarked_raise_fg()
    try:
        feature = Feature(UNMARKED_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {broken_fg: {MatchAbortFw845e}}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        if result is None:
            return _ContainedSnapshot(escaped, None, (), None, None)

        elimination = result.eliminations.get(broken_fg)
        snapshot = _ContainedSnapshot(
            escaped=None,
            failure_kind=result.failure_kind,
            eliminated_names=tuple(sorted(fg.get_class_name() for fg in result.eliminations)),
            stage=None if elimination is None else str(elimination.stage),
            reason=None if elimination is None else str(elimination.reason),
        )
        del elimination
        del result
        del plugins
        return snapshot
    finally:
        del broken_fg
        gc.collect()


class TestEscalateMatchAbortPreservesTheException:
    """The marker is provenance only: the exception object, its type, message and args stay untouched."""

    def test_escalate_returns_the_same_object_and_marks_it(self) -> None:
        """escalate_match_abort marks in place: same identity, same type, same message, same args."""
        exc = ValueError(MARKED_MESSAGE)

        marked = escalate_match_abort(exc)

        assert marked is exc
        assert type(marked) is ValueError
        assert str(marked) == MARKED_MESSAGE
        assert marked.args == (MARKED_MESSAGE,)
        assert is_match_abort(marked) is True

    def test_escalate_preserves_a_keyerror_unchanged(self) -> None:
        """A KeyError stays a KeyError with its own str()/args, so pytest.raises(KeyError, ...) still holds."""
        exc = KeyError(MARKED_MESSAGE)

        marked = escalate_match_abort(exc)

        assert marked is exc
        assert type(marked) is KeyError
        assert str(marked) == str(KeyError(MARKED_MESSAGE))
        assert marked.args == (MARKED_MESSAGE,)
        assert is_match_abort(marked) is True

    def test_unmarked_exception_is_not_a_match_abort(self) -> None:
        """An ordinary exception is unmarked, so the seam keeps containing it."""
        assert is_match_abort(ValueError(UNMARKED_MESSAGE)) is False


class TestMatchAbortCrossesTheMatchSeam:
    """A marked raise escapes evaluate(); an unmarked one is still contained as a matcher_error near-miss."""

    def test_marked_matcher_raise_propagates_out_of_evaluate(self) -> None:
        """The seam re-raises the marked exception with its original type and message."""
        escaped = _evaluate_marked_raise()

        assert escaped == f"{RAISE_TYPE_NAME}: {MARKED_MESSAGE}", (
            "a framework-owned raise must cross the match seam unchanged, not be contained as a non-match"
        )

    def test_unmarked_matcher_raise_stays_contained(self) -> None:
        """The #845 containment is unchanged for an unmarked raise: skipped, recorded as matcher_error."""
        snapshot = _evaluate_unmarked_raise()

        assert snapshot.escaped is None
        assert snapshot.failure_kind == "none"
        assert snapshot.eliminated_names == (UNMARKED_CLASS_NAME,)
        assert snapshot.stage == MATCHER_ERROR_STAGE
        assert snapshot.reason is not None
        assert RAISE_TYPE_NAME in snapshot.reason
        assert UNMARKED_MESSAGE in snapshot.reason


class BothCategoriesMixin845f(FeatureChainParserMixin):
    """Probe for the marked both-categories raise; a mixin, so it joins no other test's candidate universe."""

    PREFIX_PATTERN = BINDING_PATTERN
    # Annotated exactly as FeatureGroup declares it, so the FeatureGroup subclass below stays consistent.
    PROPERTY_MAPPING: ClassVar[Optional[dict[str, PropertySpec]]] = {
        BINDING_KEY: property_spec("the key the feature name carries")
    }


def _both_categories_options() -> Options:
    """Options holding BINDING_KEY in group AND context, the shape _determine_parameter_category rejects."""
    # Options() itself rejects a key in both categories, so this state needs a write to the second dict.
    options = Options(group={BINDING_KEY: None})
    options.context[BINDING_KEY] = "from_context_845f"
    # The group value is None so the merge reads the key as absent and asks which category it belongs in.
    return options


def _make_both_categories_fg() -> type[FeatureGroup]:
    """Candidate whose inherited chain-parser matcher hits the marked both-categories raise."""
    gc.collect()

    class BothCategoriesFG845f(BothCategoriesMixin845f, FeatureGroup):
        """Stands in for a normal chain-parser group asked to match a contradictory option set."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MatchAbortFw845e}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return BothCategoriesFG845f


def _evaluate_both_categories() -> Optional[_RaiseReadout]:
    """Evaluate the both-categories candidate through the seam; read any escaping raise out as plain data."""
    probe_fg = _make_both_categories_fg()
    try:
        feature = Feature(BINDING_FEATURE, _both_categories_options())
        plugins: FeatureGroupEnvironmentMapping = {probe_fg: {MatchAbortFw845e}}
        result, readout = _outcome(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        del result
        del plugins
        return readout
    finally:
        del probe_fg
        gc.collect()


def _make_is_root_fg(raiser: Callable[[], None]) -> type[FeatureGroup]:
    """Candidate whose input_features raises whatever ``raiser`` raises."""
    gc.collect()

    class IsRootProbeFG845f(FeatureGroup):
        """Stands in for a group whose input_features cannot answer for this feature name."""

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            raiser()
            return None

    return IsRootProbeFG845f


def _is_root_outcome(raiser: Callable[[], None]) -> tuple[Optional[bool], Optional[_RaiseReadout]]:
    """(verdict, None) when is_root answers, (None, readout) when the raise escapes it."""
    probe_fg = _make_is_root_fg(raiser)
    try:
        instance = probe_fg()
        verdict, readout = _outcome(partial(instance.is_root, Options(), "is_root_probe_845f"))
        del instance
        return verdict, readout
    finally:
        del probe_fg
        gc.collect()


def _raise_marked_value_error() -> None:
    raise escalate_match_abort(ValueError(IS_ROOT_MARKED_MESSAGE))


def _raise_unmarked_value_error() -> None:
    raise ValueError(IS_ROOT_UNMARKED_MESSAGE)


def _raise_not_implemented() -> None:
    raise NotImplementedError


class TestMarkedRaiseSurvivesTheChainParserContainment:
    """match_parser_criteria contains a parser error as a non-match, but never a MARKED one."""

    def test_marked_config_error_escapes_match_parser_criteria(self) -> None:
        """The both-categories raise is a framework invariant break: it must not read as a non-match."""
        verdict, readout = _outcome(
            partial(BothCategoriesMixin845f.match_parser_criteria, BINDING_FEATURE, _both_categories_options())
        )

        assert verdict is None, "a marked raise was swallowed into a match verdict by match_parser_criteria"
        assert readout is not None
        assert readout.marked is True
        assert readout.type_name == RAISE_TYPE_NAME
        assert BINDING_KEY in readout.message
        assert BOTH_CATEGORIES_FRAGMENT in readout.message

    def test_marked_config_error_escapes_match_feature_group_criteria(self) -> None:
        """The mark survives the matcher the engine actually calls, not just the parser helper."""
        verdict, readout = _outcome(
            partial(BothCategoriesMixin845f.match_feature_group_criteria, BINDING_FEATURE, _both_categories_options())
        )

        assert verdict is None, "a marked raise was swallowed into a match verdict by match_feature_group_criteria"
        assert readout is not None
        assert readout.marked is True
        assert readout.type_name == RAISE_TYPE_NAME
        assert BOTH_CATEGORIES_FRAGMENT in readout.message

    def test_unmarked_parser_error_is_still_a_non_match(self) -> None:
        """Containment stays the default: parse_name's own unmarked ValueError is a non-match, not a raise."""
        verdict, readout = _outcome(
            partial(BothCategoriesMixin845f.match_parser_criteria, NO_SOURCE_FEATURE, Options())
        )

        assert readout is None, "an unmarked parser error must stay contained as a non-match"
        assert verdict is False

    def test_marked_config_error_crosses_the_match_seam(self) -> None:
        """Reading the mark back at evaluate()'s caller shows that one object crossed, unwrapped."""
        readout = _evaluate_both_categories()

        assert readout is not None, "the marked raise never reached the seam; a handler on the way swallowed it"
        assert readout.marked is True
        assert readout.type_name == RAISE_TYPE_NAME
        assert BINDING_KEY in readout.message
        assert BOTH_CATEGORIES_FRAGMENT in readout.message


class TestIsRootRespectsTheMark:
    """is_root reads a raising input_features as "not a root", except when the raise is marked."""

    def test_marked_input_features_raise_escapes_is_root(self) -> None:
        """A framework-owned raise must not be downgraded into a root verdict."""
        verdict, readout = _is_root_outcome(_raise_marked_value_error)

        assert verdict is None, "a marked raise was swallowed into an is_root verdict"
        assert readout is not None
        assert readout.marked is True
        assert readout.type_name == RAISE_TYPE_NAME
        assert readout.message == IS_ROOT_MARKED_MESSAGE

    def test_unmarked_input_features_raise_still_means_not_root(self) -> None:
        """Unchanged: an unmarked failure in input_features means this group does not match."""
        verdict, readout = _is_root_outcome(_raise_unmarked_value_error)

        assert readout is None, "an unmarked raise out of input_features must stay contained"
        assert verdict is False

    def test_not_implemented_input_features_still_means_root(self) -> None:
        """Unchanged: an unimplemented input_features is the documented way to declare a root feature."""
        verdict, readout = _is_root_outcome(_raise_not_implemented)

        assert readout is None
        assert verdict is True


class TestResolveFeatureStillNeverRaises:
    """The debug path degrades the escaping raise into ResolvedFeature.error instead of propagating."""

    def test_marked_raise_reaches_resolve_feature_error(self) -> None:
        """resolve_feature reports the marked message as its error, with no winner and no no-match text."""
        marked_fg = _make_marked_raise_fg()
        try:
            result = resolve_feature(MARKED_FEATURE)
            winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
            error = result.error
            del result
        finally:
            del marked_fg
            gc.collect()

        assert winner_name is None
        assert error is not None
        assert MARKED_MESSAGE in error
        assert "No feature groups found" not in error, (
            "a framework-owned raise must not be converted into the standard no-match error"
        )
