"""os-005: a framework-owned raise escapes the match seam; an unmarked raise stays contained (#845).

The provenance marker must preserve the exception object exactly, because callers assert on its original
type at the matcher boundary. ``resolve_feature`` keeps its never-raises contract: a marked raise reaches
``ResolvedFeature.error`` instead of propagating. Doubles are dropped per test.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Optional, TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
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

T = TypeVar("T")


class MatchAbortFw845e(ComputeFramework):
    """Dummy compute framework for the match-abort escalation tests."""


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


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
