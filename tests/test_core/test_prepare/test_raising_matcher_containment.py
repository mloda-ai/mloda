"""Red-phase pins for issue #845 (part 2): a raising ``match_feature_group_criteria`` must be contained.

A raising matcher must be skipped so every other feature still resolves, and on the broken class's own
feature name it is recorded as a ``matcher_error`` near-miss whose reason is plain text, so no retained
traceback pins the class. The build-phase fail-closed contract (#790) is out of scope; it is pinned in
tests/test_core/test_api/test_sbdg_resolve_feature_broken_rule.py. Doubles are dropped per test.
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
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    IdentifyFeatureGroupClass,
    render_resolution_failure,
)
from mloda.steward import resolve_feature


RAISE_MESSAGE = "boom_845_matcher_exploded"
RAISE_TYPE_NAME = "RuntimeError"
BROKEN_CLASS_NAME = "RaisingMatcherFG845"
BROKEN_OWN_FEATURE = "raising_matcher_own_feat_845"
NEIGHBOR_CLASS_NAME = "ContainedNeighborFG845"
NEIGHBOR_FEATURE = "contained_neighbor_feat_845"
# Deliberately dissimilar, so no "Did you mean" suggestion can name the broken class.
UNRELATED_FEATURE = "zqx_no_group_owns_this_845"
MATCHER_ERROR_STAGE = "matcher_error"

T = TypeVar("T")


class ContainedFw845(ComputeFramework):
    """Dummy compute framework for the raising-matcher containment tests."""


class ContainedNeighborFG845(FeatureGroup):
    """Inert, resolvable neighbour: it must keep resolving while a broken candidate exists."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {NEIGHBOR_FEATURE}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {ContainedFw845}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _make_raising_matcher_fg() -> type[FeatureGroup]:
    """Build a candidate whose matcher raises, while still declaring its own feature name."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class RaisingMatcherFG845(FeatureGroup):
        """Declares one feature name, but its criteria hook raises for every request."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {BROKEN_OWN_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {ContainedFw845}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            raise RuntimeError(RAISE_MESSAGE)

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return RaisingMatcherFG845


@dataclass(frozen=True)
class _OwnFeatureSnapshot:
    """Plain-data readout of one own-feature evaluation. Holds no class and no exception object."""

    escaped: Optional[str]
    failure_kind: Optional[str]
    eliminated_names: tuple[str, ...]
    stage: Optional[str]
    reason: Optional[str]
    reason_type: Optional[str]
    message: Optional[str]


def _evaluate_own_feature() -> _OwnFeatureSnapshot:
    """Evaluate the broken class's OWN feature name and read the result out as plain data."""
    broken_fg = _make_raising_matcher_fg()
    try:
        feature = Feature(BROKEN_OWN_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {broken_fg: {ContainedFw845}}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        if result is None:
            return _OwnFeatureSnapshot(escaped, None, (), None, None, None, None)

        elimination = result.eliminations.get(broken_fg)
        message, render_escaped = _capture(partial(render_resolution_failure, result, feature))
        snapshot = _OwnFeatureSnapshot(
            escaped=render_escaped,
            failure_kind=result.failure_kind,
            eliminated_names=tuple(sorted(fg.get_class_name() for fg in result.eliminations)),
            stage=None if elimination is None else str(elimination.stage),
            reason=None if elimination is None else str(elimination.reason),
            reason_type=None if elimination is None else type(elimination.reason).__name__,
            message=message,
        )
        del elimination
        del result
        del plugins
        return snapshot
    finally:
        del broken_fg
        gc.collect()


class TestRaisingMatcherContainment:
    """A raising matcher is a contained non-match, recorded as a matcher_error near-miss on its own feature."""

    def test_unrelated_feature_is_not_poisoned(self) -> None:
        """resolve_feature of an unrelated name reports the ordinary no-match, never the bare raise."""
        broken_fg = _make_raising_matcher_fg()
        try:
            result = resolve_feature(UNRELATED_FEATURE)
            winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
            error = result.error
            candidate_names = [candidate.get_class_name() for candidate in result.candidates]
            del result
        finally:
            del broken_fg
            gc.collect()

        assert winner_name is None
        assert candidate_names == []
        assert error is not None
        # The ordinary no-match, not the bare raise. The broken candidate may still appear as a near-miss.
        assert error != RAISE_MESSAGE
        assert error.startswith(f"No feature groups found for feature name: '{UNRELATED_FEATURE}'.")

    def test_resolvable_neighbour_still_resolves(self) -> None:
        """A resolvable unrelated feature still wins its own group while the broken class exists."""
        broken_fg = _make_raising_matcher_fg()
        try:
            result = resolve_feature(NEIGHBOR_FEATURE)
            winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
            error = result.error
            del result
        finally:
            del broken_fg
            gc.collect()

        assert error is None
        assert winner_name == NEIGHBOR_CLASS_NAME

    def test_evaluate_skips_the_raising_candidate(self) -> None:
        """The seam does not propagate the raise: the broken candidate is skipped, the neighbour wins."""
        broken_fg = _make_raising_matcher_fg()
        try:
            feature = Feature(NEIGHBOR_FEATURE)
            plugins: FeatureGroupEnvironmentMapping = {
                broken_fg: {ContainedFw845},
                ContainedNeighborFG845: {ContainedFw845},
            }
            result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
            identified_names = [] if result is None else sorted(fg.get_class_name() for fg in result.identified)
            matched_names = [] if result is None else sorted(fg.get_class_name() for fg in result.criteria_matched)
            del result
            del plugins
        finally:
            del broken_fg
            gc.collect()

        assert escaped is None
        assert identified_names == [NEIGHBOR_CLASS_NAME]
        assert matched_names == [NEIGHBOR_CLASS_NAME]
        assert BROKEN_CLASS_NAME not in identified_names
        assert BROKEN_CLASS_NAME not in matched_names

    def test_own_feature_records_a_matcher_error_elimination(self) -> None:
        """Requesting the broken class's own name records it as a matcher_error near-miss."""
        snapshot = _evaluate_own_feature()

        assert snapshot.escaped is None
        assert snapshot.failure_kind == "none"
        assert snapshot.eliminated_names == (BROKEN_CLASS_NAME,)
        assert snapshot.stage == MATCHER_ERROR_STAGE
        assert snapshot.reason is not None
        assert RAISE_TYPE_NAME in snapshot.reason
        assert RAISE_MESSAGE in snapshot.reason

    def test_near_miss_block_names_the_broken_class(self) -> None:
        """render_resolution_failure names the broken class, its exception type and its message."""
        snapshot = _evaluate_own_feature()

        assert snapshot.escaped is None
        assert snapshot.message is not None
        message = snapshot.message
        assert f"Feature group(s) eliminated while matching '{BROKEN_OWN_FEATURE}':" in message
        bullets = [line for line in message.split("\n") if line.startswith(f"  - {BROKEN_CLASS_NAME} (")]
        assert len(bullets) == 1
        assert RAISE_TYPE_NAME in bullets[0]
        assert RAISE_MESSAGE in bullets[0]

    def test_recorded_reason_is_a_plain_string(self) -> None:
        """The contained raise is recorded as text: no exception object (and no traceback) is retained."""
        snapshot = _evaluate_own_feature()

        assert snapshot.escaped is None
        assert snapshot.stage == MATCHER_ERROR_STAGE
        assert snapshot.reason_type == "str"

    def test_resolve_feature_reports_the_matcher_error_for_the_own_feature(self) -> None:
        """The debug path agrees with the seam: same near-miss, not a bare provider crash message."""
        broken_fg = _make_raising_matcher_fg()
        try:
            result = resolve_feature(BROKEN_OWN_FEATURE)
            winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
            error = result.error
            del result
        finally:
            del broken_fg
            gc.collect()

        assert winner_name is None
        assert error is not None
        assert error.startswith(f"No feature groups found for feature name: '{BROKEN_OWN_FEATURE}'.")
        assert BROKEN_CLASS_NAME in error
        assert RAISE_TYPE_NAME in error
        assert RAISE_MESSAGE in error
