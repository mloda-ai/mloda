"""R3: a contained matcher raise must leave no partial mutation on the shared Feature.options (#845 follow-up).

``feature.options`` is ONE mutable object shared by every candidate's match hook, and a matcher that
returns True must keep its write (that is how a matched reader is linked through mloda), so the rollback
is per candidate, not a whole-loop snapshot. Doubles carry an ``845`` suffix and are dropped per test.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


SHARED_FEATURE = "shared_option_isolation_feat_845r"
SIDE_EFFECT_KEY = "side_effect_845r"
SIDE_EFFECT_VALUE = "written_by_the_raising_candidate_845r"
LINKED_KEY = "linked_reader_845r"
LINKED_VALUE = "written_by_the_matching_candidate_845r"
RAISE_MESSAGE = "boom_845r_matcher_raised_after_writing"
RAISING_CLASS_NAME = "MutatingRaiseFG845r"
CLEAN_CLASS_NAME = "CleanNameOwnerFG845r"
LINKING_CLASS_NAME = "MutatingMatchFG845r"

T = TypeVar("T")


class OptionIsolationFw845r(ComputeFramework):
    """Dummy compute framework for the option-isolation tests."""


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _make_mutating_raise_fg() -> type[FeatureGroup]:
    """Candidate that writes to the shared options and then raises an UNMARKED exception."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class MutatingRaiseFG845r(FeatureGroup):
        """Stands in for a plugin matcher that configures the feature before it breaks."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {SHARED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionIsolationFw845r}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) != SHARED_FEATURE:
                return False
            options.add_to_group(SIDE_EFFECT_KEY, SIDE_EFFECT_VALUE)
            raise RuntimeError(RAISE_MESSAGE)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return MutatingRaiseFG845r


def _make_clean_owner_fg() -> type[FeatureGroup]:
    """Rival candidate that claims the same feature name cleanly and writes nothing."""
    gc.collect()

    class CleanNameOwnerFG845r(FeatureGroup):
        """The winner whose options must not carry the eliminated candidate's write."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {SHARED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionIsolationFw845r}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return CleanNameOwnerFG845r


def _make_mutating_match_fg() -> type[FeatureGroup]:
    """Candidate that writes to the shared options and RETURNS TRUE, the linked-reader pattern."""
    gc.collect()

    class MutatingMatchFG845r(FeatureGroup):
        """Stands in for the match hook that links its matched reader through the shared options."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {SHARED_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionIsolationFw845r}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) != SHARED_FEATURE:
                return False
            if LINKED_KEY not in options:
                options.add_to_group(LINKED_KEY, LINKED_VALUE)
            return True

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return MutatingMatchFG845r


@dataclass(frozen=True)
class _OptionsSnapshot:
    """Plain-data readout of one evaluation. Holds no class and no exception object."""

    escaped: str | None
    identified_names: tuple[str, ...]
    option_keys: tuple[str, ...]
    side_effect_value: str | None
    linked_value: str | None


def _evaluate(builders: tuple[Callable[[], type[FeatureGroup]], ...]) -> _OptionsSnapshot:
    """Evaluate the shared feature against the given candidates, in order, and read the options out."""
    candidates = [build() for build in builders]
    try:
        feature = Feature(SHARED_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {candidate: {OptionIsolationFw845r} for candidate in candidates}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        identified = () if result is None else tuple(sorted(fg.get_class_name() for fg in result.identified))
        snapshot = _OptionsSnapshot(
            escaped=escaped,
            identified_names=identified,
            option_keys=tuple(sorted(str(key) for key in feature.options.keys())),
            side_effect_value=feature.options.get(SIDE_EFFECT_KEY),
            linked_value=feature.options.get(LINKED_KEY),
        )
        del result
        del plugins
        del feature
        return snapshot
    finally:
        candidates.clear()
        del candidates
        gc.collect()


class TestContainedRaiseLeavesNoPartialMutation:
    """An eliminated candidate must not configure the feature that won."""

    def test_write_before_a_contained_raise_is_rolled_back(self) -> None:
        """The raising candidate's write must not survive into the winning group's options."""
        snapshot = _evaluate((_make_mutating_raise_fg, _make_clean_owner_fg))

        assert snapshot.escaped is None
        assert snapshot.identified_names == (CLEAN_CLASS_NAME,)
        assert snapshot.side_effect_value is None, (
            f"a contained raise must leave no partial mutation, found {SIDE_EFFECT_KEY}={snapshot.side_effect_value}"
        )
        assert SIDE_EFFECT_KEY not in snapshot.option_keys


class TestMatchingMatcherKeepsItsMutation:
    """The load-bearing counterpart: a matcher that returns True keeps what it wrote."""

    def test_write_from_a_matching_candidate_survives(self) -> None:
        """This is how a matched reader is linked through mloda, so the rollback must not touch it."""
        snapshot = _evaluate((_make_mutating_match_fg,))

        assert snapshot.escaped is None
        assert snapshot.identified_names == (LINKING_CLASS_NAME,)
        assert snapshot.linked_value == LINKED_VALUE
        assert LINKED_KEY in snapshot.option_keys


class TestRollbackIsPerCandidate:
    """A whole-loop snapshot would also undo an earlier winner's write, so the rollback is per candidate."""

    def test_earlier_matching_write_survives_a_later_contained_raise(self) -> None:
        """The matching candidate runs first and keeps its write; only the later raising one is rolled back."""
        snapshot = _evaluate((_make_mutating_match_fg, _make_mutating_raise_fg))

        assert snapshot.escaped is None
        assert snapshot.identified_names == (LINKING_CLASS_NAME,)
        assert snapshot.linked_value == LINKED_VALUE, "a later contained raise must not undo an earlier match's write"
        assert snapshot.side_effect_value is None, (
            f"a contained raise must leave no partial mutation, found {SIDE_EFFECT_KEY}={snapshot.side_effect_value}"
        )
