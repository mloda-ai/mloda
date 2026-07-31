"""R4: the MatchData two-readers conflict must stay loud at the match seam (#845 follow-up).

``MatchData.add_base_input_data_to_options`` raises the same "already set with different values" conflict
its ``BaseInputData`` twin already marks, and it is reachable from the match hook. The setup mirrors
``tests/test_plugins/feature_group/input_data/test_read.py::TestTwoReader``: one feature already carries a
feature-scope connection while the run also offers a global-scope one. Doubles are dropped per test.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


CONFLICT_FEATURE = "match_data_conflict_feat_845r"
CONFLICT_CLASS_NAME = "ConflictMatchDataFG845r"
RIVAL_CLASS_NAME = "MatchDataRivalFG845r"
FEATURE_SCOPE_ACCESS = "feature_scope_conn_845r"
GLOBAL_SCOPE_ACCESS = "global_scope_conn_845r"
CONFLICT_TEXT = "already set with different values"
RAISE_TYPE_NAME = "ValueError"

T = TypeVar("T")


class MatchDataFw845r(ComputeFramework):
    """Dummy compute framework for the MatchData conflict tests."""


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _make_conflicting_match_data_fg() -> type[FeatureGroup]:
    """Candidate whose global-scope access contradicts the feature-scope one already in the options."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class ConflictMatchDataFG845r(FeatureGroup, MatchData):
        """Declines the feature-scope connection, then resolves a different global-scope one."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {CONFLICT_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MatchDataFw845r}

        @classmethod
        def match_data_access(
            cls,
            feature_name: str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
            framework_connection_object: Optional[Any] = None,
        ) -> Any:
            if str(feature_name) != CONFLICT_FEATURE:
                return None
            # Only the global scope resolves, so the feature-scope value stays in the options unclaimed.
            if data_access_collection is None:
                return None
            return GLOBAL_SCOPE_ACCESS

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return ConflictMatchDataFG845r


def _make_rival_fg() -> type[FeatureGroup]:
    """Rival candidate claiming the same feature name cleanly, so a contained conflict would let it win."""
    gc.collect()

    class MatchDataRivalFG845r(FeatureGroup):
        """The group that would silently win while the two-readers misconfiguration is swallowed."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {CONFLICT_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MatchDataFw845r}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return MatchDataRivalFG845r


@dataclass(frozen=True)
class _ConflictSnapshot:
    """Plain-data readout of one evaluation. Holds no class and no exception object."""

    escaped: Optional[str]
    identified_names: tuple[str, ...]


def _evaluate_conflict(with_rival: bool) -> _ConflictSnapshot:
    """Evaluate the conflicting feature at the seam and read the outcome out as plain data."""
    conflict_fg = _make_conflicting_match_data_fg()
    rival_fg = _make_rival_fg() if with_rival else None
    try:
        options = Options(group={CONFLICT_CLASS_NAME: FEATURE_SCOPE_ACCESS})
        feature = Feature(CONFLICT_FEATURE, options=options)
        data_access = DataAccessCollection(connections={"match_data_handle_845r": GLOBAL_SCOPE_ACCESS})
        plugins: FeatureGroupEnvironmentMapping = {conflict_fg: {MatchDataFw845r}}
        if rival_fg is not None:
            plugins[rival_fg] = {MatchDataFw845r}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None, data_access))
        identified = () if result is None else tuple(sorted(fg.get_class_name() for fg in result.identified))
        snapshot = _ConflictSnapshot(escaped=escaped, identified_names=identified)
        del result
        del plugins
        del feature
        return snapshot
    finally:
        del conflict_fg
        del rival_fg
        gc.collect()


class TestMatchDataConflictAbortsTheMatch:
    """Two conflicting readers for one feature is a misconfiguration, not a non-match."""

    def test_conflict_reaches_the_caller(self) -> None:
        """The conflict ValueError must cross the match seam instead of becoming a matcher_error near-miss."""
        snapshot = _evaluate_conflict(with_rival=False)

        assert snapshot.escaped is not None, "the two-readers conflict must not be contained as a non-match"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: "), (
            f"the conflict's own ValueError must reach the caller, got: {snapshot.escaped}"
        )
        assert CONFLICT_TEXT in snapshot.escaped
        assert snapshot.identified_names == ()

    def test_conflict_is_not_dropped_when_a_rival_claims_the_name(self) -> None:
        """A rival matching the same name must not swallow the misconfiguration and win in its place."""
        snapshot = _evaluate_conflict(with_rival=True)

        assert snapshot.identified_names != (RIVAL_CLASS_NAME,), (
            "the rival must not silently win while the two-readers conflict is swallowed"
        )
        assert snapshot.escaped is not None, "a rival candidate must not hide the two-readers conflict"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert CONFLICT_TEXT in snapshot.escaped
