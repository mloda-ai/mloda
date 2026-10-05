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
from typing import Any, TypeVar

import pytest

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


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
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
            data_access_collection: DataAccessCollection | None = None,
            framework_connection_object: Any | None = None,
        ) -> Any:
            if str(feature_name) != CONFLICT_FEATURE:
                return None
            # Only the global scope resolves, so the feature-scope value stays in the options unclaimed.
            if data_access_collection is None:
                return None
            return GLOBAL_SCOPE_ACCESS

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return MatchDataRivalFG845r


@dataclass(frozen=True)
class _ConflictSnapshot:
    """Plain-data readout of one evaluation. Holds no class and no exception object."""

    escaped: str | None
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


SAME_NAME_FEATURE = "same_name_match_data_feat_845s"
SAME_NAME_CLASS = "SameNameMatchDataFG845s"
SAME_NAME_MODULES = ("same_name_module_one_845s", "same_name_module_two_845s")
ACCESS_ONE = "same_name_access_one_845s"
ACCESS_TWO = "same_name_access_two_845s"


class _NonBoolEq845s:
    """Expression-style value whose ``__eq__`` returns a non-bool."""

    def __eq__(self, other: object) -> Any:
        return ("expr", id(other))

    __hash__ = object.__hash__


def _make_same_name_fg(module: str, access: Any, feature_scope: bool = False) -> type[FeatureGroup]:
    """MatchData group named SAME_NAME_CLASS in the given module, resolving access in global scope only.

    With feature_scope, it instead claims the feature-scope connection object and ignores the global scope.
    """
    gc.collect()

    class SameNameMatchDataFG845s(FeatureGroup, MatchData):
        __module__ = module

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {SAME_NAME_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MatchDataFw845r}

        @classmethod
        def match_data_access(
            cls,
            feature_name: str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
            framework_connection_object: Any | None = None,
        ) -> Any:
            if str(feature_name) != SAME_NAME_FEATURE:
                return None
            if feature_scope:
                return framework_connection_object if data_access_collection is None else None
            return access if data_access_collection is not None else None

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return SameNameMatchDataFG845s


@dataclass(frozen=True)
class _SameNameSnapshot:
    escaped: str | None
    identified_count: int
    group_untouched: bool


def _evaluate_same_name(
    accesses: tuple[str, str], reverse: bool, scope_value: Any = None, use_scope: bool = False
) -> _SameNameSnapshot:
    """Evaluate two same-name groups with the given global-scope accesses, optionally in reverse order.

    With use_scope, the request carries scope_value under SAME_NAME_CLASS and both groups match it via feature scope.
    """
    fg_one = _make_same_name_fg(SAME_NAME_MODULES[0], accesses[0], use_scope)
    fg_two = _make_same_name_fg(SAME_NAME_MODULES[1], accesses[1], use_scope)
    try:
        original_group: dict[str, Any] = {SAME_NAME_CLASS: scope_value} if use_scope else {}
        feature = Feature(SAME_NAME_FEATURE, options=Options(group=dict(original_group)))
        data_access: DataAccessCollection | None = (
            None if use_scope else DataAccessCollection(connections={"h1": ACCESS_ONE, "h2": ACCESS_TWO})
        )
        ordered = [fg_two, fg_one] if reverse else [fg_one, fg_two]
        plugins: FeatureGroupEnvironmentMapping = {fg: {MatchDataFw845r} for fg in ordered}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None, data_access))
        count = 0 if result is None else len(result.identified)
        group = feature.options.group
        if use_scope:
            untouched = list(group) == [SAME_NAME_CLASS] and group[SAME_NAME_CLASS] is scope_value
        else:
            untouched = group == original_group and SAME_NAME_CLASS not in group
        snapshot = _SameNameSnapshot(escaped=escaped, identified_count=count, group_untouched=untouched)
        del result
        del plugins
        del feature
        return snapshot
    finally:
        del fg_one
        del fg_two
        gc.collect()


class TestMatchDataConflictBetweenSurvivors:
    """Same-named MatchData survivors with different global accesses must raise the conflict."""

    @pytest.mark.parametrize("reverse", [False, True])
    def test_different_accesses_escape_with_conflict(self, reverse: bool) -> None:
        snapshot = _evaluate_same_name((ACCESS_ONE, ACCESS_TWO), reverse)

        assert snapshot.escaped is not None, "different accesses under one class name must conflict"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert f"{SAME_NAME_CLASS} {CONFLICT_TEXT}" in snapshot.escaped
        assert snapshot.identified_count == 0
        assert ACCESS_ONE not in snapshot.escaped
        assert ACCESS_TWO not in snapshot.escaped
        assert snapshot.group_untouched

    @pytest.mark.parametrize("reverse", [False, True])
    def test_equal_accesses_keep_both_survivors(self, reverse: bool) -> None:
        snapshot = _evaluate_same_name((ACCESS_ONE, ACCESS_ONE), reverse)

        assert snapshot.escaped is None
        assert snapshot.identified_count == 2
        assert snapshot.group_untouched

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("scope_value", [_NonBoolEq845s(), float("nan")], ids=["non_bool_eq", "nan"])
    def test_feature_scope_value_self_compare_keeps_both_survivors(self, reverse: bool, scope_value: Any) -> None:
        snapshot = _evaluate_same_name((ACCESS_ONE, ACCESS_ONE), reverse, scope_value, use_scope=True)

        assert snapshot.escaped is None
        assert snapshot.identified_count == 2
        assert snapshot.group_untouched
