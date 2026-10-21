"""An option-write conflict during MatchData selection escalates instead of being contained as a non-match.

``MatchData.add_base_input_data_to_options`` decides on presence, and reports class and type names only.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


MATCH_DATA_FEATURE = "option_write_match_data_feat_932"
MATCH_DATA_CLASS_NAME = "OptionWriteMatchDataFG932"
MATCH_DATA_RIVAL_NAME = "OptionWriteMatchDataRivalFG932"
GLOBAL_ACCESS = "option_write_global_access_932"
RAISE_TYPE_NAME = "ValueError"
# Pins WHICH raise escaped: Options.add_to_group raises a ValueError naming the key too.
CONFLICT_TEXT = "already set with different values"

T = TypeVar("T")


class OptionWriteFw932(ComputeFramework):
    pass


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Returns (value, None) or (None, 'Type: message'); no traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class _ConflictSnapshot:
    """Readout of one evaluation, holding no class and no exception object."""

    escaped: str | None
    identified_names: tuple[str, ...]


def _make_match_data_fg() -> type[FeatureGroup]:
    """MatchData candidate resolving a global access while its class-name key is present as None."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class OptionWriteMatchDataFG932(FeatureGroup, MatchData):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {MATCH_DATA_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionWriteFw932}

        @classmethod
        def match_data_access(
            cls,
            feature_name: str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
            framework_connection_object: Any | None = None,
        ) -> Any:
            if str(feature_name) != MATCH_DATA_FEATURE:
                return None
            if data_access_collection is None:
                return None
            return GLOBAL_ACCESS

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return OptionWriteMatchDataFG932


def _make_match_data_rival_fg() -> type[FeatureGroup]:
    """Rival claiming the MatchData feature name cleanly, so a contained conflict would let it win."""
    gc.collect()

    class OptionWriteMatchDataRivalFG932(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {MATCH_DATA_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionWriteFw932}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return OptionWriteMatchDataRivalFG932


def _evaluate(
    feature_name: str,
    options: Options,
    conflict_fg: type[FeatureGroup],
    rival_fg: type[FeatureGroup] | None,
    data_access: DataAccessCollection | None,
) -> _ConflictSnapshot:
    """Evaluates one feature at the match seam and reads the outcome out as plain data."""
    feature = Feature(feature_name, options=options)
    plugins: FeatureGroupEnvironmentMapping = {conflict_fg: {OptionWriteFw932}}
    if rival_fg is not None:
        plugins[rival_fg] = {OptionWriteFw932}
    result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None, data_access))
    identified = () if result is None else tuple(sorted(fg.get_class_name() for fg in result.identified))
    snapshot = _ConflictSnapshot(escaped=escaped, identified_names=identified)
    del result
    del plugins
    del feature
    return snapshot


def _evaluate_match_data_conflict(with_rival: bool) -> _ConflictSnapshot:
    """MatchData twin: class-name key present as None while a global access resolves."""
    conflict_fg = _make_match_data_fg()
    rival_fg = _make_match_data_rival_fg() if with_rival else None
    try:
        options = Options(group={MATCH_DATA_CLASS_NAME: None})
        data_access = DataAccessCollection(connections={"option_write_handle_932": GLOBAL_ACCESS})
        return _evaluate(MATCH_DATA_FEATURE, options, conflict_fg, rival_fg, data_access)
    finally:
        del conflict_fg
        del rival_fg
        gc.collect()


class TestMatchDataOptionWriteConflictAbortsTheMatch:
    """A contradicting option write under MatchData is a user contradiction, not a non-match."""

    def test_conflict_reaches_the_caller(self) -> None:
        snapshot = _evaluate_match_data_conflict(with_rival=False)

        assert snapshot.escaped is not None
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert MATCH_DATA_CLASS_NAME in snapshot.escaped
        assert CONFLICT_TEXT in snapshot.escaped, "the twin's own raise must escape, not the deeper option write"
        assert snapshot.identified_names == ()

    def test_conflict_is_not_dropped_when_a_rival_claims_the_name(self) -> None:
        snapshot = _evaluate_match_data_conflict(with_rival=True)

        assert snapshot.identified_names != (MATCH_DATA_RIVAL_NAME,)
        assert snapshot.escaped is not None
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert MATCH_DATA_CLASS_NAME in snapshot.escaped
        assert CONFLICT_TEXT in snapshot.escaped

    def test_conflict_message_names_no_access_value(self) -> None:
        snapshot = _evaluate_match_data_conflict(with_rival=False)

        assert snapshot.escaped is not None
        assert GLOBAL_ACCESS not in snapshot.escaped
