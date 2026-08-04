"""#932: an option-write conflict during reader selection must escalate, like the reader conflict above it.

Both ``add_base_input_data_to_options`` twins guard on truthiness, so a reserved key PRESENT with a
falsy value slips past the marked escalation and reaches ``Options.add_to_group``, which raises
unmarked and is contained as a non-match. Doubles are per test and dropped afterwards.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


MATCH_DATA_FEATURE = "option_write_match_data_feat_932"
MATCH_DATA_CLASS_NAME = "OptionWriteMatchDataFG932"
MATCH_DATA_RIVAL_NAME = "OptionWriteMatchDataRivalFG932"
READER_FEATURE = "option_write_reader_feat_932"
READER_RIVAL_NAME = "OptionWriteReaderRivalFG932"
READER_CLASS_NAME = "OptionWriteReader932"
RESERVED_KEY = "BaseInputData"
SCOPE_ACCESS = "option_write_scope_access_932"
GLOBAL_ACCESS = "option_write_global_access_932"
RAISE_TYPE_NAME = "ValueError"

T = TypeVar("T")


class OptionWriteFw932(ComputeFramework):
    """Dummy compute framework for the option-write conflict tests."""


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class _ConflictSnapshot:
    """Plain-data readout of one evaluation. Holds no class and no exception object."""

    escaped: Optional[str]
    identified_names: tuple[str, ...]


def _make_match_data_fg() -> type[FeatureGroup]:
    """MatchData candidate whose reserved key is already PRESENT with a falsy value."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class OptionWriteMatchDataFG932(FeatureGroup, MatchData):
        """Resolves a global-scope access while its own class-name key sits in the options as None."""

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
            data_access_collection: Optional[DataAccessCollection] = None,
            framework_connection_object: Optional[Any] = None,
        ) -> Any:
            if str(feature_name) != MATCH_DATA_FEATURE:
                return None
            if data_access_collection is None:
                return None
            return GLOBAL_ACCESS

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return OptionWriteMatchDataFG932


def _make_match_data_rival_fg() -> type[FeatureGroup]:
    """Rival claiming the MatchData feature name cleanly, so a contained conflict would let it win."""
    gc.collect()

    class OptionWriteMatchDataRivalFG932(FeatureGroup):
        """The group that would silently win while the option-write conflict is swallowed."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {MATCH_DATA_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionWriteFw932}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return OptionWriteMatchDataRivalFG932


class OptionWriteReaderFamily932(BaseInputData):
    """Family base of the test reader; never final itself, so only its child is discovered."""


class OptionWriteReader932(OptionWriteReaderFamily932):
    """Final reader that matches ONLY its own marker value, so process-wide discovery cannot collide."""

    @classmethod
    def match_subclass_data_access(
        cls, data_access: Any, feature_names: list[str], options: Optional[Options] = None
    ) -> Any:
        if isinstance(data_access, DataAccessCollection):
            if GLOBAL_ACCESS in data_access.connections.values():
                return GLOBAL_ACCESS
            return None
        if data_access == SCOPE_ACCESS:
            return SCOPE_ACCESS
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return None


def _make_reader_fg() -> type[FeatureGroup]:
    """Reader-selecting candidate whose reserved 'BaseInputData' key is PRESENT with a falsy value."""
    gc.collect()

    class OptionWriteReaderFG932(FeatureGroup):
        """Root feature group selecting the test reader family."""

        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return OptionWriteReaderFamily932()

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {READER_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionWriteFw932}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return OptionWriteReaderFG932


def _make_reader_rival_fg() -> type[FeatureGroup]:
    """Rival claiming the reader feature name cleanly, without touching any reader option."""
    gc.collect()

    class OptionWriteReaderRivalFG932(FeatureGroup):
        """The group that would silently win while the option-write conflict is swallowed."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {READER_FEATURE}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {OptionWriteFw932}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return OptionWriteReaderRivalFG932


def _evaluate(
    feature_name: str,
    options: Options,
    conflict_fg: type[FeatureGroup],
    rival_fg: Optional[type[FeatureGroup]],
    data_access: Optional[DataAccessCollection],
) -> _ConflictSnapshot:
    """Evaluate one feature at the match seam and read the outcome out as plain data."""
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
    """MatchData twin: the class-name key is present as None while a global access resolves."""
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


def _evaluate_reader_conflict(with_rival: bool, global_scope: bool) -> _ConflictSnapshot:
    """BaseInputData twin: the reserved key is present as None while a reader resolves."""
    conflict_fg = _make_reader_fg()
    rival_fg = _make_reader_rival_fg() if with_rival else None
    try:
        group: dict[str, Any] = {RESERVED_KEY: None}
        data_access: Optional[DataAccessCollection] = None
        if global_scope:
            data_access = DataAccessCollection(connections={"option_write_handle_932": GLOBAL_ACCESS})
        else:
            group[READER_CLASS_NAME] = SCOPE_ACCESS
        return _evaluate(READER_FEATURE, Options(group=group), conflict_fg, rival_fg, data_access)
    finally:
        del conflict_fg
        del rival_fg
        gc.collect()


class TestMatchDataOptionWriteConflictAbortsTheMatch:
    """A contradicting option write under MatchData is a user contradiction, not a non-match."""

    def test_conflict_reaches_the_caller(self) -> None:
        snapshot = _evaluate_match_data_conflict(with_rival=False)

        assert snapshot.escaped is not None, "the option-write conflict must not be contained as a non-match"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: "), (
            f"the conflict's own ValueError must reach the caller, got: {snapshot.escaped}"
        )
        assert MATCH_DATA_CLASS_NAME in snapshot.escaped
        assert snapshot.identified_names == ()

    def test_conflict_is_not_dropped_when_a_rival_claims_the_name(self) -> None:
        snapshot = _evaluate_match_data_conflict(with_rival=True)

        assert snapshot.identified_names != (MATCH_DATA_RIVAL_NAME,), (
            "the rival must not silently win while the option-write conflict is swallowed"
        )
        assert snapshot.escaped is not None, "a rival candidate must not hide the option-write conflict"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert MATCH_DATA_CLASS_NAME in snapshot.escaped


class TestBaseInputDataOptionWriteConflictAbortsTheMatch:
    """The same contradiction during reader selection must escalate on both scope paths."""

    def test_feature_scope_conflict_reaches_the_caller(self) -> None:
        snapshot = _evaluate_reader_conflict(with_rival=False, global_scope=False)

        assert snapshot.escaped is not None, "the option-write conflict must not be contained as a non-match"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: "), (
            f"the conflict's own ValueError must reach the caller, got: {snapshot.escaped}"
        )
        assert RESERVED_KEY in snapshot.escaped
        assert snapshot.identified_names == ()

    def test_global_scope_conflict_reaches_the_caller(self) -> None:
        snapshot = _evaluate_reader_conflict(with_rival=False, global_scope=True)

        assert snapshot.escaped is not None, "the option-write conflict must not be contained as a non-match"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert RESERVED_KEY in snapshot.escaped
        assert snapshot.identified_names == ()

    def test_conflict_is_not_dropped_when_a_rival_claims_the_name(self) -> None:
        snapshot = _evaluate_reader_conflict(with_rival=True, global_scope=False)

        assert snapshot.identified_names != (READER_RIVAL_NAME,), (
            "the rival must not silently win while the option-write conflict is swallowed"
        )
        assert snapshot.escaped is not None, "a rival candidate must not hide the option-write conflict"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: ")
        assert RESERVED_KEY in snapshot.escaped
