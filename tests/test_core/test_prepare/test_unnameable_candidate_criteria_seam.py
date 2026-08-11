"""The criteria seam names its candidate in both reports it writes, and that name is plugin-overridable.

Each report reads it BEFORE the containment it describes, so the read must degrade rather than end the
resolution pass. Probe classes live inside factory functions and are dropped before any assert runs.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_types import EliminationStage


IDENTIFY_LOGGER_NAME = IdentifyFeatureGroupClass.__module__

UCS_FEATURE = "ucs_probe_feat"  # the name every probe here is asked about
UCS_MATCHER_MESSAGE = "ucs_matcher_boom"  # what the defective matcher raises
UCS_DECLINE_MESSAGE = "ucs_declined_value"  # what the declining matcher raises
UCS_CLASS_NAME_MESSAGE = "ucs_class_name_boom"  # what the unreadable class name raises

UNNAMED_GROUP_FALLBACK = "<unnamed feature group>"  # what a guarded read of a group's class name degrades to
DECLINE_REPORT_FRAGMENT = "rejected an option value"  # the decline report's own wording

MATCHER_ERROR_STAGE: EliminationStage = "matcher_error"
VALUE_REJECTION_STAGE: EliminationStage = "value_rejection"

T = TypeVar("T")

_Factory = Callable[[], type[FeatureGroup]]


class UcsFw(ComputeFramework):
    """Dummy compute framework for the unnameable-candidate probes."""


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _fact_of(elimination: Any) -> tuple[str, str]:
    """(stage, reason) of one elimination. Any, so the readout never depends on the record's declared shape."""
    return str(elimination.stage), str(elimination.reason)


def _reported(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    """Formatted messages the identify seam logged, whatever level it chose for each."""
    return tuple(record.getMessage() for record in caplog.records if record.name == IDENTIFY_LOGGER_NAME)


def _carrying(messages: Sequence[str], fragment: str) -> tuple[str, ...]:
    """The messages carrying `fragment`."""
    return tuple(message for message in messages if fragment in message)


def _make_unnameable_matcher_defect_fg() -> type[FeatureGroup]:
    """A throwaway candidate whose matcher raises and that cannot say what it is called either."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class UcsMatcherDefectFG(FeatureGroup):
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {UcsFw}

        # The @final on get_class_name is a mypy pin; a plugin can still install this override at runtime.
        @classmethod  # type: ignore[misc]
        def get_class_name(cls) -> str:
            raise RuntimeError(UCS_CLASS_NAME_MESSAGE)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            raise RuntimeError(UCS_MATCHER_MESSAGE)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return UcsMatcherDefectFG


def _make_unnameable_value_decline_fg() -> type[FeatureGroup]:
    """A throwaway candidate that declines the option value and cannot say what it is called either."""
    gc.collect()

    class UcsValueDeclineFG(FeatureGroup):
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {UcsFw}

        @classmethod  # type: ignore[misc]
        def get_class_name(cls) -> str:
            raise RuntimeError(UCS_CLASS_NAME_MESSAGE)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            raise PropertyValueRejection(UCS_DECLINE_MESSAGE)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return UcsValueDeclineFG


@dataclass(frozen=True)
class _SeamSnapshot:
    """Plain-data readout of one evaluation. Holds no class, no exception and no Elimination object."""

    escaped: str | None
    failure_kind: str | None
    facts: tuple[tuple[str, str], ...]
    reported: tuple[str, ...]


def _drive(make: _Factory, caplog: pytest.LogCaptureFixture) -> _SeamSnapshot:
    """Evaluate the probe alone, folding the outcome and the seam's own log records to plain data."""
    caplog.clear()
    fg = make()
    plugins: FeatureGroupEnvironmentMapping = {fg: {UcsFw}}
    result = None
    try:
        with caplog.at_level(logging.DEBUG, logger=IDENTIFY_LOGGER_NAME):
            result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, Feature(UCS_FEATURE), plugins, None))
        snapshot = _SeamSnapshot(
            escaped=escaped,
            failure_kind=None if result is None else result.failure_kind,
            # The stored values only: reading a key back would ask the class the question it cannot answer.
            facts=() if result is None else tuple(sorted(_fact_of(e) for e in result.eliminations.values())),
            reported=_reported(caplog),
        )
        del result
        result = None
        return snapshot
    finally:
        del fg, plugins, result
        gc.collect()


class TestAnUnreadableCandidateNameOnTheCriteriaSeamDegrades:
    """Both reports of the seam name their candidate as an eager argument, built before the guard they label."""

    def test_a_contained_matcher_raise_whose_group_cannot_name_itself_still_evaluates(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The report runs past the hook call's containment, so its own read must not end the resolution pass."""
        snapshot = _drive(_make_unnameable_matcher_defect_fg, caplog)

        assert snapshot.escaped is None, f"nothing may cross evaluate(): {snapshot.escaped}"
        assert snapshot.failure_kind == "none", f"a contained raise is a non-match, got: {snapshot.failure_kind}"
        assert len(snapshot.facts) == 1, f"the near-miss must still be recorded, got: {snapshot.facts}"
        stage, reason = snapshot.facts[0]
        assert stage == MATCHER_ERROR_STAGE, f"a contained raise is a matcher_error, got stage: {stage}"
        assert UCS_MATCHER_MESSAGE in reason, f"the reason must name the contained raise: {reason}"
        reported = _carrying(snapshot.reported, UCS_MATCHER_MESSAGE)
        assert len(reported) == 1, f"the contained raise must still be reported once, got: {snapshot.reported}"
        assert UNNAMED_GROUP_FALLBACK in reported[0], f"an unreadable group name must degrade: {reported[0]}"
        assert UCS_FEATURE in reported[0], f"the report must name the feature it was matching: {reported[0]}"

    def test_a_contained_value_decline_whose_group_cannot_name_itself_still_evaluates(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The other branch of the same seam: a declined value is reported by name too, and degrades the same."""
        snapshot = _drive(_make_unnameable_value_decline_fg, caplog)

        assert snapshot.escaped is None, f"nothing may cross evaluate(): {snapshot.escaped}"
        assert snapshot.failure_kind == "none", f"a declined value is a non-match, got: {snapshot.failure_kind}"
        assert len(snapshot.facts) == 1, f"the near-miss must still be recorded, got: {snapshot.facts}"
        stage, reason = snapshot.facts[0]
        assert stage == VALUE_REJECTION_STAGE, f"a declined value is a value_rejection, got stage: {stage}"
        assert UCS_DECLINE_MESSAGE in reason, f"the reason must name the decline: {reason}"
        reported = _carrying(snapshot.reported, DECLINE_REPORT_FRAGMENT)
        assert len(reported) == 1, f"the decline must still be reported once, got: {snapshot.reported}"
        assert UNNAMED_GROUP_FALLBACK in reported[0], f"an unreadable group name must degrade: {reported[0]}"
        assert UCS_DECLINE_MESSAGE in reported[0], f"the fields that ARE readable must still be named: {reported[0]}"
        assert UCS_FEATURE in reported[0], f"the report must name the feature it was matching: {reported[0]}"
