"""raise_on_live_connection_for_worker: a live-only ConnectionSource for a step that can run in a
spawned MULTIPROCESSING worker must be rejected at plan time, since the live handle never crosses
into the worker process.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource, ConnectionSpec
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.runtime.validate_multiprocessing_link import raise_on_live_connection_for_worker
from mloda.provider import FeatureGroup


class _MpConnFeatureGroup(FeatureGroup):
    """Module-level, picklable feature group used only to build steps."""


class _MpCapableFramework(ComputeFramework):
    """A compute framework whose modes include MULTIPROCESSING."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING}


class _SyncOnlyConnFramework(ComputeFramework):
    """A compute framework restricted to SYNC: never dispatched to a worker."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC}


def _feature_group_step(compute_framework: type[ComputeFramework]) -> FeatureGroupStep:
    return FeatureGroupStep(_MpConnFeatureGroup, FeatureSet([Feature("f")]), set(), compute_framework)


def _transform_step(to_framework: type[ComputeFramework]) -> TransformFrameworkStep:
    return TransformFrameworkStep(_MpCapableFramework, to_framework, set(), _MpConnFeatureGroup, _MpConnFeatureGroup)


def _join_step(destination_framework: type[ComputeFramework]) -> JoinStep:
    return JoinStep(MagicMock(), destination_framework, _MpCapableFramework, set(), set(), set())


def test_transform_framework_step_with_live_only_source_raises() -> None:
    step = _transform_step(_MpCapableFramework)
    connection_sources = {_MpCapableFramework: ConnectionSource(live=object())}

    with pytest.raises(ValueError) as excinfo:
        raise_on_live_connection_for_worker([step], connection_sources)

    message = str(excinfo.value)
    assert "_MpCapableFramework" in message
    assert "TransformFrameworkStep" in message
    assert "live" in message
    assert "ConnectionSpec(" in message
    assert "ParallelizationMode.MULTIPROCESSING" in message


def test_feature_group_step_with_live_only_source_raises() -> None:
    step = _feature_group_step(_MpCapableFramework)
    connection_sources = {_MpCapableFramework: ConnectionSource(live=object())}

    with pytest.raises(ValueError) as excinfo:
        raise_on_live_connection_for_worker([step], connection_sources)

    assert "FeatureGroupStep" in str(excinfo.value)


def test_join_step_with_live_only_source_raises() -> None:
    step = _join_step(_MpCapableFramework)
    connection_sources = {_MpCapableFramework: ConnectionSource(live=object())}

    with pytest.raises(ValueError) as excinfo:
        raise_on_live_connection_for_worker([step], connection_sources)

    assert "JoinStep" in str(excinfo.value)


def test_spec_backed_source_does_not_raise() -> None:
    step = _transform_step(_MpCapableFramework)
    connection_sources = {_MpCapableFramework: ConnectionSource(spec=ConnectionSpec(_MpCapableFramework))}

    raise_on_live_connection_for_worker([step], connection_sources)


def test_spec_backed_source_with_unpicklable_params_raises() -> None:
    step = _transform_step(_MpCapableFramework)
    spec = ConnectionSpec(_MpCapableFramework, callback=lambda: None)
    connection_sources = {_MpCapableFramework: ConnectionSource(spec=spec)}

    with pytest.raises(ValueError) as excinfo:
        raise_on_live_connection_for_worker([step], connection_sources)

    message = str(excinfo.value)
    assert "picklable" in message
    assert "ConnectionSpec" in message


def test_source_with_both_live_and_spec_present_does_not_raise() -> None:
    step = _transform_step(_MpCapableFramework)
    source = ConnectionSource(live=object(), spec=ConnectionSpec(_MpCapableFramework))
    connection_sources = {_MpCapableFramework: source}

    raise_on_live_connection_for_worker([step], connection_sources)


def test_live_source_for_a_non_multiprocessing_step_does_not_raise() -> None:
    step = _transform_step(_SyncOnlyConnFramework)
    connection_sources = {_SyncOnlyConnFramework: ConnectionSource(live=object())}

    raise_on_live_connection_for_worker([step], connection_sources)


def test_empty_connection_sources_does_not_raise() -> None:
    step = _transform_step(_MpCapableFramework)

    raise_on_live_connection_for_worker([step], {})


def test_step_whose_destination_class_has_no_entry_does_not_raise() -> None:
    step = _transform_step(_MpCapableFramework)
    connection_sources = {_SyncOnlyConnFramework: ConnectionSource(live=object())}

    raise_on_live_connection_for_worker([step], connection_sources)


def test_orchestrator_enter_rejects_a_live_only_connection_before_spawning_a_manager() -> None:
    """Mirrors test_orchestrator_rejects_unpicklable_extender.py: the rejection lands before any Manager/worker process starts."""
    step = _feature_group_step(_MpCapableFramework)
    plan = ExecutionPlan()
    plan.execution_plan = [step]
    connection_sources: dict[type[ComputeFramework], ConnectionSource] = {
        _MpCapableFramework: ConnectionSource(live=object())
    }
    orchestrator = ExecutionOrchestrator(plan, connection_sources=connection_sources)

    try:
        with pytest.raises(ValueError, match=r"ConnectionSpec\("):
            orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})
        assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"
    finally:
        if orchestrator.manager is not None:
            orchestrator.__exit__(None, None, None)
