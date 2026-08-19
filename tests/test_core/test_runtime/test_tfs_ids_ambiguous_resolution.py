"""Tests for FeatureGroupStep.tfs_ids ambiguity detection.

Guards prepare_execute_step (ComputeFrameworkExecutor) and _cfw_to_occupy
(ExecutionOrchestrator): if two tfs_ids resolve to two DIFFERENT registered
compute frameworks, resolution must raise instead of picking one via set order.
"""

from __future__ import annotations

from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.runtime.worker_manager import WorkerManager

CLASS_NAME = "TestCFW"


def _register_two_distinct_cfws(cfw_register: CfwManager, tfs_id_a: UUID, tfs_id_b: UUID) -> tuple[UUID, UUID]:
    """Register two cfws under CLASS_NAME, each rooted on a different tfs_id."""
    cfw_uuid_a, cfw_uuid_b = uuid4(), uuid4()
    cfw_register.add_cfw_to_compute_frameworks(cfw_uuid_a, CLASS_NAME, {tfs_id_a})
    cfw_register.add_cfw_to_compute_frameworks(cfw_uuid_b, CLASS_NAME, {tfs_id_b})
    return cfw_uuid_a, cfw_uuid_b


def _register_one_shared_cfw(cfw_register: CfwManager, tfs_id_a: UUID, tfs_id_b: UUID) -> UUID:
    """Register a single cfw rooted on both tfs_ids (non-ambiguous)."""
    cfw_uuid = uuid4()
    cfw_register.add_cfw_to_compute_frameworks(cfw_uuid, CLASS_NAME, {tfs_id_a, tfs_id_b})
    return cfw_uuid


def _feature_group_step(tfs_ids: set[UUID]) -> Mock:
    step = Mock(spec=FeatureGroupStep)
    step.tfs_ids = tfs_ids
    step.compute_framework = Mock()
    step.compute_framework.get_class_name.return_value = CLASS_NAME
    step.get_parallelization_mode.return_value = set()
    return step


class TestPrepareExecuteStepTfsIdsAmbiguity:
    """ComputeFrameworkExecutor.prepare_execute_step, FeatureGroupStep branch."""

    def test_raises_when_two_tfs_ids_resolve_to_different_cfws(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        tfs_id_a, tfs_id_b = uuid4(), uuid4()
        _register_two_distinct_cfws(cfw_register, tfs_id_a, tfs_id_b)

        step = _feature_group_step({tfs_id_a, tfs_id_b})

        with pytest.raises(ValueError, match="(?i)ambiguous"):
            executor.prepare_execute_step(step, ParallelizationMode.SYNC)

    def test_returns_cfw_uuid_when_all_tfs_ids_resolve_to_same_cfw(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        tfs_id_a, tfs_id_b = uuid4(), uuid4()
        cfw_uuid = _register_one_shared_cfw(cfw_register, tfs_id_a, tfs_id_b)

        step = _feature_group_step({tfs_id_a, tfs_id_b})

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == cfw_uuid

    def test_returns_cfw_uuid_when_only_one_tfs_id_resolves(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        tfs_id_resolved, tfs_id_unresolved = uuid4(), uuid4()
        cfw_uuid = uuid4()
        cfw_register.add_cfw_to_compute_frameworks(cfw_uuid, CLASS_NAME, {tfs_id_resolved})

        step = _feature_group_step({tfs_id_resolved, tfs_id_unresolved})

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == cfw_uuid

    def test_falls_through_to_any_uuid_when_no_tfs_id_resolves(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        any_uuid = uuid4()
        cfw_uuid = uuid4()
        cfw_register.add_cfw_to_compute_frameworks(cfw_uuid, CLASS_NAME, {any_uuid})

        step = _feature_group_step({uuid4(), uuid4()})
        step.features = Mock()
        step.features.any_uuid = any_uuid
        step.children_if_root = set()

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == cfw_uuid


class TestCfwToOccupyTfsIdsAmbiguity:
    """ExecutionOrchestrator._cfw_to_occupy."""

    def _orchestrator(self, cfw_register: CfwManager) -> ExecutionOrchestrator:
        orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))
        orchestrator.cfw_register = cfw_register
        return orchestrator

    def test_raises_when_two_tfs_ids_resolve_to_different_cfws(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        orchestrator = self._orchestrator(cfw_register)

        tfs_id_a, tfs_id_b = uuid4(), uuid4()
        _register_two_distinct_cfws(cfw_register, tfs_id_a, tfs_id_b)

        step = _feature_group_step({tfs_id_a, tfs_id_b})

        with pytest.raises(ValueError, match="(?i)ambiguous"):
            orchestrator._cfw_to_occupy(step)

    def test_returns_cfw_uuid_when_all_tfs_ids_resolve_to_same_cfw(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        orchestrator = self._orchestrator(cfw_register)

        tfs_id_a, tfs_id_b = uuid4(), uuid4()
        cfw_uuid = _register_one_shared_cfw(cfw_register, tfs_id_a, tfs_id_b)

        step = _feature_group_step({tfs_id_a, tfs_id_b})

        result = orchestrator._cfw_to_occupy(step)

        assert result == cfw_uuid

    def test_returns_cfw_uuid_when_only_one_tfs_id_resolves(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        orchestrator = self._orchestrator(cfw_register)

        tfs_id_resolved, tfs_id_unresolved = uuid4(), uuid4()
        cfw_uuid = uuid4()
        cfw_register.add_cfw_to_compute_frameworks(cfw_uuid, CLASS_NAME, {tfs_id_resolved})

        step = _feature_group_step({tfs_id_resolved, tfs_id_unresolved})

        result = orchestrator._cfw_to_occupy(step)

        assert result == cfw_uuid

    def test_falls_through_to_any_uuid_when_no_tfs_id_resolves(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        orchestrator = self._orchestrator(cfw_register)

        any_uuid = uuid4()
        cfw_uuid = uuid4()
        cfw_register.add_cfw_to_compute_frameworks(cfw_uuid, CLASS_NAME, {any_uuid})

        step = _feature_group_step({uuid4(), uuid4()})
        step.features = Mock()
        step.features.any_uuid = any_uuid

        result = orchestrator._cfw_to_occupy(step)

        assert result == cfw_uuid
