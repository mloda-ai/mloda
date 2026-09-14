"""Tests for CfwManager RunContext get/set round-trip and merge-relation cycles."""

from multiprocessing.managers import BaseManager
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager, MyManager


def _module_level_bootstrap() -> None:
    pass


class TestCfwManagerRunContextDefault:
    def test_default_run_context_is_an_empty_run_context(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        assert cfw_register.get_run_context() == RunContext()


class TestCfwManagerDoesNotCarryExtenders:
    """The register is a cross-process coordination object; it must not carry extenders."""

    def test_get_function_extender_method_does_not_exist(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        assert not hasattr(cfw_register, "get_function_extender")


class TestCfwManagerRunContextRoundTrip:
    def test_set_then_get_round_trips_a_run_context_with_a_module_level_picklable_bootstrap(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        cfw_register.set_run_context(RunContext(child_bootstrap=_module_level_bootstrap))

        assert cfw_register.get_run_context().child_bootstrap is _module_level_bootstrap

    def test_set_empty_run_context_after_a_previous_set_clears_it_back_to_default(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cfw_register.set_run_context(RunContext(child_bootstrap=_module_level_bootstrap))

        cfw_register.set_run_context(RunContext())

        assert cfw_register.get_run_context() == RunContext()


class TestCfwManagerFindLeftmostCycleDetection:
    """A merge-relation cycle must raise, not hang find_leftmost forever."""

    @pytest.mark.timeout(2)
    def test_find_leftmost_raises_value_error_on_a_two_node_cycle(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        uuid_a = uuid4()
        uuid_b = uuid4()
        cls_name = "SomeComputeFramework"

        cfw_register.add_to_merge_relation(uuid_a, uuid_b, cls_name)
        cfw_register.add_to_merge_relation(uuid_b, uuid_a, cls_name)

        with pytest.raises(ValueError):
            cfw_register.find_leftmost(uuid_a, cls_name)

    @pytest.mark.timeout(2)
    def test_find_leftmost_raises_value_error_on_a_three_node_cycle(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        uuid_a = uuid4()
        uuid_b = uuid4()
        uuid_c = uuid4()
        cls_name = "SomeComputeFramework"

        cfw_register.add_to_merge_relation(uuid_a, uuid_b, cls_name)
        cfw_register.add_to_merge_relation(uuid_b, uuid_c, cls_name)
        cfw_register.add_to_merge_relation(uuid_c, uuid_a, cls_name)

        with pytest.raises(ValueError):
            cfw_register.find_leftmost(uuid_a, cls_name)

    def test_find_leftmost_returns_the_root_uuid_for_a_non_cyclic_multi_hop_chain(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        uuid_a = uuid4()
        uuid_b = uuid4()
        uuid_c = uuid4()
        cls_name = "SomeComputeFramework"

        cfw_register.add_to_merge_relation(uuid_a, uuid_b, cls_name)
        cfw_register.add_to_merge_relation(uuid_b, uuid_c, cls_name)

        assert cfw_register.find_leftmost(uuid_c, cls_name) == uuid_a
        assert cfw_register.find_leftmost(uuid_b, cls_name) == uuid_a
        assert cfw_register.find_leftmost(uuid_a, cls_name) == uuid_a


class TestCfwManagerGetCfwUuidBackHopResolution:
    """mloda-ai/mloda#1428: a descendant reading both a root feature and a feature hopped back
    into the root's own framework must resolve to the back-hop's cfw, not the root's, even
    though the root's children_if_root transitively lists the same descendant uuid."""

    def test_get_unique_cfw_uuid_resolves_a_tfs_steps_own_registered_uuid(self) -> None:
        """A TFS-created cfw is registered under its own step uuid (see
        ComputeFrameworkExecutor.init_compute_framework's `uuid=step.uuid` call for a
        TransformFrameworkStep). get_unique_cfw_uuid(cls_name, {that uuid}) must resolve to that
        cfw itself, the way ComputeFrameworkExecutor.prepare_execute_step relies on via
        step.tfs_ids, not only to a cfw whose children_if_root happens to list it as a member.
        """
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cls_name = "PythonDictFramework"
        hop_cfw_uuid = uuid4()  # the back-hop TFS step's own uuid, used as the created cfw's key

        cfw_register.add_cfw_to_compute_frameworks(hop_cfw_uuid, cls_name, {uuid4(), uuid4()})

        resolved = cfw_register.get_unique_cfw_uuid(cls_name, {hop_cfw_uuid})

        assert resolved == hop_cfw_uuid

    def test_get_cfw_uuid_picks_the_back_hop_cfw_over_an_earlier_root_cfw_sharing_the_uuid(self) -> None:
        """The root cfw's children_if_root transitively lists every descendant uuid, including
        one actually produced by a back-hop cfw registered afterwards under the same class name.
        get_cfw_uuid must not let the earlier, root cfw win just because it was registered first.
        """
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cls_name = "PythonDictFramework"
        descendant_feature_uuid = uuid4()
        root_cfw_uuid = uuid4()
        hop_cfw_uuid = uuid4()

        # Root cfw registered first; its children_if_root transitively includes every descendant,
        # including the one actually produced by the back-hop cfw registered afterwards.
        cfw_register.add_cfw_to_compute_frameworks(root_cfw_uuid, cls_name, {descendant_feature_uuid, uuid4()})
        cfw_register.add_cfw_to_compute_frameworks(hop_cfw_uuid, cls_name, {descendant_feature_uuid})

        resolved = cfw_register.get_cfw_uuid(cls_name, descendant_feature_uuid)

        assert resolved == hop_cfw_uuid


class TestMyManagerStartAlwaysChainsTheParentDeathWatchdog:
    """MyManager.start() must always run the watchdog in the manager server process, and if the
    caller also supplied an initializer, chain it after the watchdog rather than replacing it."""

    def test_start_with_no_initializer_still_invokes_the_watchdog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_super_start(self: BaseManager, initializer: Any = None, initargs: Any = ()) -> None:
            captured["initializer"] = initializer
            captured["initargs"] = initargs

        monkeypatch.setattr(BaseManager, "start", fake_super_start)

        watchdog_mock = Mock()
        monkeypatch.setattr("mloda.core.core.cfw_manager.start_parent_death_watchdog", watchdog_mock)

        MyManager().start()

        # Invoke whatever MyManager.start() actually handed to super().start(), simulating
        # BaseManager._run_server() calling it inside the freshly spawned manager process.
        captured["initializer"](*captured["initargs"])

        watchdog_mock.assert_called_once()

    def test_start_with_an_explicit_initializer_chains_it_after_the_watchdog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_super_start(self: BaseManager, initializer: Any = None, initargs: Any = ()) -> None:
            captured["initializer"] = initializer
            captured["initargs"] = initargs

        monkeypatch.setattr(BaseManager, "start", fake_super_start)

        watchdog_mock = Mock()
        monkeypatch.setattr("mloda.core.core.cfw_manager.start_parent_death_watchdog", watchdog_mock)

        caller_calls: list[tuple[Any, ...]] = []

        def my_initializer(*args: Any) -> None:
            caller_calls.append(args)

        MyManager().start(initializer=my_initializer, initargs=("a", "b"))

        captured["initializer"](*captured["initargs"])

        watchdog_mock.assert_called_once()
        assert caller_calls == [("a", "b")]
