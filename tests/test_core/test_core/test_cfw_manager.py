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
