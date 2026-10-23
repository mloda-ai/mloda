"""Tests for CfwManager RunContext get/set round-trip and merge-relation cycles."""

import multiprocessing.process
import os
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

import mloda.core.core.cfw_manager as cfw_manager_module
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager, MyManager

from mloda.core.runtime.worker.multiprocessing_worker import standby_worker
from mloda.core.runtime.worker_manager import WorkerManager

cfw_module: Any = cfw_manager_module


def _module_level_bootstrap() -> None:
    pass


def _hold_register_target(command_queue: Any, result_queue: Any, register: Any, worker_index: int) -> None:
    register.get_location()
    result_queue.put("ready")
    time.sleep(60)


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
    """A descendant reading both a root feature and a feature hopped back
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

    def test_get_cfw_uuid_does_not_pick_a_winner_between_two_identically_childrened_same_class_cfws(self) -> None:
        """Two plain hops out of one shared source cfw into the same destination class (see
        ComputeFrameworkExecutor.prepare_execute_step's TransformFrameworkStep branch) each copy
        `children_if_root` verbatim from that shared source, so they register two same-class cfws
        with an IDENTICAL children_if_root set. Queried by a single member of that shared set (not
        either cfw's own key), get_cfw_uuid's bare class+single-uuid signature has no signal to
        prefer one sibling over the other: both matches are equally valid. It must not silently
        pick a winner by registration order (today's `<=` comparison makes whichever cfw is
        registered LAST win, purely because it is the last one dict-iteration visits) - it must
        defer by returning None, the same as a genuine not-found, rather than guess.
        """
        cls_name = "PythonDictFramework"
        shared_member_uuid = uuid4()

        cfw_register_a_then_b = CfwManager({ParallelizationMode.SYNC})
        cfw_a = uuid4()
        cfw_b = uuid4()
        cfw_register_a_then_b.add_cfw_to_compute_frameworks(cfw_a, cls_name, {shared_member_uuid})
        cfw_register_a_then_b.add_cfw_to_compute_frameworks(cfw_b, cls_name, {shared_member_uuid})

        assert cfw_register_a_then_b.get_cfw_uuid(cls_name, shared_member_uuid) is None

        cfw_register_b_then_a = CfwManager({ParallelizationMode.SYNC})
        cfw_register_b_then_a.add_cfw_to_compute_frameworks(cfw_b, cls_name, {shared_member_uuid})
        cfw_register_b_then_a.add_cfw_to_compute_frameworks(cfw_a, cls_name, {shared_member_uuid})

        assert cfw_register_b_then_a.get_cfw_uuid(cls_name, shared_member_uuid) is None

    def test_get_unique_cfw_uuid_prefers_the_tfs_id_that_is_a_cfws_own_key_over_an_identically_childrened_sibling(
        self,
    ) -> None:
        """Same two identically-childrened sibling hops as above, but resolved the way a real
        caller does: get_unique_cfw_uuid(cls_name, step.tfs_ids), where tfs_ids is the FULL set of
        parent uuids a step reads, not a single bare uuid. That set can include one sibling's own
        registered key directly (`cfw_a`) alongside the shared descendant uuid that ties both
        siblings. A cfw's own key being directly among the queried tfs_ids is a far more meaningful
        signal than indirect, identical children_if_root membership, and get_unique_cfw_uuid is the
        layer with access to it (get_cfw_uuid, called with one uuid at a time, is not). The
        resolution must prefer `cfw_a` regardless of which order the two siblings were registered
        in, not flip depending on which was registered last.
        """
        cls_name = "PythonDictFramework"
        shared_member_uuid = uuid4()

        cfw_register_a_then_b = CfwManager({ParallelizationMode.SYNC})
        cfw_a = uuid4()
        cfw_b = uuid4()
        cfw_register_a_then_b.add_cfw_to_compute_frameworks(cfw_a, cls_name, {shared_member_uuid})
        cfw_register_a_then_b.add_cfw_to_compute_frameworks(cfw_b, cls_name, {shared_member_uuid})
        tfs_ids = {cfw_a, shared_member_uuid}

        assert cfw_register_a_then_b.get_unique_cfw_uuid(cls_name, tfs_ids) == cfw_a

        cfw_register_b_then_a = CfwManager({ParallelizationMode.SYNC})
        cfw_register_b_then_a.add_cfw_to_compute_frameworks(cfw_b, cls_name, {shared_member_uuid})
        cfw_register_b_then_a.add_cfw_to_compute_frameworks(cfw_a, cls_name, {shared_member_uuid})

        assert cfw_register_b_then_a.get_unique_cfw_uuid(cls_name, tfs_ids) == cfw_a


class TestCfwManagerGetCfwUuidAsRegisteredTieSafety:
    """get_cfw_uuid_as_registered mirrors get_cfw_uuid's narrowest-children_if_root-wins tie-break
    (but not its find_leftmost canonicalization, by design): a same-framework JoinStep's source side
    can span more than one FeatureGroupStep, so add_tfs's same-framework JoinStep branch can tag more
    than one FeatureGroupStep's own children_if_root with the same join link uuid."""

    def test_prefers_the_narrower_children_if_root_over_an_earlier_broader_one(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cls_name = "PythonDictFramework"
        link_uuid = uuid4()
        broad_cfw_uuid = uuid4()
        narrow_cfw_uuid = uuid4()

        cfw_register.add_cfw_to_compute_frameworks(broad_cfw_uuid, cls_name, {link_uuid, uuid4()})
        cfw_register.add_cfw_to_compute_frameworks(narrow_cfw_uuid, cls_name, {link_uuid})

        resolved = cfw_register.get_cfw_uuid_as_registered(cls_name, link_uuid)

        assert resolved == narrow_cfw_uuid

    def test_does_not_pick_a_winner_between_two_identically_childrened_same_class_cfws(self) -> None:
        """Two FeatureGroupSteps tagged with the same join link uuid, each producing genuinely
        distinct data but with an IDENTICAL children_if_root set (e.g. each carries only that one
        link uuid), give this lookup no signal to prefer one over the other. It must not silently
        pick a winner by registration order; it must defer by returning None."""
        cls_name = "PythonDictFramework"
        link_uuid = uuid4()

        cfw_register_a_then_b = CfwManager({ParallelizationMode.SYNC})
        cfw_a = uuid4()
        cfw_b = uuid4()
        cfw_register_a_then_b.add_cfw_to_compute_frameworks(cfw_a, cls_name, {link_uuid})
        cfw_register_a_then_b.add_cfw_to_compute_frameworks(cfw_b, cls_name, {link_uuid})

        assert cfw_register_a_then_b.get_cfw_uuid_as_registered(cls_name, link_uuid) is None

        cfw_register_b_then_a = CfwManager({ParallelizationMode.SYNC})
        cfw_register_b_then_a.add_cfw_to_compute_frameworks(cfw_b, cls_name, {link_uuid})
        cfw_register_b_then_a.add_cfw_to_compute_frameworks(cfw_a, cls_name, {link_uuid})

        assert cfw_register_b_then_a.get_cfw_uuid_as_registered(cls_name, link_uuid) is None


class TestCfwManagerGetUniqueCfwUuidOwnKeyAmbiguityAfterMerge:
    """get_unique_cfw_uuid classifies each tfs_id's resolution as an own-key match (never
    ambiguous: two independent hops into one framework class is a normal, non-ambiguous shape) or
    a genuine children_if_root-membership match (ambiguous when several disagree) by comparing the
    resolved uuid to the queried tfs_id AFTER get_cfw_uuid ran it through find_leftmost. A JoinStep
    merging one of those own-key cfws into its destination (JoinStep.execute ->
    CfwManager.add_to_merge_relation) re-points that comparison's result, so a same-shape,
    genuinely non-ambiguous case gets misclassified as membership-resolved and wrongly raises."""

    def test_two_own_key_resolutions_repointed_by_a_merge_do_not_raise_ambiguous(self) -> None:
        """hop1 and hop2 are each registered under their own step uuid (own-key cfws, exactly like
        the back-hop cfw in TestCfwManagerGetCfwUuidBackHopResolution above), then each merged into
        its own, distinct destination cfw the way JoinStep.execute merges a hop into a join's
        destination. Resolving {hop1, hop2} together is the ordinary shape of a consumer reaching
        two independent hops into the same framework, not an internal error: it must not raise,
        the same as it does not raise before either hop is merged (see
        test_get_unique_cfw_uuid_resolves_a_tfs_steps_own_registered_uuid above).
        """
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cls_name = "PythonDictFramework"
        hop1_uuid = uuid4()
        hop2_uuid = uuid4()
        dest1_uuid = uuid4()
        dest2_uuid = uuid4()

        cfw_register.add_cfw_to_compute_frameworks(hop1_uuid, cls_name, {uuid4()})
        cfw_register.add_cfw_to_compute_frameworks(hop2_uuid, cls_name, {uuid4()})

        # Sanity check, mirroring the pre-merge behavior pinned above: before either hop is
        # merged, resolving both own keys together already does not raise.
        assert cfw_register.get_unique_cfw_uuid(cls_name, {hop1_uuid, hop2_uuid}) is None

        cfw_register.add_to_merge_relation(dest1_uuid, hop1_uuid, cls_name)
        cfw_register.add_to_merge_relation(dest2_uuid, hop2_uuid, cls_name)

        # Each hop's own-key match now resolves, via find_leftmost, to its own distinct
        # destination cfw rather than to its own uuid - the merge is real and re-points both.
        assert cfw_register.get_cfw_uuid(cls_name, hop1_uuid) == dest1_uuid
        assert cfw_register.get_cfw_uuid(cls_name, hop2_uuid) == dest2_uuid

        resolved = cfw_register.get_unique_cfw_uuid(cls_name, {hop1_uuid, hop2_uuid})

        assert resolved is None


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


@pytest.fixture
def isolated_shared_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Empties the process-wide cache and shuts down every real manager the test started."""
    started: list[MyManager] = []
    real_start = MyManager.start

    def recording_start(self: MyManager, initializer: Any = None, initargs: Any = ()) -> None:
        real_start(self, initializer, initargs)
        started.append(self)

    monkeypatch.setattr(cfw_manager_module, "_shared_manager", None)
    monkeypatch.setattr(MyManager, "start", recording_start)
    yield
    for manager in started:
        manager.shutdown()


@pytest.mark.usefixtures("isolated_shared_manager")
class TestSharedManagerWithARealServer:
    @pytest.mark.timeout(30)
    def test_two_calls_return_the_same_alive_manager(self) -> None:
        first = cfw_module.shared_manager()
        second = cfw_module.shared_manager()

        assert first is second
        assert first.is_alive()

    @pytest.mark.timeout(30)
    def test_registers_from_one_manager_do_not_share_state(self) -> None:
        manager: Any = cfw_module.shared_manager()
        modes = {ParallelizationMode.MULTIPROCESSING}
        register_a = manager.CfwManager(modes)
        register_b = manager.CfwManager(modes)

        register_a.set_location("grpc://a")

        assert register_a.get_location() == "grpc://a"
        assert register_b.get_location() is None

        register_a.release()

        assert register_a.get_location() is None
        register_b.set_location("grpc://b")
        assert register_b.get_location() == "grpc://b"

    @pytest.mark.parametrize("breakage", ["shutdown", "socket_file_removed"])
    @pytest.mark.timeout(30)
    def test_an_unusable_cached_server_is_replaced_by_a_working_one(self, breakage: str) -> None:
        first = cfw_module.shared_manager()
        if breakage == "shutdown":
            first.shutdown()
        else:
            os.unlink(first.address)
            assert first.is_alive() is False

        second = cfw_module.shared_manager()
        register = second.CfwManager({ParallelizationMode.MULTIPROCESSING})

        assert second is not first
        assert second.is_alive()
        register.set_location("grpc://replacement")
        assert register.get_location() == "grpc://replacement"

    @pytest.mark.timeout(30)
    def test_a_replaced_manager_with_a_user_keeps_its_server_until_the_last_release(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        first = cfw_module.shared_manager()
        monkeypatch.syspath_prepend(str(tmp_path))
        second = cfw_module.shared_manager()

        assert second is not first
        assert first.is_alive()

        process = getattr(first, "_process")
        cfw_module.release_shared_manager(first)

        assert not process.is_alive()
        assert second.is_alive()

    @pytest.mark.timeout(30)
    def test_a_register_given_to_a_standby_bound_worker_is_gone_once_the_worker_and_parent_let_go(self) -> None:
        manager: Any = cfw_module.shared_manager()
        register = manager.CfwManager({ParallelizationMode.MULTIPROCESSING})
        workers = WorkerManager()
        try:
            workers.prestart_workers(1, standby_worker)
            process, _, result_queue = workers.create_worker_process(uuid4(), _hold_register_target, (register,))
            assert result_queue.get(timeout=20) == "ready"

            process.terminate()
            process.join(10)
            del register

            deadline = time.monotonic() + 10
            while manager._number_of_objects() != 0 and time.monotonic() < deadline:
                time.sleep(0.05)

            assert manager._number_of_objects() == 0
        finally:
            workers.stop_standbys()
            workers.terminate_alive_processes()


@pytest.mark.usefixtures("isolated_shared_manager")
class TestSharedManagerWithAFakedStart:
    @staticmethod
    def _fake_start(
        monkeypatch: pytest.MonkeyPatch, delay: float = 0.0, shutdowns: list[tuple[MyManager, bool]] | None = None
    ) -> list[MyManager]:
        starts: list[MyManager] = []

        def fake_start(self: MyManager, initializer: Any = None, initargs: Any = ()) -> None:
            time.sleep(delay)
            starts.append(self)

        def fake_shutdown(self: MyManager) -> None:
            if shutdowns is not None:
                shutdowns.append((self, cfw_manager_module._shared_manager_lock.locked()))

        monkeypatch.setattr(MyManager, "start", fake_start)
        monkeypatch.setattr(MyManager, "shutdown", fake_shutdown, raising=False)
        monkeypatch.setattr(MyManager, "is_alive", lambda self: True, raising=False)
        return starts

    def test_every_call_adds_one_user_to_the_manager_it_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_start(monkeypatch)
        manager: Any = cfw_module.shared_manager()
        assert manager.users == 1

        assert cfw_module.shared_manager() is manager
        assert cfw_module.shared_manager() is manager

        assert manager.users == 3
        assert cfw_module.MyManager().users == 0

    @pytest.mark.parametrize("case", ["not_replaced", "replaced_no_users", "replaced_one_user"])
    def test_a_manager_is_shut_down_only_when_replaced_and_its_last_user_lets_go(
        self, case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        shutdowns: list[tuple[MyManager, bool]] = []
        self._fake_start(monkeypatch, shutdowns=shutdowns)
        first: Any = cfw_module.shared_manager()
        if case == "not_replaced":
            cfw_module.release_shared_manager(first)

            assert first.users == 0
            assert shutdowns == []
            assert cfw_module.shared_manager() is first
            return
        if case == "replaced_no_users":
            cfw_module.release_shared_manager(first)
        monkeypatch.syspath_prepend(str(tmp_path))

        second = cfw_module.shared_manager()

        assert second is not first
        if case == "replaced_no_users":
            assert shutdowns == [(first, False)]
            return
        assert shutdowns == []

        cfw_module.release_shared_manager(first)
        assert shutdowns == [(first, False)]
        assert first.users == 0

        cfw_module.release_shared_manager(first)
        assert shutdowns == [(first, False)]
        assert first.users == 0

    def test_a_manager_with_a_non_string_address_is_alive_when_its_process_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = MyManager()
        monkeypatch.setattr(MyManager, "address", property(lambda self: ("127.0.0.1", 1)))
        process = Mock()
        process.is_alive.return_value = True
        manager._process = process  # type: ignore[attr-defined]

        assert manager.is_alive() is True

    def test_forgetting_the_cache_after_a_fork_swaps_the_lock_and_clears_the_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._fake_start(monkeypatch)
        cached: Any = cfw_module.shared_manager()
        other = MyManager()
        stand_ins = [object(), object()]
        cached._process = stand_ins[0]
        setattr(other, "_process", stand_ins[1])
        monkeypatch.setattr(cfw_manager_module, "_started_managers", weakref.WeakSet([cached, other]), raising=False)
        children: Any = multiprocessing.process._children  # type: ignore[attr-defined]
        children.update(stand_ins)
        held = threading.Lock()
        held.acquire()
        monkeypatch.setattr(cfw_manager_module, "_shared_manager_lock", held)

        try:
            cfw_module._forget_shared_manager_after_fork()

            assert not any(stand_in in children for stand_in in stand_ins)
        finally:
            children.difference_update(stand_ins)

        assert cfw_module._shared_manager is None
        assert cfw_module._shared_manager_lock is not held
        assert not cfw_module._shared_manager_lock.locked()
        held.release()

    @pytest.mark.parametrize("reason", ["unchanged", "sys_path_changed", "pid_changed", "dead"])
    def test_restarts_only_when_the_cached_manager_is_stale(
        self, reason: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        starts = self._fake_start(monkeypatch)
        first = cfw_module.shared_manager()

        if reason == "sys_path_changed":
            monkeypatch.syspath_prepend(str(tmp_path))
        if reason == "dead":
            monkeypatch.setattr(MyManager, "is_alive", lambda self: False, raising=False)
        if reason == "pid_changed":
            cached_manager, cached_pid, cached_path = cfw_module._shared_manager
            monkeypatch.setattr(cfw_manager_module, "_shared_manager", (cached_manager, cached_pid + 1, cached_path))

            def is_alive_must_not_run(self: MyManager) -> bool:
                raise AssertionError("is_alive called for a manager started in another pid")

            monkeypatch.setattr(MyManager, "is_alive", is_alive_must_not_run, raising=False)

        second = cfw_module.shared_manager()

        if reason == "unchanged":
            assert second is first
            assert len(starts) == 1
            assert cfw_module._shared_manager == (first, os.getpid(), tuple(sys.path))
        else:
            assert second is not first
            assert starts == [first, second]

    def test_manager_is_built_with_the_mp_start_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_start(monkeypatch)
        sentinel = object()
        monkeypatch.setattr(cfw_manager_module, "mp_start_context", lambda: sentinel)

        manager = cfw_module.shared_manager()

        assert manager._ctx is sentinel

    def test_concurrent_calls_share_one_manager_and_one_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        starts = self._fake_start(monkeypatch, delay=0.2)
        results: list[Any] = []
        barrier = threading.Barrier(6)

        def call() -> None:
            barrier.wait()
            results.append(cfw_module.shared_manager())

        threads = [threading.Thread(target=call) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 6
        assert all(result is results[0] for result in results)
        assert len(starts) == 1


class TestCfwManagerRelease:
    def test_release_resets_the_register_to_its_fresh_state(self) -> None:
        modes = {ParallelizationMode.MULTIPROCESSING}
        register = CfwManager(modes)
        cfw_uuid, left_uuid, right_uuid, dataset_uuid = uuid4(), uuid4(), uuid4(), uuid4()
        register.set_location("grpc://somewhere")
        register.set_error("msg", ("exc_info",), ValueError("boom"))
        register.set_artifact_to_save("artifact", {"k": 1})
        register.set_runtime_artifacts({"runtime": 1})
        register.set_api_data({"api": 1})
        register.set_run_context(RunContext(run_id="run-1"))
        register.add_cfw_to_compute_frameworks(cfw_uuid, "SomeComputeFramework", {uuid4()})
        register.add_to_merge_relation(left_uuid, right_uuid, "SomeComputeFramework")
        register.add_uuid_flyway_datasets(cfw_uuid, {dataset_uuid})
        register.add_join_tokens(cfw_uuid, {uuid4()})

        register.release()

        assert vars(register) == vars(CfwManager(modes))


class TestCfwManagerJoinTokens:
    """Join tokens a reused frame lists are kept apart from compute_frameworks."""

    cls_name = "PythonDictFramework"

    def test_reuse_registered_token_resolves_and_follows_the_merge_relation(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        frame, token, left = uuid4(), uuid4(), uuid4()
        register.add_cfw_to_compute_frameworks(frame, self.cls_name, {uuid4()})
        register.add_join_tokens(frame, {token})

        assert register.get_join_source_cfw_uuid(self.cls_name, token) == frame

        register.add_to_merge_relation(left, frame, self.cls_name)

        assert register.get_join_source_cfw_uuid(self.cls_name, token) == left

    def test_unregistered_token_resolves_to_none(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        register.add_cfw_to_compute_frameworks(uuid4(), self.cls_name, {uuid4()})

        assert register.get_join_source_cfw_uuid(self.cls_name, uuid4()) is None

    def test_add_join_tokens_does_not_change_get_cfw_uuid(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        broad, narrow, feature, token = uuid4(), uuid4(), uuid4(), uuid4()
        register.add_cfw_to_compute_frameworks(broad, self.cls_name, {feature, uuid4()})
        register.add_cfw_to_compute_frameworks(narrow, self.cls_name, {feature})

        register.add_join_tokens(narrow, {token})

        assert register.get_cfw_uuid(self.cls_name, token) is None
        assert register.get_cfw_uuid_as_registered(self.cls_name, token) is None
        assert register.get_unique_cfw_uuid(self.cls_name, {token}) is None
        assert register.get_cfw_uuid(self.cls_name, feature) == narrow

    def test_token_in_a_real_children_if_root_wins_over_a_reuse_registration(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        real, reused, token = uuid4(), uuid4(), uuid4()
        register.add_cfw_to_compute_frameworks(real, self.cls_name, {token})
        register.add_cfw_to_compute_frameworks(reused, self.cls_name, {uuid4()})
        register.add_join_tokens(reused, {token})

        assert register.get_join_source_cfw_uuid(self.cls_name, token) == real

    def test_token_reuse_registered_on_two_distinct_frames_is_ambiguous(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        first, second, token = uuid4(), uuid4(), uuid4()
        register.add_cfw_to_compute_frameworks(first, self.cls_name, {uuid4()})
        register.add_cfw_to_compute_frameworks(second, self.cls_name, {uuid4()})
        register.add_join_tokens(first, {token})
        register.add_join_tokens(second, {token})

        assert register.get_join_source_cfw_uuid(self.cls_name, token) is None

    def test_two_frames_merged_into_one_leftmost_are_not_ambiguous(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        left, right, token = uuid4(), uuid4(), uuid4()
        register.add_cfw_to_compute_frameworks(left, self.cls_name, {uuid4()})
        register.add_cfw_to_compute_frameworks(right, self.cls_name, {uuid4()})
        register.add_join_tokens(left, {token})
        register.add_join_tokens(right, {token})
        register.add_to_merge_relation(left, right, self.cls_name)

        assert register.get_join_source_cfw_uuid(self.cls_name, token) == left

    def test_class_name_filters_reuse_registrations(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        frame, token = uuid4(), uuid4()
        register.add_cfw_to_compute_frameworks(frame, self.cls_name, {uuid4()})
        register.add_join_tokens(frame, {token})

        assert register.get_join_source_cfw_uuid("OtherFramework", token) is None


class TestCfwManagerResolveByTfsIdsReadsJoinResult:
    """The own-feature override replaces a hop's own frame only for a step that reads a join result."""

    @staticmethod
    def _register_root_and_hop() -> tuple[CfwManager, str, Any, Any, Any]:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cls_name = "PythonDictFramework"
        root_cfw, hop_cfw, feature_uuid = uuid4(), uuid4(), uuid4()
        # Both children sets hold the feature but neither contains the other.
        cfw_register.add_cfw_to_compute_frameworks(root_cfw, cls_name, {feature_uuid, uuid4()})
        cfw_register.add_cfw_to_compute_frameworks(hop_cfw, cls_name, {feature_uuid, uuid4()})
        return cfw_register, cls_name, root_cfw, hop_cfw, feature_uuid

    def test_a_step_not_reading_a_join_result_keeps_the_hop_frame(self) -> None:
        cfw_register, cls_name, _, hop_cfw, feature_uuid = self._register_root_and_hop()

        resolved = cfw_register.resolve_cfw_uuid_by_tfs_ids(cls_name, {hop_cfw}, feature_uuid, reads_join_result=False)

        assert resolved == hop_cfw

    def test_a_step_reading_a_join_result_still_gets_the_override(self) -> None:
        cfw_register, cls_name, root_cfw, hop_cfw, feature_uuid = self._register_root_and_hop()

        resolved = cfw_register.resolve_cfw_uuid_by_tfs_ids(cls_name, {hop_cfw}, feature_uuid, reads_join_result=True)

        assert resolved == root_cfw

    def test_the_override_applies_when_the_argument_is_omitted(self) -> None:
        cfw_register, cls_name, root_cfw, hop_cfw, feature_uuid = self._register_root_and_hop()

        assert cfw_register.resolve_cfw_uuid_by_tfs_ids(cls_name, {hop_cfw}, feature_uuid) == root_cfw
