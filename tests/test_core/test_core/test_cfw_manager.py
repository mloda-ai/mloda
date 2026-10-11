"""Tests for CfwManager RunContext get/set round-trip and merge-relation cycles."""

import os
import sys
import threading
import time
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

cfw_module: Any = cfw_manager_module


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

    @pytest.mark.timeout(30)
    def test_after_shutdown_the_next_call_returns_a_different_alive_manager(self) -> None:
        first = cfw_module.shared_manager()
        first.shutdown()

        second = cfw_module.shared_manager()

        assert second is not first
        assert second.is_alive()


@pytest.mark.usefixtures("isolated_shared_manager")
class TestSharedManagerWithAFakedStart:
    @staticmethod
    def _fake_start(monkeypatch: pytest.MonkeyPatch, delay: float = 0.0) -> list[MyManager]:
        starts: list[MyManager] = []

        def fake_start(self: MyManager, initializer: Any = None, initargs: Any = ()) -> None:
            time.sleep(delay)
            starts.append(self)

        monkeypatch.setattr(MyManager, "start", fake_start)
        monkeypatch.setattr(MyManager, "is_alive", lambda self: True, raising=False)
        return starts

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

        register.release()

        assert vars(register) == vars(CfwManager(modes))
