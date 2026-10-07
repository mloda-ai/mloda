"""Tests for the plan/run lifecycle hooks on Extender and the LifecycleOutcome they receive."""

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.plan_context import PlanContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import FeatureResolutionError, PlanStep, plan_content_hash, plan_structure_hash
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_COLUMN = "lifecycle_hooks_col"


class _LifecycleFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {_COLUMN: [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_LifecycleFeatureGroup})
_SYNC = {ParallelizationMode.SYNC}


def _prepare(extenders: set[Extender]) -> Any:
    return mloda.prepare(
        [Feature(name=_COLUMN)],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED,
        parallelization_modes=_SYNC,
        function_extender=extenders,
    )


class _Recorder(Extender):
    def __init__(
        self, label: str, log: list[tuple[Any, ...]], priority: int = 100, raises_in: str | None = None
    ) -> None:
        self.label = label
        self.log = log
        self.priority = priority
        self.raises_in = raises_in

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_MATCHED}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.log.append((self.label, "matched"))
        return func(*args, **kwargs)

    def _record(self, hook: str, *payload: Any) -> None:
        self.log.append((self.label, hook, *payload))
        if self.raises_in == hook:
            raise RuntimeError(f"{hook}-boom")

    def on_plan_start(self, plan: Any) -> None:
        self._record("plan_start", plan)

    def on_plan_complete(self, plan: Any, outcome: Any) -> None:
        self._record("plan_complete", plan, outcome)

    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        self._record("run_start", run, plan, steps)

    def on_run_complete(self, run: Any, outcome: Any) -> None:
        self._record("run_complete", run, outcome)


class _Minimal(Extender):
    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class TestLifecycleOutcome:
    def test_is_a_frozen_dataclass_with_status_and_error_type(self) -> None:
        from mloda.steward import LifecycleOutcome

        outcome = LifecycleOutcome(status="failed", error_type="ValueError")

        assert (outcome.status, outcome.error_type) == ("failed", "ValueError")
        assert outcome == LifecycleOutcome(status="failed", error_type="ValueError")
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.status = "succeeded"  # type: ignore[misc]


class TestBaseExtenderLifecycleDefaults:
    def test_all_four_hooks_are_noops_on_a_minimal_extender(self) -> None:
        from datetime import datetime, timezone

        from mloda.steward import LifecycleOutcome

        extender = _Minimal()
        plan = PlanContext(
            plan_id="p", tenant_id=None, project_id=None, principal=None, created_at=datetime.now(timezone.utc)
        )
        run = RunContext(run_id="r", plan_id="p")
        outcome = LifecycleOutcome(status="succeeded", error_type=None)

        extender.on_plan_start(plan)
        extender.on_plan_complete(plan, outcome)
        extender.on_run_start(run, plan, ())
        extender.on_run_complete(run, outcome)


class _OldSignatureExtender(_Minimal):
    def on_run_complete(self, run_id: str | None) -> None:  # type: ignore[override]
        pass


class TestOldOneParameterOnRunCompleteIsRejectedAtSetup:
    @pytest.mark.parametrize("entry", ["prepare", "run_all"])
    def test_raises_type_error_naming_the_extender(self, entry: str) -> None:
        with pytest.raises(TypeError, match="_OldSignatureExtender"):
            if entry == "prepare":
                _prepare({_OldSignatureExtender()})
            else:
                mloda.run_all(
                    [Feature(name=_COLUMN)],
                    compute_frameworks=["PythonDictFramework"],
                    plugin_collector=_ENABLED,
                    function_extender={_OldSignatureExtender()},
                )

    def test_error_names_on_run_complete(self) -> None:
        with pytest.raises(TypeError, match="on_run_complete"):
            _prepare({_OldSignatureExtender()})


class TestHookArguments:
    def test_plan_hooks_receive_the_plan_context_and_a_succeeded_outcome(self) -> None:
        log: list[tuple[Any, ...]] = []

        session = _prepare({_Recorder("r", log)})

        starts = [e for e in log if e[1] == "plan_start"]
        completes = [e for e in log if e[1] == "plan_complete"]
        assert len(starts) == 1 and len(completes) == 1
        plan = starts[0][2]
        assert isinstance(plan, PlanContext)
        assert plan == session.plan_context
        assert completes[0][2] == plan
        assert completes[0][3].status == "succeeded"
        assert completes[0][3].error_type is None

    def test_structure_hash_is_none_at_plan_start_and_set_from_plan_complete_on(self) -> None:
        log: list[tuple[Any, ...]] = []
        session = _prepare({_Recorder("r", log)})
        session.run(parallelization_modes=_SYNC)

        (start,) = [e for e in log if e[1] == "plan_start"]
        (complete,) = [e for e in log if e[1] == "plan_complete"]
        (run_start,) = [e for e in log if e[1] == "run_start"]
        expected = plan_structure_hash(session.resolved_plan())
        assert start[2].structure_hash is None
        assert complete[2].structure_hash is not None
        assert complete[2].structure_hash == expected
        assert run_start[3].structure_hash == expected
        assert run_start[2].structure_hash == expected
        assert session.plan_context.structure_hash == expected
        expected_content = plan_content_hash(session.resolved_plan())
        assert start[2].content_hash is None
        assert complete[2].content_hash == expected_content
        assert run_start[3].content_hash == expected_content
        assert session.plan_context.content_hash == expected_content
        assert start[2] == complete[2]
        assert start[2].plan_id == complete[2].plan_id

    def test_a_failed_plan_keeps_structure_hash_none(self) -> None:
        log: list[tuple[Any, ...]] = []

        with pytest.raises(FeatureResolutionError):
            mloda.prepare(
                [Feature(name="lifecycle_hooks_missing_col")],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                parallelization_modes=_SYNC,
                function_extender={_Recorder("r", log)},
            )

        (complete,) = [e for e in log if e[1] == "plan_complete"]
        assert complete[3].status == "failed"
        assert complete[2].structure_hash is None
        assert complete[2].content_hash is None

    def test_a_session_without_extenders_has_no_structure_hash(self) -> None:
        session = mloda.prepare(
            [Feature(name=_COLUMN)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes=_SYNC,
        )

        assert session.plan_context.structure_hash is None
        assert session.plan_context.content_hash is None
        assert session._build_run_context(None, None).structure_hash is None

    def test_plan_start_fires_before_the_match_hook_and_plan_complete_after_it(self) -> None:
        log: list[tuple[Any, ...]] = []

        _prepare({_Recorder("r", log)})

        assert [e[1] for e in log] == ["plan_start", "matched", "plan_complete"]

    def test_run_start_receives_run_plan_and_a_tuple_of_plan_steps(self) -> None:
        log: list[tuple[Any, ...]] = []
        session = _prepare({_Recorder("r", log)})

        session.run(parallelization_modes=_SYNC)

        (start,) = [e for e in log if e[1] == "run_start"]
        _, _, run, plan, steps = start
        assert isinstance(run, RunContext)
        assert isinstance(plan, PlanContext)
        assert run.plan_id == plan.plan_id == session.plan_id
        assert isinstance(steps, tuple)
        assert steps and all(isinstance(s, PlanStep) for s in steps)
        assert steps == tuple(session.resolved_plan())

    def test_run_complete_receives_the_same_run_and_a_succeeded_outcome(self) -> None:
        log: list[tuple[Any, ...]] = []
        session = _prepare({_Recorder("r", log)})

        session.run(parallelization_modes=_SYNC)

        (start,) = [e for e in log if e[1] == "run_start"]
        (complete,) = [e for e in log if e[1] == "run_complete"]
        assert complete[2].run_id == start[2].run_id
        assert complete[3].status == "succeeded"
        assert complete[3].error_type is None


_MISSING = "lifecycle_hooks_missing_col"


def _kwargs(ext: Extender) -> dict[str, Any]:
    return {"compute_frameworks": ["PythonDictFramework"], "plugin_collector": _ENABLED, "function_extender": {ext}}


def _origin_run_all(ext: Extender) -> None:
    mloda.run_all([Feature(name=_COLUMN)], parallelization_modes=_SYNC, **_kwargs(ext))


def _origin_stream_all(ext: Extender) -> None:
    list(mloda.stream_all([Feature(name=_COLUMN)], parallelization_modes=_SYNC, **_kwargs(ext)))


def _origin_prepare(ext: Extender) -> None:
    mloda.prepare([Feature(name=_COLUMN)], parallelization_modes=_SYNC, **_kwargs(ext))


def _origin_direct(ext: Extender) -> None:
    mloda([Feature(name=_COLUMN)], parallelization_modes=_SYNC, **_kwargs(ext))


def _origin_explain(ext: Extender) -> None:
    mloda.explain([Feature(name=_COLUMN)], **_kwargs(ext))


def _origin_diagnose(ext: Extender) -> None:
    assert mloda.diagnose([Feature(name=_COLUMN)], **_kwargs(ext)).complete


def _origin_diagnose_failing(ext: Extender) -> None:
    assert not mloda.diagnose([Feature(name=_MISSING)], **_kwargs(ext)).complete


class _OriginExplainer(_Recorder):
    """On the first plan_start, plans a nested explain."""

    def __init__(self, label: str, log: list[tuple[Any, ...]]) -> None:
        super().__init__(label, log)
        self.nested = False

    def on_plan_start(self, plan: Any) -> None:
        super().on_plan_start(plan)
        if not self.nested:
            self.nested = True
            mloda.explain([Feature(name=_COLUMN)], **_kwargs(self))


class TestPlanContextOrigin:
    @pytest.mark.parametrize(
        "entry, expected",
        [
            (_origin_run_all, "run_all"),
            (_origin_stream_all, "stream_all"),
            (_origin_prepare, "prepare"),
            (_origin_direct, "prepare"),
            (_origin_explain, "explain"),
            (_origin_diagnose, "diagnose"),
            (_origin_diagnose_failing, "diagnose"),
        ],
    )
    def test_plan_hooks_see_the_entry_point_origin(self, entry: Callable[[Extender], None], expected: str) -> None:
        log: list[tuple[Any, ...]] = []

        entry(_Recorder("r", log))

        (start,) = [e for e in log if e[1] == "plan_start"]
        (complete,) = [e for e in log if e[1] == "plan_complete"]
        assert start[2].origin == expected
        assert complete[2].origin == expected

    @pytest.mark.parametrize("entry, expected", [(_origin_run_all, "run_all"), (_origin_stream_all, "stream_all")])
    def test_run_start_and_session_carry_the_origin(self, entry: Callable[[Extender], None], expected: str) -> None:
        log: list[tuple[Any, ...]] = []

        entry(_Recorder("r", log))

        (run_start,) = [e for e in log if e[1] == "run_start"]
        assert run_start[3].origin == expected

    def test_session_plan_context_carries_the_origin(self) -> None:
        session = mloda.prepare([Feature(name=_COLUMN)], parallelization_modes=_SYNC, **_kwargs(_Minimal()))

        assert session.plan_context.origin == "prepare"

    def test_nested_planning_gets_its_own_origin(self) -> None:
        log: list[tuple[Any, ...]] = []

        _origin_run_all(_OriginExplainer("r", log))

        origins = [e[2].origin for e in log if e[1] == "plan_start"]
        assert origins == ["run_all", "explain"]
        outer_complete = [e[2].origin for e in log if e[1] == "plan_complete"]
        assert outer_complete[-1] == "run_all"

    @pytest.mark.parametrize("first", [_origin_diagnose, _origin_explain, _origin_diagnose_failing])
    def test_origin_does_not_leak_into_a_following_prepare(self, first: Callable[[Extender], None]) -> None:
        first(_Minimal())
        log: list[tuple[Any, ...]] = []

        _origin_prepare(_Recorder("r", log))

        assert [e[2].origin for e in log if e[1] == "plan_start"] == ["prepare"]

    def test_origin_does_not_leak_after_a_raising_entry_point(self) -> None:
        with pytest.raises(FeatureResolutionError):
            mloda.run_all([Feature(name=_MISSING)], parallelization_modes=_SYNC, **_kwargs(_Minimal()))
        log: list[tuple[Any, ...]] = []

        _origin_prepare(_Recorder("r", log))

        assert [e[2].origin for e in log if e[1] == "plan_start"] == ["prepare"]


class TestHooksRunInAscendingPriorityOrder:
    @pytest.mark.parametrize("hook", ["plan_start", "plan_complete", "run_start", "run_complete"])
    def test_every_hook_visits_extenders_by_priority(self, hook: str) -> None:
        log: list[tuple[Any, ...]] = []
        priorities = [50, 20, 60, 10, 40, 30]
        session = _prepare({_Recorder(f"p{p}", log, priority=p) for p in priorities})
        session.run(parallelization_modes=_SYNC)

        order = [e[0] for e in log if e[1] == hook]

        assert order == [f"p{p}" for p in sorted(priorities)]

    @pytest.mark.parametrize("hook", ["plan_start", "plan_complete", "run_start", "run_complete"])
    def test_every_hook_visits_gates_before_other_extenders_regardless_of_priority(self, hook: str) -> None:
        log: list[tuple[Any, ...]] = []
        late_gate = _Recorder("gate", log, priority=900)
        late_gate.never_fall_back = True
        early = _Recorder("early", log, priority=1)
        session = _prepare({early, late_gate})
        session.run(parallelization_modes=_SYNC)

        assert [e[0] for e in log if e[1] == hook] == ["gate", "early"]

    def test_an_extender_wrapping_no_hook_still_gets_every_lifecycle_hook(self) -> None:
        seen: list[str] = []

        class _Silent(_Minimal):
            def on_plan_start(self, plan: Any) -> None:
                seen.append("plan_start")

            def on_plan_complete(self, plan: Any, outcome: Any) -> None:
                seen.append("plan_complete")

            def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
                seen.append("run_start")

            def on_run_complete(self, run: Any, outcome: Any) -> None:
                seen.append("run_complete")

        _prepare({_Silent()}).run(parallelization_modes=_SYNC)

        assert seen == ["plan_start", "plan_complete", "run_start", "run_complete"]


class TestContainedHooksLogAndContinue:
    @pytest.mark.parametrize("hook", ["plan_start", "plan_complete", "run_complete"])
    def test_a_raising_extender_is_logged_and_later_extenders_still_run(
        self, hook: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: list[tuple[Any, ...]] = []
        raiser = _Recorder("raiser", log, priority=10, raises_in=hook)
        survivor = _Recorder("survivor", log, priority=20)

        with caplog.at_level(logging.ERROR):
            session = _prepare({raiser, survivor})
            session.run(parallelization_modes=_SYNC)

        assert [e[0] for e in log if e[1] == hook] == ["raiser", "survivor"]
        records = [r for r in caplog.records if r.levelno == logging.ERROR and f"{hook}-boom" in r.getMessage()]
        assert len(records) == 1
        assert "RuntimeError" in records[0].getMessage()
        assert records[0].exc_info is None
        args = records[0].args
        arg_values = args.values() if isinstance(args, Mapping) else (args or ())
        assert not [a for a in arg_values if isinstance(a, BaseException)]


class _BreakingRefuser(_Recorder):
    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        self.log.append((self.label, "run_start"))
        raise RuntimeError("refused-at-run-start")


class _GateRefuser(_BreakingRefuser):
    never_fall_back = True

    def __init__(self, label: str, log: list[tuple[Any, ...]], priority: int = 100) -> None:
        super().__init__(label, log, priority)
        self.raise_on_error = False


class _WarningOnlyRefuser(_BreakingRefuser):
    def __init__(self, label: str, log: list[tuple[Any, ...]], priority: int = 100) -> None:
        super().__init__(label, log, priority)
        self.raise_on_error = False


class _Outer:
    class Inner(Exception):
        pass


class _InnerRefuser(_BreakingRefuser):
    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        raise _Outer.Inner("nested-boom")


class TestOnRunStartRefusalSemantics:
    @pytest.mark.parametrize("refuser_type", [_BreakingRefuser, _GateRefuser])
    def test_raise_on_error_or_never_fall_back_propagates_and_the_run_is_refused(
        self, refuser_type: type[_BreakingRefuser]
    ) -> None:
        log: list[tuple[Any, ...]] = []
        session = _prepare({refuser_type("refuser", log)})

        with pytest.raises(RuntimeError, match="refused-at-run-start"):
            session.run(parallelization_modes=_SYNC)

        complete = [e for e in log if e[1] == "run_complete"]
        assert [e[3].status for e in complete] == ["failed"]
        assert complete[0][3].error_type == "builtins.RuntimeError"

    def test_a_warning_only_extender_is_logged_and_the_run_proceeds(self, caplog: pytest.LogCaptureFixture) -> None:
        log: list[tuple[Any, ...]] = []
        calculated: list[str] = []
        session = _prepare({_WarningOnlyRefuser("warner", log), _Calculated(calculated)})

        with caplog.at_level(logging.WARNING):
            result = session.run(parallelization_modes=_SYNC)

        assert result and calculated == ["calculated"]
        assert any("refused-at-run-start" in r.getMessage() for r in caplog.records)
        assert [e[3].status for e in log if e[1] == "run_complete"] == ["succeeded"]

    def test_a_refusal_stops_later_extenders_from_seeing_run_start_but_they_still_see_run_complete(self) -> None:
        log: list[tuple[Any, ...]] = []
        session = _prepare({_BreakingRefuser("first", log, priority=10), _Recorder("later", log, priority=20)})

        with pytest.raises(RuntimeError, match="refused-at-run-start"):
            session.run(parallelization_modes=_SYNC)

        later = [e[1] for e in log if e[0] == "later" and e[1].startswith("run_")]
        assert later == ["run_complete"]

    def test_a_refused_run_never_calculates(self) -> None:
        log: list[tuple[Any, ...]] = []
        calculated: list[str] = []
        session = _prepare({_BreakingRefuser("refuser", log, priority=10), _Calculated(calculated)})

        with pytest.raises(RuntimeError, match="refused-at-run-start"):
            session.run(parallelization_modes=_SYNC)

        assert calculated == []

    def test_nested_exception_class_reports_module_and_qualname(self) -> None:
        log: list[tuple[Any, ...]] = []
        session = _prepare({_InnerRefuser("refuser", log), _Recorder("later", log, priority=200)})

        with pytest.raises(_Outer.Inner):
            session.run(parallelization_modes=_SYNC)

        complete = [e for e in log if e[0] == "later" and e[1] == "run_complete"]
        assert complete[0][3].error_type == f"{__name__}._Outer.Inner"


class _Calculated(Extender):
    def __init__(self, calculated: list[str]) -> None:
        self.priority = 500
        self.calculated = calculated

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calculated.append("calculated")
        return func(*args, **kwargs)
