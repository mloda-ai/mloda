"""Tests for ``mlodaAPI.run_all_with_plan``: the plan of the run that actually happened.

Why a new entry point at all:
  ``run_all()`` builds a session, runs it and throws it away, so the plan of an actual run is
  unreachable from the most-used entry point. ``explain()`` does not close that gap: it re-resolves
  a *fresh* plan, answering "what would this request resolve to", not "what served that run".

Contract under test:
  * ``mlodaAPI.run_all_with_plan(features, ...) -> tuple[list[Any], list[PlanStep]]`` is a
    ``@classmethod`` whose signature is identical to ``run_all``'s (same names, order, kinds and
    defaults), so a kwarg added to ``run_all`` cannot silently skip it.
  * Every parameter is forwarded exactly as ``run_all`` forwards it. The signature parity above
    guards the signature only: a new kwarg could be declared here and then dropped in the body.
    ``TestForwardingParityWithRunAll`` closes that hole by comparing what actually reaches
    ``prepare`` and ``run``.
  * ``results`` is exactly what ``run_all(...)`` returns for the same inputs.
  * ``plan`` is the plan of *that* run: one planning pass (prepare -> run -> resolved_plan), never
    a re-plan, and it equals the plan the runner actually executed.
  * Planning happens under ``run_all``'s parameter semantics, in particular its
    ``parallelization_modes={ParallelizationMode.SYNC}`` default, which ``prepare``/``explain``
    do not share (they default to ``None``) and which ``SetupComputeFramework`` uses to filter
    compute frameworks.

This is a separate module from ``test_plan_info.py`` on purpose: that module asserts (in
``TestCallerNeedsNoInternalImport``) that it imports nothing from ``mloda.core.core``, and the
"this plan is the executed plan, resolved exactly once" claims here can only be pinned by spying on
``mloda.core.core.engine.Engine``.

Feature names carry a ``run_all_plan_`` prefix: a ``DataCreator`` claim is registry-wide, so generic
names would leak into every other test.
"""

import inspect
from collections.abc import Callable
from itertools import chain
from typing import Any, Optional

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.api.plan_info import build_plan_steps
from mloda.core.core.engine import Engine
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PlanStep, PluginCollector, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


class RunAllPlanSource(FeatureGroup):
    """Pandas root source the chained aggregation request builds on."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"run_all_plan_sales", "run_all_plan_price"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"run_all_plan_sales": [100, 200, 300], "run_all_plan_price": [1.0, 2.0, 3.0]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_PLUGINS = PluginCollector.enabled_feature_groups({RunAllPlanSource, PandasAggregatedFeatureGroup})

# A derived feature chained off a root source: the plan must name both concrete feature groups.
_CHAINED_FEATURES: list[Feature | str] = ["run_all_plan_sales__mean_aggr"]


def _run_all_with_plan() -> tuple[list[Any], list[PlanStep]]:
    return mlodaAPI.run_all_with_plan(
        _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        plugin_collector=_PLUGINS,
    )


def _run_all() -> list[Any]:
    return mlodaAPI.run_all(
        _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        plugin_collector=_PLUGINS,
    )


# ---------------------------------------------------------------------------
# 1. Results equivalence
# ---------------------------------------------------------------------------


class TestResultsEquivalence:
    """``run_all_with_plan`` must not be a second, subtly different execution path."""

    def test_returns_results_and_plan(self) -> None:
        returned = _run_all_with_plan()

        assert isinstance(returned, tuple)
        assert len(returned) == 2

        results, plan = returned
        assert isinstance(results, list)
        assert isinstance(plan, list)

    def test_results_equal_run_all_results(self) -> None:
        results, _plan = _run_all_with_plan()
        expected = _run_all()

        assert len(results) == len(expected)
        for actual_frame, expected_frame in zip(results, expected):
            pd.testing.assert_frame_equal(actual_frame, expected_frame)

    def test_results_contain_the_requested_feature(self) -> None:
        results, _plan = _run_all_with_plan()

        assert len(results) == 1
        assert "run_all_plan_sales__mean_aggr" in results[0].columns


# ---------------------------------------------------------------------------
# 2. Plan correctness
# ---------------------------------------------------------------------------


class TestPlanCorrectness:
    """Mirrors the definition-of-done case of ``test_plan_info.py``, but for an executed run."""

    def test_plan_is_a_non_empty_list_of_plan_steps(self) -> None:
        _results, plan = _run_all_with_plan()

        assert len(plan) > 0
        assert all(isinstance(step, PlanStep) for step in plan)

    def test_compute_steps_carry_the_concrete_group_and_framework(self) -> None:
        _results, plan = _run_all_with_plan()

        compute_steps = [step for step in plan if step.step_kind == "compute"]
        assert len(compute_steps) == 2, f"expected two compute steps, got {[s.step_kind for s in plan]}"

        source_step, aggregation_step = compute_steps

        assert source_step.feature_names == ("run_all_plan_sales",)
        assert source_step.feature_group is RunAllPlanSource
        assert source_step.compute_framework is PandasDataFrame

        assert aggregation_step.feature_names == ("run_all_plan_sales__mean_aggr",)
        assert aggregation_step.feature_group is PandasAggregatedFeatureGroup
        assert aggregation_step.compute_framework is PandasDataFrame

        assert aggregation_step.feature_group_name == "PandasAggregatedFeatureGroup"
        assert aggregation_step.compute_framework_name == "PandasDataFrame"

    def test_plan_is_in_execution_order(self) -> None:
        _results, plan = _run_all_with_plan()

        groups = [step.feature_group for step in plan if step.step_kind == "compute"]
        assert groups.index(RunAllPlanSource) < groups.index(PandasAggregatedFeatureGroup)


# ---------------------------------------------------------------------------
# 3. classmethod + signature parity with run_all
# ---------------------------------------------------------------------------


class TestSignatureMatchesRunAll:
    """A kwarg added to ``run_all`` must not silently skip ``run_all_with_plan``."""

    def test_is_a_classmethod(self) -> None:
        assert isinstance(inspect.getattr_static(mlodaAPI, "run_all_with_plan"), classmethod)
        # Accessed on the class it is already bound, exactly like run_all.
        assert inspect.ismethod(mlodaAPI.run_all_with_plan)

    def test_parameters_match_run_all_exactly(self) -> None:
        """Same names, same order, same kinds, same defaults. ``cls`` is bound away on both."""
        run_all_params = list(inspect.signature(mlodaAPI.run_all).parameters.values())
        with_plan_params = list(inspect.signature(mlodaAPI.run_all_with_plan).parameters.values())

        assert with_plan_params == run_all_params

    def test_defaults_match_run_all(self) -> None:
        """Spelled out separately: an equal-but-differently-defaulted parameter is the whole footgun."""
        run_all_defaults = {
            name: param.default for name, param in inspect.signature(mlodaAPI.run_all).parameters.items()
        }
        with_plan_defaults = {
            name: param.default for name, param in inspect.signature(mlodaAPI.run_all_with_plan).parameters.items()
        }

        assert with_plan_defaults == run_all_defaults
        assert with_plan_defaults["parallelization_modes"] == {ParallelizationMode.SYNC}


# ---------------------------------------------------------------------------
# 4. Forwarding parity with run_all: the body, not just the signature
# ---------------------------------------------------------------------------


class _Sentinel:
    """A unique value for one parameter. Compares by identity, so a swap or a drop cannot pass."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<sentinel {self.name}>"


# ``prepare`` is a classmethod accessed on the class, so ``cls`` is already bound away; ``run`` is a
# plain method, so drop ``self``. Captured at import time, before any test patches them.
_PREPARE_SIGNATURE = inspect.signature(mlodaAPI.prepare)
_RUN_SIGNATURE = inspect.signature(mlodaAPI.run).replace(
    parameters=[p for name, p in inspect.signature(mlodaAPI.run).parameters.items() if name != "self"]
)

_RUN_RESULTS: list[Any] = [_Sentinel("run-results")]
_RUN_PLAN: list[PlanStep] = []


def _bind(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize one call to ``{parameter_name: value}``, defaults filled in.

    Positional and keyword forwarding of the same value are the same call, and an *omitted* argument
    shows up as its default, which is exactly how a dropped argument must be detected.
    """
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


class _RecordingSession:
    """Stands in for the session ``prepare`` returns, recording what ``run`` is called with."""

    def __init__(self, run_calls: list[dict[str, Any]]) -> None:
        self.run_calls = run_calls

    def run(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.run_calls.append(_bind(_RUN_SIGNATURE, args, kwargs))
        return _RUN_RESULTS

    def resolved_plan(self) -> list[PlanStep]:
        return _RUN_PLAN


def _capture(
    monkeypatch: pytest.MonkeyPatch, entry_point: Callable[..., Any], arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Call ``entry_point(**arguments)`` with ``prepare``/``run`` stubbed out, return what they saw.

    Only the public surface is observed: which arguments reach ``prepare`` and which reach the
    session's ``run``. Whether the entry point does that inline or through a shared private helper is
    none of this test's business, so a later refactor cannot break it.
    """
    prepare_calls: list[dict[str, Any]] = []
    run_calls: list[dict[str, Any]] = []

    def fake_prepare(cls: type[mlodaAPI], *args: Any, **kwargs: Any) -> _RecordingSession:
        prepare_calls.append(_bind(_PREPARE_SIGNATURE, args, kwargs))
        return _RecordingSession(run_calls)

    monkeypatch.setattr(mlodaAPI, "prepare", classmethod(fake_prepare))
    returned = entry_point(**arguments)

    assert len(prepare_calls) == 1, f"expected exactly one prepare call, got {len(prepare_calls)}"
    assert len(run_calls) == 1, f"expected exactly one run call, got {len(run_calls)}"
    return prepare_calls[0], run_calls[0], returned


def _sentinel_arguments() -> dict[str, Any]:
    """One distinct sentinel per ``run_all`` parameter, derived from the signature.

    Derived, not hand-listed: a parameter added to ``run_all`` tomorrow is covered without touching
    this test. Nothing validates these values, because ``prepare`` and ``run`` are stubbed out.
    """
    return {name: _Sentinel(name) for name in inspect.signature(mlodaAPI.run_all).parameters}


class TestForwardingParityWithRunAll:
    """Signature parity guards the signature; this guards the body.

    Without it, a kwarg added to ``run_all`` and forwarded there can be forced by the signature test
    into ``run_all_with_plan``'s signature and still be silently dropped on the way to ``prepare`` or
    ``run``. Comparing the *calls* both entry points make catches that for every parameter, present
    or future.
    """

    def test_prepare_receives_identical_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        arguments = _sentinel_arguments()

        run_all_prepare, _run_all_run, _r = _capture(monkeypatch, mlodaAPI.run_all, arguments)
        with_plan_prepare, _with_plan_run, _p = _capture(monkeypatch, mlodaAPI.run_all_with_plan, arguments)

        assert with_plan_prepare == run_all_prepare

    def test_run_receives_identical_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        arguments = _sentinel_arguments()

        _run_all_prepare, run_all_run, _r = _capture(monkeypatch, mlodaAPI.run_all, arguments)
        _with_plan_prepare, with_plan_run, _p = _capture(monkeypatch, mlodaAPI.run_all_with_plan, arguments)

        assert with_plan_run == run_all_run

    def test_no_parameter_is_dropped_on_the_way_to_prepare_or_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parity alone would still pass if *both* entry points dropped a parameter. This does not."""
        arguments = _sentinel_arguments()

        prepare_call, run_call, _returned = _capture(monkeypatch, mlodaAPI.run_all_with_plan, arguments)
        forwarded = list(chain(prepare_call.values(), run_call.values()))

        for name, sentinel in arguments.items():
            assert any(value is sentinel for value in forwarded), f"{name} never reached prepare or run"

    def test_returns_the_run_results_and_the_session_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The results are the run's own; the plan is the session's, not a re-resolved one."""
        arguments = _sentinel_arguments()

        _prepare_call, _run_call, returned = _capture(monkeypatch, mlodaAPI.run_all_with_plan, arguments)

        results, plan = returned
        assert results is _RUN_RESULTS
        assert plan is _RUN_PLAN


# ---------------------------------------------------------------------------
# 5. The plan reported is the plan that executed, resolved once
# ---------------------------------------------------------------------------


class TestPlanIsTheExecutedPlanNotAReplay:
    def test_resolves_the_plan_exactly_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One planning pass: prepare -> run -> resolved_plan.

        Each planning pass constructs an ``Engine``. An implementation that ran and then re-resolved
        (``run_all`` followed by ``explain``) would construct two, and would be exactly the bug this
        entry point exists to remove.
        """
        engine_constructions: list[Engine] = []
        original_init = Engine.__init__

        def counting_init(engine_self: Engine, *args: Any, **kwargs: Any) -> None:
            engine_constructions.append(engine_self)
            original_init(engine_self, *args, **kwargs)

        monkeypatch.setattr(Engine, "__init__", counting_init)

        _results, plan = _run_all_with_plan()

        assert len(engine_constructions) == 1, f"expected a single planning pass, got {len(engine_constructions)}"
        assert len(plan) > 0

    def test_plan_equals_the_plan_the_runner_executed(self) -> None:
        """The plan a run reports is the plan *its own* runner executed.

        ``run_all_with_plan`` is ``prepare -> run -> resolved_plan`` on one session, so the claim
        "this is the plan of that run" rests on a single session-level invariant: after ``run()``,
        that session's ``resolved_plan()`` equals the plan its runner executed. ``Engine.compute``
        deep-copies the planner before executing, so the runner's planner *is* what ran, and a bug
        there would make the plan a story about an execution that never happened.

        Pinned directly on one session (same session in, same session out), not on a twin: comparing
        two sessions would only prove planning is deterministic across sessions.
        """
        session = mlodaAPI.prepare(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        session.run()

        assert session.runner is not None
        executed_plan = build_plan_steps(session.runner.execution_planner)

        assert len(executed_plan) > 0
        assert session.resolved_plan() == executed_plan


# ---------------------------------------------------------------------------
# 6. parallelization_modes: run_all's defaults, not prepare's/explain's
# ---------------------------------------------------------------------------


class TestParallelizationModeSemantics:
    """``run_all`` defaults ``parallelization_modes`` to ``{SYNC}``; ``prepare``/``explain`` default
    it to ``None``, and ``SetupComputeFramework`` filters compute frameworks by mode. So
    ``explain(...)`` and ``run_all(...)`` with the same arguments can resolve different plans.

    Empirically (see the report accompanying these tests): every ComputeFramework shipped in this
    repo supports SYNC, so with shipped frameworks the ``{SYNC}`` filter removes nothing and the two
    defaults resolve the same plan. The divergence needs a third-party framework that omits SYNC.
    These tests therefore pin what is true and load-bearing: the plan reported is the plan resolved
    under ``run_all``'s mode semantics, and the modes really do reach the planner.
    """

    def test_plan_is_the_sync_resolved_plan(self) -> None:
        _results, plan = _run_all_with_plan()

        sync_explained = mlodaAPI.explain(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert plan == sync_explained

    def test_parallelization_modes_reach_the_planner(self) -> None:
        """SqliteFramework supports SYNC only, so requesting THREADING must be rejected *while
        planning*, exactly as ``run_all`` rejects it.

        A ``run_all_with_plan`` that prepared mode-blind (like ``prepare``/``explain`` default to)
        would not raise this error, it would fail later and differently.
        """
        with pytest.raises(ValueError, match="parallelization modes"):
            mlodaAPI.run_all(
                _CHAINED_FEATURES,
                compute_frameworks={SqliteFramework},
                parallelization_modes={ParallelizationMode.THREADING},
                plugin_collector=_PLUGINS,
            )

        with pytest.raises(ValueError, match="parallelization modes"):
            mlodaAPI.run_all_with_plan(
                _CHAINED_FEATURES,
                compute_frameworks={SqliteFramework},
                parallelization_modes={ParallelizationMode.THREADING},
                plugin_collector=_PLUGINS,
            )

    def test_explicit_modes_are_accepted_positionally_like_run_all(self) -> None:
        """``parallelization_modes`` is ``run_all``'s 5th positional parameter. Same call shape here."""
        results, plan = mlodaAPI.run_all_with_plan(
            _CHAINED_FEATURES,
            {PandasDataFrame},
            None,
            None,
            {ParallelizationMode.SYNC},
            plugin_collector=_PLUGINS,
        )

        assert len(results) == 1
        assert [step.feature_group for step in plan if step.step_kind == "compute"] == [
            RunAllPlanSource,
            PandasAggregatedFeatureGroup,
        ]
