"""Tests for run results that know their run (issue #647 follow-up).

Contract under test:
  * ``mloda.core.api.run_result.RunResult`` is the return type of ``mlodaAPI.run_all``. It IS the
    results list (drop-in for every existing list use) and additionally exposes a read-only
    ``plan`` property returning ``list[PlanStep]``, the resolved plan of the run that produced it.
  * ``mloda.core.api.run_result.ResultStream`` is the return type of ``mlodaAPI.stream_all``. It
    iterates exactly like the old generator, satisfies ``collections.abc.Generator``, is not a
    list, and exposes the same ``plan`` property, available before any element is consumed.
  * ``stream_all`` plans eagerly: an unresolvable request raises at the call, not at iteration.
  * One planning pass per one-shot call: ``run_all`` constructs exactly one Engine and accessing
    ``plan`` afterwards constructs no further Engine.
  * Both classes are exported from ``mloda.user``.

The chained ``plan_info_sales__mean_aggr`` request reuses the feature groups registered by
``tests/test_core/test_api/test_plan_info.py`` so no new registry-wide DataCreator claim is made.
"""

import collections.abc
from typing import Any
from unittest.mock import patch

import pytest

# Aliased: a bare ``import mloda.user`` would bind the name ``mloda`` to the package and collide
# with the ``mloda`` mlodaAPI alias imported below.
import mloda.user as mloda_user
from mloda.core.api.request import Engine
from mloda.user import Feature, ParallelizationMode, PlanStep, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from tests.test_core.test_api.test_plan_info import PlanInfoPandasSource

_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoPandasSource, PandasAggregatedFeatureGroup})

# The chained request from issue #647: an aggregated feature over a source feature.
_CHAINED_FEATURES: list[Feature | str] = ["plan_info_sales__mean_aggr"]

# Two feature groups resolve here, so the stream yields two elements and can be closed early.
_TWO_GROUP_FEATURES: list[Feature | str] = ["plan_info_sales", "plan_info_sales__mean_aggr"]


# Helpers return Any on purpose: the concrete return types (RunResult/ResultStream) are exactly
# what these tests assert at runtime.
def _run_all(features: list[Feature | str] | None = None) -> Any:
    return mlodaAPI.run_all(
        features if features is not None else _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_PLUGINS,
    )


def _stream_all(features: list[Feature | str] | None = None) -> Any:
    return mlodaAPI.stream_all(
        features if features is not None else _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_PLUGINS,
    )


class TestRunAllReturnsRunResult:
    """run_all returns a RunResult that is a list in every observable way."""

    def test_run_all_returns_a_run_result(self) -> None:
        from mloda.user import RunResult

        results = _run_all()

        assert type(results) is RunResult
        assert isinstance(results, list)

    def test_run_result_behaves_like_a_plain_list(self) -> None:
        """Backward-compat pin: len, indexing, iteration and equality with a plain list."""
        results = _run_all()

        as_list = list(results)
        assert results == as_list
        assert len(results) == len(as_list) == 1
        assert results[0] is as_list[0]
        assert [item for item in results] == as_list
        assert "plan_info_sales__mean_aggr" in results[0].columns


class TestRunResultPlan:
    """results.plan is the resolved plan of the run that produced the results."""

    def test_plan_is_a_list_of_plan_steps(self) -> None:
        results = _run_all()

        plan = results.plan

        assert isinstance(plan, list)
        assert len(plan) > 0
        assert all(isinstance(step, PlanStep) for step in plan)

    def test_repeated_plan_access_returns_equal_plans(self) -> None:
        results = _run_all()

        assert results.plan == results.plan

    def test_plan_is_read_only(self) -> None:
        results = _run_all()

        with pytest.raises(AttributeError):
            setattr(results, "plan", [])

    def test_chained_request_plan_contains_the_aggregation_compute_step(self) -> None:
        """Issue #647 DoD: the plan names the concrete aggregation group and framework."""
        results = _run_all()

        compute_steps = [step for step in results.plan if step.step_kind == "compute"]
        aggregation_steps = [step for step in compute_steps if "plan_info_sales__mean_aggr" in step.feature_names]
        assert len(aggregation_steps) == 1, f"expected one aggregation step, got {compute_steps}"

        aggregation = aggregation_steps[0]
        assert aggregation.feature_group is PandasAggregatedFeatureGroup
        assert aggregation.compute_framework is PandasDataFrame

    def test_plan_equals_resolved_plan_of_an_equivalent_prepared_session(self) -> None:
        results = _run_all()

        session = mloda.prepare(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert results.plan == session.resolved_plan()


class TestSinglePlanningPass:
    """run_all plans exactly once; reading the plan afterwards must not re-plan."""

    def test_run_all_constructs_one_engine_and_plan_access_adds_none(self) -> None:
        constructions: list[object] = []

        class CountingEngine(Engine):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                constructions.append(object())
                super().__init__(*args, **kwargs)

        with patch("mloda.core.api.request.Engine", CountingEngine):
            results = _run_all()
            assert len(constructions) == 1

            _ = results.plan
            assert len(constructions) == 1


class TestStreamAllReturnsResultStream:
    """stream_all returns a ResultStream: a Generator with a plan, not a list."""

    def test_stream_all_returns_a_result_stream(self) -> None:
        from mloda.user import ResultStream

        stream = _stream_all()

        assert isinstance(stream, ResultStream)
        assert isinstance(stream, collections.abc.Generator)
        assert not isinstance(stream, list)

    def test_plan_available_before_consumption_and_unchanged_after(self) -> None:
        stream = _stream_all()

        plan_before = stream.plan
        assert isinstance(plan_before, list)
        assert all(isinstance(step, PlanStep) for step in plan_before)
        assert any(step.step_kind == "compute" for step in plan_before)

        results = list(stream)
        assert len(results) == 1

        assert stream.plan == plan_before

    def test_stream_content_still_matches_run_all(self) -> None:
        """Backward-compat pin: list(stream_all(...)) equals run_all(...) content-wise."""
        streamed = list(_stream_all())
        ran = _run_all()

        assert len(streamed) == len(ran) == 1
        assert streamed[0]["plan_info_sales__mean_aggr"].tolist() == ran[0]["plan_info_sales__mean_aggr"].tolist()


class TestStreamAllPlansEagerly:
    """An unresolvable request must raise at the stream_all call, not at first iteration."""

    def test_unresolvable_feature_raises_at_the_call(self) -> None:
        # Deliberately no iteration: the bare call itself must raise.
        with pytest.raises(ValueError, match="No feature groups found"):
            mlodaAPI.stream_all(
                ["run_result_unresolvable_647"],
                compute_frameworks={PandasDataFrame},
                parallelization_modes={ParallelizationMode.SYNC},
                plugin_collector=_PLUGINS,
            )


class TestStreamEarlyClose:
    """A partially consumed stream can be closed without error."""

    def test_close_after_first_element(self) -> None:
        from mloda.user import ResultStream

        stream = _stream_all(_TWO_GROUP_FEATURES)
        assert isinstance(stream, ResultStream)

        first = next(stream)
        assert first is not None

        stream.close()


class TestPublicExports:
    """RunResult and ResultStream are exported from mloda.user and back the entry points."""

    def test_exported_from_user_and_defined_in_run_result_module(self) -> None:
        from mloda.core.api.run_result import ResultStream as ModuleResultStream
        from mloda.core.api.run_result import RunResult as ModuleRunResult
        from mloda.user import ResultStream, RunResult

        assert RunResult is ModuleRunResult
        assert ResultStream is ModuleResultStream
        assert "RunResult" in mloda_user.__all__
        assert "ResultStream" in mloda_user.__all__

    def test_entry_points_return_the_exported_classes(self) -> None:
        from mloda.user import ResultStream, RunResult

        assert type(_run_all()) is RunResult
        assert isinstance(_stream_all(), ResultStream)


class TestSubclassDispatch:
    """Subclassed entry points keep dispatching and keep the new return type."""

    def test_subclass_run_all_returns_a_run_result(self) -> None:
        from mloda.user import RunResult

        class CustomAPI(mlodaAPI):
            pass

        results = CustomAPI.run_all(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_PLUGINS,
        )

        assert isinstance(results, RunResult)
        assert len(results) == 1
