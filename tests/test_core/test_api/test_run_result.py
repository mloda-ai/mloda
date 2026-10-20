"""Tests for run results that know their run.

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
  * Both classes expose read-only ``run_id``/``plan_id``, the run's ``HookContext`` ids, set on a
    stream before consumption.
  * Both classes are exported from ``mloda.user``.
  * ``RunResult.frames()`` pairs each frame with the compute ``PlanStep`` that produced it, by
    ``step_uuid``, not by list position.
  * ``RunResult.frame_for(feature)`` returns the frame whose compute step requested the feature: a
    ``str`` matches by name, a ``Feature`` also by a subset of its group options; no match or an
    ambiguous name raises ``ValueError``.

The chained ``plan_info_sales__mean_aggr`` request reuses the feature groups registered by
``tests/test_core/test_api/test_plan_info.py`` so no new registry-wide DataCreator claim is made.
"""

import collections.abc
import copy
import pickle  # nosec B403
import threading
import time
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pytest

# Aliased: a bare ``import mloda.user`` would bind the name ``mloda`` to the package and collide
# with the ``mloda`` mlodaAPI alias imported below.
import mloda.user as mloda_user
from mloda.core.api.request import Engine
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet, PropertySpec
from mloda.user import Feature, Options, ParallelizationMode, PlanStep, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from tests.test_core.test_abstract_plugins.test_multi_column_auto_resolution import (
    MultiColumnProducer,
    MultiColumnTestDataCreator,
)
from tests.test_core.test_api.test_plan_info import PlanInfoPandasSource

_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoPandasSource, PandasAggregatedFeatureGroup})

# A chained request: an aggregated feature over a source feature.
_CHAINED_FEATURES: list[Feature | str] = ["plan_info_sales__mean_aggr"]

# Two feature groups resolve here, so the stream yields two elements and can be closed early.
_TWO_GROUP_FEATURES: list[Feature | str] = ["plan_info_sales", "plan_info_sales__mean_aggr"]

_PROBE_VALUES = {"A": [1, 2, 3], "B": [10, 20, 30]}


class RunResultProbe(FeatureGroup):
    """Root source whose column values depend on the ``source_table`` group option."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_result_probe_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        assert features.options is not None
        return pd.DataFrame({"run_result_probe_value": _PROBE_VALUES[features.options.get("source_table")]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_PROBE_PLUGINS = PluginCollector.enabled_feature_groups({RunResultProbe})


class RunResultOrderProbe(FeatureGroup):
    """The "slow" role blocks on a class-level gate until "fast" signals it, then adds a short
    delay so a single THREADING poll pass can't hide the fix (a class attribute, since a
    threading.Event in Options wouldn't survive Engine.compute()'s per-step deepcopy)."""

    _gate: threading.Event = threading.Event()

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_result_order_probe_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        assert features.options is not None
        role = features.options.get("role")
        if role == "slow":
            assert cls._gate.wait(timeout=5.0), "fast role never signalled the gate"
            time.sleep(0.25)
        elif role == "fast":
            cls._gate.set()
        return pd.DataFrame({"run_result_order_probe_value": [features.options.get("probe_id")]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_ORDER_PROBE_PLUGINS = PluginCollector.enabled_feature_groups({RunResultOrderProbe})


def _order_probe_feature(probe_id: str, role: str | None = None) -> Feature:
    context: dict[str, Any] = {} if role is None else {"role": role}
    return Feature("run_result_order_probe_value", options=Options(group={"probe_id": probe_id}, context=context))


def _order_probe_ids_in_plan(plan: list[PlanStep]) -> list[str]:
    """The requested probe_id of each order-probe compute step, in plan order."""
    ids: list[str] = []
    for step in plan:
        if step.step_kind != "compute" or "run_result_order_probe_value" not in step.feature_names:
            continue
        assert step.feature_set_options is not None
        ids.append(step.feature_set_options.get("probe_id"))
    return ids


def _run_probe(features: list[Feature | str]) -> Any:
    return mlodaAPI.run_all(
        features,
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_PROBE_PLUGINS,
    )


# Helpers return Any on purpose: the concrete return types (RunResult/ResultStream) are exactly
# what these tests assert at runtime.
def _run_all(features: list[Feature | str] | None = None) -> Any:
    return mlodaAPI.run_all(
        features if features is not None else _CHAINED_FEATURES,
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_PLUGINS,
    )


def _stream_all(features: list[Feature | str] | None = None) -> Any:
    return mlodaAPI.stream_all(
        features if features is not None else _CHAINED_FEATURES,
        compute_frameworks=[PandasDataFrame],
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

    @pytest.mark.parametrize("name", ["plan", "run_id", "plan_id"])
    def test_plan_and_ids_are_read_only(self, name: str) -> None:
        results = _run_all()

        with pytest.raises(AttributeError):
            setattr(results, name, [])

    def test_chained_request_plan_contains_the_aggregation_compute_step(self) -> None:
        """The plan names the concrete aggregation group and framework."""
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
            compute_frameworks=[PandasDataFrame],
            plugin_collector=_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert results.plan == session.resolved_plan()
        session_results: Any = session.run(parallelization_modes={ParallelizationMode.SYNC})
        assert session_results.plan == session.resolved_plan()


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
                compute_frameworks=[PandasDataFrame],
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

        session = mloda.prepare(
            _CHAINED_FEATURES,
            compute_frameworks=[PandasDataFrame],
            plugin_collector=_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert type(session.run(parallelization_modes={ParallelizationMode.SYNC})) is RunResult
        assert isinstance(session.stream_run(parallelization_modes={ParallelizationMode.SYNC}), ResultStream)


class TestRunResultIsPlainData:
    """RunResult is plain data: picklable, deep-copyable, and free of the producing session."""

    def test_pickle_round_trip_preserves_content(self) -> None:
        results = _run_all()

        loaded = pickle.loads(pickle.dumps(results))  # nosec B301

        assert len(loaded) == len(results) == 1
        assert loaded[0]["plan_info_sales__mean_aggr"].tolist() == results[0]["plan_info_sales__mean_aggr"].tolist()

    def test_pickle_round_trip_preserves_type_plan_and_ids(self) -> None:
        from mloda.user import RunResult

        results = _run_all()

        loaded = pickle.loads(pickle.dumps(results))  # nosec B301

        assert type(loaded) is RunResult
        assert loaded.plan == results.plan
        assert results.run_id is not None
        assert loaded.run_id == results.run_id
        assert loaded.plan_id == results.plan_id

    def test_deepcopy_preserves_content(self) -> None:
        results = _run_all()

        copied = copy.deepcopy(results)

        assert len(copied) == len(results) == 1
        assert copied[0]["plan_info_sales__mean_aggr"].tolist() == results[0]["plan_info_sales__mean_aggr"].tolist()

    def test_run_result_does_not_retain_the_session(self) -> None:
        """Memory pin: the plan is snapshotted eagerly, the session reference is dropped."""
        results = _run_all()

        assert not hasattr(results, "_session")

    def test_result_stream_does_not_retain_the_session(self) -> None:
        """Memory pin: same session-free contract for stream_all."""
        stream = _stream_all()

        assert not hasattr(stream, "_session")


class TestSubclassDispatch:
    """Subclassed entry points keep dispatching and keep the new return type."""

    def test_subclass_run_all_returns_a_run_result(self) -> None:
        from mloda.user import RunResult

        class CustomAPI(mlodaAPI):
            pass

        results = CustomAPI.run_all(
            _CHAINED_FEATURES,
            compute_frameworks=[PandasDataFrame],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_PLUGINS,
        )

        assert isinstance(results, RunResult)
        assert len(results) == 1


class TestRunResultFrames:
    """frames() pairs each frame with the compute PlanStep that produced it."""

    def test_frames_pair_each_frame_with_the_step_whose_options_produced_it(self) -> None:
        results = _run_probe([Feature("run_result_probe_value", options={"source_table": t}) for t in ("A", "B")])

        pairs = results.frames()

        assert {step.feature_set_options.get("source_table") for step, _ in pairs} == {"A", "B"}
        for step, frame in pairs:
            table = step.feature_set_options.get("source_table")
            assert frame["run_result_probe_value"].tolist() == _PROBE_VALUES[table]

    def test_frames_pair_by_step_uuid_not_by_position(self) -> None:
        from mloda.user import RunResult

        uuid_a = uuid4()
        uuid_b = uuid4()
        step_a = PlanStep("compute", ("a",), None, None, None, None, step_uuid=uuid_a)
        step_b = PlanStep("compute", ("b",), None, None, None, None, step_uuid=uuid_b)
        frame_a, frame_b = object(), object()

        results = RunResult([frame_b, frame_a], [step_a, step_b], [(uuid_b, frame_b), (uuid_a, frame_a)])

        assert results.frames() == [(step_b, frame_b), (step_a, frame_a)]

    def test_unpicklable_options_context_does_not_block_pickling(self) -> None:
        options = Options(group={"source_table": "A"}, context={"unpicklable": lambda: None})
        results = _run_probe([Feature("run_result_probe_value", options=options)])

        loaded = pickle.loads(pickle.dumps(results))  # nosec B301

        [(step, frame)] = loaded.frames()
        assert step.feature_set_options == Options(group={"source_table": "A"})
        assert frame["run_result_probe_value"].tolist() == _PROBE_VALUES["A"]


class RunResultDefaultsProbe(FeatureGroup):
    """Root source whose step options gain a declared group default."""

    PROPERTY_MAPPING = {"run_result_mode": PropertySpec("Engine-filled group default", context=False, default="fast")}

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_result_defaults_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"run_result_defaults_value": [7]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_DEFAULTS_PLUGINS = PluginCollector.enabled_feature_groups({RunResultDefaultsProbe})


def _run_defaults_probe() -> Any:
    return mlodaAPI.run_all(
        ["run_result_defaults_value"],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_DEFAULTS_PLUGINS,
    )


def _run_defaults_probe_with(features: list[Feature | str]) -> Any:
    return mlodaAPI.run_all(
        features,
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=_DEFAULTS_PLUGINS,
    )


class RunResultAnyFrameworkProbe(FeatureGroup):
    """Root source with no framework rule, so it runs on any requested framework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_result_any_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"run_result_any_value": [1, 2, 3]}


_ANY_PLUGINS = PluginCollector.enabled_feature_groups({RunResultAnyFrameworkProbe})


class TestRunResultFrameFor:
    """frame_for finds the frame whose compute step requested the feature."""

    def test_each_name_in_two_groups_returns_its_own_frame(self) -> None:
        results = _run_all(_TWO_GROUP_FEATURES)

        source = results.frame_for("plan_info_sales")
        aggregated = results.frame_for("plan_info_sales__mean_aggr")

        assert source is not aggregated
        assert "plan_info_sales" in source.columns
        assert "plan_info_sales__mean_aggr" in aggregated.columns

    def test_two_features_of_one_feature_set_return_the_same_frame(self) -> None:
        results = _run_all(["plan_info_sales", "plan_info_price"])

        assert results.frame_for("plan_info_sales") is results.frame_for("plan_info_price")

    def test_same_name_with_two_option_sets_is_ambiguous_for_a_str(self) -> None:
        results = _run_probe([Feature("run_result_probe_value", options={"source_table": t}) for t in ("A", "B")])

        with pytest.raises(ValueError, match="ambiguous") as excinfo:
            results.frame_for("run_result_probe_value")

        assert "'A'" in str(excinfo.value)
        assert "'B'" in str(excinfo.value)

    def test_feature_with_options_disambiguates(self) -> None:
        results = _run_probe([Feature("run_result_probe_value", options={"source_table": t}) for t in ("A", "B")])

        frame = results.frame_for(Feature("run_result_probe_value", options={"source_table": "B"}))

        assert frame["run_result_probe_value"].tolist() == _PROBE_VALUES["B"]

    def test_unknown_name_raises_and_lists_available_names(self) -> None:
        results = _run_probe([Feature("run_result_probe_value", options={"source_table": "A"})])

        with pytest.raises(ValueError, match="does_not_exist") as excinfo:
            results.frame_for("does_not_exist")

        assert "run_result_probe_value" in str(excinfo.value)
        assert "computed by" not in str(excinfo.value)

    def test_computed_but_unrequested_name_names_its_feature_group(self) -> None:
        """The chained source was computed, not requested: the error says who computed it."""
        results = _run_all()

        with pytest.raises(ValueError, match="computed by PlanInfoPandasSource") as excinfo:
            results.frame_for("plan_info_sales")

        assert "not requested; request it to get its frame" in str(excinfo.value)

    def test_feature_with_unmatched_group_options_raises(self) -> None:
        results = _run_probe([Feature("run_result_probe_value", options={"source_table": t}) for t in ("A", "B")])

        with pytest.raises(ValueError, match="run_result_probe_value"):
            results.frame_for(Feature("run_result_probe_value", options={"source_table": "C"}))

    def test_engine_added_group_keys_are_ignored_by_the_subset_rule(self) -> None:
        results = _run_defaults_probe()
        [(step, expected)] = results.frames()
        assert step.feature_set_options.group.get("run_result_mode") == "fast"

        assert results.frame_for(Feature("run_result_defaults_value")) is expected
        assert results.frame_for(Feature("run_result_defaults_value", options={"run_result_mode": "fast"})) is expected

    def test_group_option_not_on_the_step_does_not_match(self) -> None:
        results = _run_defaults_probe()

        with pytest.raises(ValueError, match="run_result_defaults_value"):
            results.frame_for(Feature("run_result_defaults_value", options={"run_result_absent": 1}))

    def test_options_object_form_disambiguates(self) -> None:
        results = _run_probe([Feature("run_result_probe_value", options={"source_table": t}) for t in ("A", "B")])

        frame = results.frame_for(Feature("run_result_probe_value", options=Options(group={"source_table": "B"})))

        assert frame["run_result_probe_value"].tolist() == _PROBE_VALUES["B"]

    def test_fewest_extra_group_keys_wins_and_str_stays_ambiguous(self) -> None:
        plain = Feature("run_result_defaults_value")
        extra = Feature("run_result_defaults_value", options={"run_result_extra": 1})
        results = _run_defaults_probe_with([plain, extra])
        by_extra = {"run_result_extra" in step.feature_set_options.group: frame for step, frame in results.frames()}
        assert len(by_extra) == 2

        assert results.frame_for(plain) is by_extra[False]
        assert results.frame_for(extra) is by_extra[True]
        with pytest.raises(ValueError, match="ambiguous"):
            results.frame_for("run_result_defaults_value")

    def test_compute_framework_pin_matches_only_that_framework(self) -> None:
        pandas_pin = Feature("run_result_any_value", compute_framework="PandasDataFrame")
        arrow_pin = Feature("run_result_any_value", compute_framework="PyArrowTable")
        results = mlodaAPI.run_all(
            [pandas_pin, arrow_pin],
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_ANY_PLUGINS,
        )

        assert isinstance(results.frame_for(pandas_pin), pd.DataFrame)
        assert isinstance(results.frame_for(arrow_pin), pa.Table)
        with pytest.raises(ValueError, match="ambiguous"):
            results.frame_for("run_result_any_value")

    def test_credential_group_values_are_redacted_in_errors(self) -> None:
        dsn = "postgres://user:s3cretpw@host/db"
        results = _run_probe(
            [Feature("run_result_probe_value", options={"source_table": t, "dsn": dsn}) for t in ("A", "B")]
        )

        with pytest.raises(ValueError, match="ambiguous") as ambiguous:
            results.frame_for("run_result_probe_value")
        with pytest.raises(ValueError, match="run_result_probe_value") as no_match:
            results.frame_for(Feature("run_result_probe_value", options={"source_table": "C", "dsn": dsn}))

        assert "s3cretpw" not in str(ambiguous.value)
        assert "s3cretpw" not in str(no_match.value)

    def test_multi_output_feature_is_found_by_its_base_name(self) -> None:
        results = mlodaAPI.run_all(
            ["base_feature"],
            compute_frameworks=[PandasDataFrame],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=PluginCollector.enabled_feature_groups({MultiColumnTestDataCreator, MultiColumnProducer}),
        )

        frame = results.frame_for("base_feature")

        assert "base_feature~0" in frame.columns


class TestResultStreamFrames:
    """ResultStream.frames() pairs each frame with the compute PlanStep that produced it."""

    def test_frames_pair_each_frame_with_the_step_whose_options_produced_it(self) -> None:
        stream = mlodaAPI.stream_all(
            [Feature("run_result_probe_value", options={"source_table": t}) for t in ("A", "B")],
            compute_frameworks=[PandasDataFrame],
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=_PROBE_PLUGINS,
        )

        pairs = list(stream.frames())

        tables = set()
        for step, frame in pairs:
            assert step.feature_set_options is not None
            table = step.feature_set_options.get("source_table")
            tables.add(table)
            assert frame["run_result_probe_value"].tolist() == _PROBE_VALUES[table]
        assert tables == {"A", "B"}

    def test_frames_pair_by_step_uuid_not_by_position(self) -> None:
        from mloda.user import ResultStream

        uuid_a = uuid4()
        uuid_b = uuid4()
        step_a = PlanStep("compute", ("a",), None, None, None, None, step_uuid=uuid_a)
        step_b = PlanStep("compute", ("b",), None, None, None, None, step_uuid=uuid_b)
        frame_a, frame_b = object(), object()

        stream = ResultStream((item for item in [(uuid_b, frame_b), (uuid_a, frame_a)]), [step_a, step_b])

        assert list(stream.frames()) == [(step_b, frame_b), (step_a, frame_a)]


class TestRunAllListOrderMatchesPlanOrder:
    """run_all's list and frames() follow plan order, not THREADING completion order."""

    @pytest.fixture(params=["run_all", "session_run"])
    def entry(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def _run_with_plan_first_gated(self, entry: str = "run_all") -> Any:
        session = mloda.prepare(
            [_order_probe_feature("alpha"), _order_probe_feature("beta")],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=_ORDER_PROBE_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        slow_probe_id = _order_probe_ids_in_plan(session.resolved_plan())[0]
        fast_probe_id = "beta" if slow_probe_id == "alpha" else "alpha"
        RunResultOrderProbe._gate = threading.Event()

        if entry == "session_run":
            gated = mloda.prepare(
                [
                    _order_probe_feature(slow_probe_id, role="slow"),
                    _order_probe_feature(fast_probe_id, role="fast"),
                ],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=_ORDER_PROBE_PLUGINS,
                parallelization_modes={ParallelizationMode.THREADING},
            )
            return gated.run(parallelization_modes={ParallelizationMode.THREADING})

        return mlodaAPI.run_all(
            [
                _order_probe_feature(slow_probe_id, role="slow"),
                _order_probe_feature(fast_probe_id, role="fast"),
            ],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=_ORDER_PROBE_PLUGINS,
            parallelization_modes={ParallelizationMode.THREADING},
        )

    def test_threading_list_and_frames_match_plan_order(self, entry: str) -> None:
        results = self._run_with_plan_first_gated(entry)

        expected_order = _order_probe_ids_in_plan(results.plan)
        actual_order = [frame["run_result_order_probe_value"].iloc[0] for frame in list(results)]
        frame_order = [frame["run_result_order_probe_value"].iloc[0] for _, frame in results.frames()]
        assert actual_order == expected_order
        assert frame_order == expected_order

    def test_sync_list_order_already_matches_plan_order(self) -> None:
        """Regression guard: SYNC is single-threaded, so this already passes today."""
        results = mlodaAPI.run_all(
            [_order_probe_feature("alpha"), _order_probe_feature("beta")],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=_ORDER_PROBE_PLUGINS,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        expected_order = _order_probe_ids_in_plan(results.plan)
        actual_order = [frame["run_result_order_probe_value"].iloc[0] for frame in list(results)]
        assert actual_order == expected_order
