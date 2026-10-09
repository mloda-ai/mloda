"""Dictionary projections preserve only columns still needed by direct siblings."""

import pickle  # nosec B403
from typing import Any, ClassVar
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.cfw_manager import CfwManager
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


def columns(data: Any) -> dict[str, list[Any]]:
    if isinstance(data, pa.Table):
        return dict(data.to_pydict())
    if isinstance(data, pd.DataFrame):
        return dict(data.to_dict("list"))
    return dict(data)


class DirectSiblingRoot(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sibling_source", "sibling_other"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sibling_source": [1, 2, 3], "sibling_other": [4, 5, 6]}


class DirectSiblingFirst(FeatureGroup):
    result_mode: ClassVar[str] = "partial"
    input_name: ClassVar[str] = "sibling_source"
    calls: ClassVar[list[str]] = []

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(self.input_name)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls.append(cls.get_class_name())
        previous = columns(data)
        output = {cls.get_class_name(): [value * 2 for value in previous[cls.input_name]]}
        if cls.result_mode == "full":
            return {**previous, **output}
        if cls.result_mode == "inplace":
            data.update(output)
            return data
        if cls.result_mode == "delete":
            data.clear()
            data.update(output)
            return data
        if cls.result_mode == "arrow":
            return pa.table(output)
        return output


class DirectSiblingSecond(DirectSiblingFirst):
    pass


class ProcessFullFirst(DirectSiblingFirst):
    result_mode = "full"


class ProcessFullSecond(ProcessFullFirst):
    pass


class ProcessOtherReader(DirectSiblingFirst):
    result_mode = "full"
    input_name = "sibling_other"


class ProcessLastProjection(DirectSiblingFirst):
    pass


FRAMEWORKS = [PyArrowTable, PandasDataFrame, PythonDictFramework]
MODES = [ParallelizationMode.SYNC, ParallelizationMode.THREADING]
NAMES = ["DirectSiblingFirst", "DirectSiblingSecond"]


@pytest.fixture(autouse=True)
def reset_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DirectSiblingFirst, "calls", [])


def run(framework: type[ComputeFramework], mode: ParallelizationMode, names: list[str]) -> dict[str, list[Any]]:
    results = mloda.run_all(
        names,
        compute_frameworks=[framework],
        parallelization_modes={mode},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {DirectSiblingRoot, DirectSiblingFirst, DirectSiblingSecond}
        ),
    )
    return {name: values for result in results for name, values in columns(result).items()}


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("names", [NAMES, list(reversed(NAMES))])
def test_pending_direct_sibling_rejects_at_producer(
    framework: type[ComputeFramework], mode: ParallelizationMode, names: list[str]
) -> None:
    with pytest.raises(ValueError) as exc:
        run(framework, mode, names)
    assert len(DirectSiblingFirst.calls) == 1
    message = str(exc.value)
    assert DirectSiblingFirst.calls[0] in message
    assert "sibling_source" in message
    assert "dict" in message.lower()
    assert "preserve" in message.lower() or "append" in message.lower()
    assert "missing Links" not in message


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("mode", MODES)
def test_single_consumer_can_project(framework: type[ComputeFramework], mode: ParallelizationMode) -> None:
    assert run(framework, mode, NAMES[:1]) == {"DirectSiblingFirst": [2, 4, 6]}


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("mode", MODES)
def test_full_dict_preserves_siblings(
    framework: type[ComputeFramework], mode: ParallelizationMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DirectSiblingFirst, "result_mode", "full")
    assert run(framework, mode, NAMES) == {"DirectSiblingFirst": [2, 4, 6], "DirectSiblingSecond": [2, 4, 6]}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("result_mode", ["inplace", "delete"])
def test_python_dict_inplace_outputs(
    mode: ParallelizationMode, result_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DirectSiblingFirst, "result_mode", result_mode)
    if result_mode == "delete":
        with pytest.raises(ValueError):
            run(PythonDictFramework, mode, NAMES)
        assert len(DirectSiblingFirst.calls) == 1
    else:
        assert run(PythonDictFramework, mode, NAMES) == {
            "DirectSiblingFirst": [2, 4, 6],
            "DirectSiblingSecond": [2, 4, 6],
        }


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reader_first", [False, True])
def test_different_column_obligation_ends_when_reader_finishes(
    framework: type[ComputeFramework],
    mode: ParallelizationMode,
    reader_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DirectSiblingSecond, "input_name", "sibling_other")
    monkeypatch.setattr(DirectSiblingSecond, "result_mode", "full")
    first = "DirectSiblingSecond" if reader_first else "DirectSiblingFirst"
    later = "DirectSiblingFirst" if reader_first else "DirectSiblingSecond"

    def defer(self: Any, step: Any, made_progress: bool) -> bool:
        group = getattr(step, "feature_group", None)
        return group is not None and group.get_class_name() == later and first not in DirectSiblingFirst.calls

    monkeypatch.setattr(ExecutionOrchestrator, "_defer_ready_step", defer)
    if reader_first:
        assert run(framework, mode, NAMES) == {"DirectSiblingFirst": [2, 4, 6], "DirectSiblingSecond": [8, 10, 12]}
    else:
        with pytest.raises(ValueError) as exc:
            run(framework, mode, NAMES)
        assert DirectSiblingFirst.calls == ["DirectSiblingFirst"]
        assert "sibling_other" in str(exc.value)


def test_native_arrow_projection_remains_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DirectSiblingFirst, "result_mode", "arrow")
    assert run(PyArrowTable, ParallelizationMode.SYNC, NAMES[:1]) == {"DirectSiblingFirst": [2, 4, 6]}


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_root_with_supplied_data_remains_valid(framework: type[ComputeFramework]) -> None:
    instance = framework(mode=ParallelizationMode.SYNC)
    supplied = instance.transform({"previous_input": [9]}, ["previous_input"])
    instance.run_calculation(DirectSiblingRoot, FeatureSet([Feature("sibling_source")]), None, data=supplied)
    assert columns(instance.data) == {"sibling_source": [1, 2, 3], "sibling_other": [4, 5, 6]}


@pytest.mark.parametrize("mode", MODES)
def test_pending_reader_on_converted_frame_does_not_protect_source(
    mode: ParallelizationMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DirectSiblingRoot, "compute_framework_rule", classmethod(lambda cls: {PandasDataFrame}))
    monkeypatch.setattr(DirectSiblingFirst, "compute_framework_rule", classmethod(lambda cls: {PandasDataFrame}))
    monkeypatch.setattr(DirectSiblingSecond, "compute_framework_rule", classmethod(lambda cls: {PyArrowTable}))
    converted = []
    original_execute = TransformFrameworkStep.execute

    def execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(self, *args, **kwargs)
        converted.append(self.uuid)
        return result

    def defer(self: Any, step: Any, made_progress: bool) -> bool:
        group = getattr(step, "feature_group", None)
        if group is DirectSiblingFirst:
            return not converted
        if group is DirectSiblingSecond:
            return "DirectSiblingFirst" not in DirectSiblingFirst.calls
        return False

    monkeypatch.setattr(TransformFrameworkStep, "execute", execute)
    monkeypatch.setattr(ExecutionOrchestrator, "_defer_ready_step", defer)
    results = mloda.run_all(
        NAMES,
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        parallelization_modes={mode},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {DirectSiblingRoot, DirectSiblingFirst, DirectSiblingSecond}
        ),
    )
    assert converted
    assert DirectSiblingFirst.calls == NAMES
    assert {name: values for result in results for name, values in columns(result).items()} == {
        "DirectSiblingFirst": [2, 4, 6],
        "DirectSiblingSecond": [2, 4, 6],
    }


def test_worker_completion_survives_serialization_before_parent_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DirectSiblingFirst, "result_mode", "full")
    monkeypatch.setattr(DirectSiblingSecond, "result_mode", "partial")
    framework = PythonDictFramework(mode=ParallelizationMode.SYNC)
    framework.set_data({"sibling_source": [1, 2, 3]})
    framework.set_column_names()
    first = FeatureGroupStep(DirectSiblingFirst, FeatureSet([Feature(NAMES[0])]), {uuid4()}, PythonDictFramework)
    second = FeatureGroupStep(
        DirectSiblingSecond, FeatureSet([Feature(NAMES[1])]), first.required_uuids, PythonDictFramework
    )
    second.direct_sibling_readers = ((first.uuid, next(iter(first.get_uuids())), frozenset({"sibling_source"})),)
    first.direct_sibling_readers = ((second.uuid, next(iter(second.get_uuids())), frozenset({"sibling_source"})),)
    registry = Mock(spec=CfwManager)
    registry.get_location.return_value = None
    registry.get_runtime_artifacts.return_value = None
    registry.resolve_cfw_uuid_by_tfs_ids.return_value = framework.uuid
    first.execute(registry, framework)
    assert not first.step_is_done
    # Only deserialize the fixture objects constructed in this test.
    restored_framework, restored_second = pickle.loads(pickle.dumps((framework, second)))  # nosec B301
    restored_second.execute(registry, restored_framework)
    assert not first.step_is_done
    assert not restored_second.step_is_done
    assert restored_framework.data == {NAMES[1]: [2, 4, 6]}
    assert second.uuid not in framework.completed_feature_group_steps
    assert {first.uuid, second.uuid} <= restored_framework.completed_feature_group_steps


def run_process(
    framework: type[ComputeFramework],
    names: list[str],
    groups: set[type[FeatureGroup]],
    flight_server: ParallelRunnerFlightServer,
) -> dict[str, list[Any]]:
    results = mloda.run_all(
        names,
        compute_frameworks=[framework],
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        plugin_collector=PluginCollector.enabled_feature_groups({DirectSiblingRoot, *groups}),
        flight_server=flight_server,
    )
    return {name: values for result in results for name, values in columns(result).items()}


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_process_single_consumer_can_project(
    framework: type[ComputeFramework],
    flight_server: ParallelRunnerFlightServer,
) -> None:
    assert run_process(framework, NAMES[:1], {DirectSiblingFirst}, flight_server) == {
        "DirectSiblingFirst": [2, 4, 6],
    }


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("names", [NAMES, list(reversed(NAMES))])
def test_process_pending_sibling_rejects_at_producer(
    framework: type[ComputeFramework],
    names: list[str],
    flight_server: ParallelRunnerFlightServer,
) -> None:
    with pytest.raises(ValueError) as exc:
        run_process(framework, names, {DirectSiblingFirst, DirectSiblingSecond}, flight_server)
    message = str(exc.value)
    assert any(name in message for name in NAMES)
    assert "sibling_source" in message
    assert "returned a dict" in message
    assert "pending direct siblings" in message
    assert "Preserve" in message
    assert "missing Links" not in message


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_process_full_dict_preserves_siblings(
    framework: type[ComputeFramework],
    flight_server: ParallelRunnerFlightServer,
) -> None:
    names = ["ProcessFullFirst", "ProcessFullSecond"]
    assert run_process(framework, names, {ProcessFullFirst, ProcessFullSecond}, flight_server) == {
        "ProcessFullFirst": [2, 4, 6],
        "ProcessFullSecond": [2, 4, 6],
    }


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_process_completed_reader_allows_projection(
    framework: type[ComputeFramework],
    flight_server: ParallelRunnerFlightServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def defer(self: ExecutionOrchestrator, step: Any, made_progress: bool) -> bool:
        if getattr(step, "feature_group", None) is not ProcessLastProjection:
            return False
        return any(
            isinstance(candidate, FeatureGroupStep)
            and candidate.feature_group is ProcessOtherReader
            and not candidate.step_is_done
            for candidate in self.execution_planner
        )

    monkeypatch.setattr(ExecutionOrchestrator, "_defer_ready_step", defer)
    names = ["ProcessLastProjection", "ProcessOtherReader"]
    assert run_process(framework, names, {ProcessLastProjection, ProcessOtherReader}, flight_server) == {
        "ProcessLastProjection": [2, 4, 6],
        "ProcessOtherReader": [8, 10, 12],
    }
