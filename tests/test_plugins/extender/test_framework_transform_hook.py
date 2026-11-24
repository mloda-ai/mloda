"""ExtenderHook.FRAMEWORK_TRANSFORM: fires once per conversion hop of a framework transfer and of the
output_framework conversion, with the hop as args and the framework pair on the HookContext.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.framework_transformer.transform_hook import transform_hook_invoker
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, GateBypassError
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_DENIED = "ft_hook_denied_transform"


class FtDictSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ft_dict_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ft_dict_value": [1, 2, 3]}


class FtArrowSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ft_arrow_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"ft_arrow_value": [1, 2, 3]})


class FtPandasConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ft_dict_value")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data["ft_dict_value"] * 2
        return data


_ENABLED = PluginCollector.enabled_feature_groups({FtDictSource, FtArrowSource, FtPandasConsumer})
_FRAMEWORKS = [PythonDictFramework, PyArrowTable, PandasDataFrame]


class _CapturingExtender(Extender):
    """Records args and the HookContext of every FRAMEWORK_TRANSFORM call, after the hop ran."""

    def __init__(self) -> None:
        self.priority = 100
        self.captured: list[tuple[tuple[Any, ...], HookContext]] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FRAMEWORK_TRANSFORM}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append((args, context))
        return result


class _RaisingExtender(Extender):
    def __init__(self) -> None:
        self.priority = 100

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FRAMEWORK_TRANSFORM}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(_DENIED)


def _prepare(
    feature: str, extender: Extender, mode: ParallelizationMode = ParallelizationMode.SYNC, **kwargs: Any
) -> Any:
    return mloda.prepare(
        [Feature(feature)],
        compute_frameworks=_FRAMEWORKS,
        plugin_collector=_ENABLED,
        parallelization_modes={mode},
        function_extender={extender},
        **kwargs,
    )


def _step_uuid(result: Any, kind: str) -> Any:
    (step,) = [s for s in result.plan if s.step_kind == kind]
    return step.step_uuid


def _assert_identity(context: HookContext, session: Any, carrier: dict[str, str]) -> None:
    assert context.hook == ExtenderHook.FRAMEWORK_TRANSFORM
    assert context.feature_group_class is None
    assert context.feature_group_version is None
    assert context.run_id is not None
    assert context.plan_id == session.plan_id
    assert context.run_id != session.plan_id
    assert context.structure_hash == session.plan_context.structure_hash
    assert context.carrier == carrier
    assert context.tenant_id == "acme"
    assert context.project_id == "proj1"
    assert context.principal == "hash123"
    assert context.duration_seconds is not None
    assert context.status == "success"
    assert context.rows_out is None


class _BypassingExtender(Extender):
    """Returns without calling the wrapped function."""

    def __init__(self, raise_on_error: bool) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FRAMEWORK_TRANSFORM}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return "bypassed"


class TestInvokerWithoutExtender:
    def test_returns_none_when_there_is_no_extender(self) -> None:
        assert transform_hook_invoker(None, RunContext(), None, PyArrowTable, PandasDataFrame) is None

    def test_a_breaking_extender_that_skips_the_hop_raises_gate_bypass(self) -> None:
        calls: list[Any] = []
        invoke = transform_hook_invoker(_BypassingExtender(True), RunContext(), None, PyArrowTable, PandasDataFrame)
        assert invoke is not None

        with pytest.raises(GateBypassError):
            invoke(lambda *args: calls.append(args), 1)

        assert calls == []

    def test_a_warning_only_extender_that_skips_the_hop_falls_back_to_it_once(self) -> None:
        calls: list[Any] = []

        def func(*args: Any) -> str:
            calls.append(args)
            return "converted"

        invoke = transform_hook_invoker(_BypassingExtender(False), RunContext(), None, PyArrowTable, PandasDataFrame)
        assert invoke is not None

        assert invoke(func, 1) == "converted"
        assert len(calls) == 1

    def test_the_worker_index_reaches_the_hook_context(self) -> None:
        extender = _CapturingExtender()
        invoke = transform_hook_invoker(extender, RunContext(), None, PyArrowTable, PandasDataFrame, worker_index=3)
        assert invoke is not None

        invoke(lambda *args: None, 1)

        assert extender.captured[0][1].worker_index == 3


class TestTransferHop:
    def test_a_two_hop_transfer_fires_twice_with_the_hop_and_the_framework_pair(self) -> None:
        extender = _CapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        session = _prepare("FtPandasConsumer", extender)

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            result = session.run(parallelization_modes={ParallelizationMode.SYNC}, carrier=carrier)

        assert list(result[0]["FtPandasConsumer"]) == [2, 4, 6]
        assert len(extender.captured) == 2
        (first_args, first), (second_args, second) = extender.captured
        assert first_args[:2] == (dict, pa.Table)
        assert first_args[2] == {"ft_dict_value": [1, 2, 3]}
        assert second_args[:2] == (pa.Table, pd.DataFrame)
        assert isinstance(second_args[2], pa.Table)
        step_uuid = _step_uuid(result, "transform")
        for context in (first, second):
            _assert_identity(context, session, carrier)
            assert context.step_uuid == step_uuid
            assert context.source_compute_framework is PythonDictFramework
            assert context.target_compute_framework is PandasDataFrame
            assert context.compute_framework_name == "PandasDataFrame"

    @pytest.mark.timeout(30)
    def test_a_raising_extender_fails_a_multiprocessing_run_in_the_worker(self, flight_server: Any) -> None:
        session = _prepare("FtPandasConsumer", _RaisingExtender(), ParallelizationMode.MULTIPROCESSING)

        with pytest.raises(Exception, match=_DENIED):
            session.run(parallelization_modes={ParallelizationMode.MULTIPROCESSING}, flight_server=flight_server)


class TestOutputFrameworkConversion:
    def test_fires_once_with_the_compute_step_and_the_run_and_output_frameworks(self) -> None:
        extender = _CapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        session = _prepare("ft_arrow_value", extender, output_framework="PandasDataFrame")

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            result = session.run(parallelization_modes={ParallelizationMode.SYNC}, carrier=carrier)

        assert isinstance(result[0], pd.DataFrame)
        assert len(extender.captured) == 1
        args, context = extender.captured[0]
        assert args[:2] == (pa.Table, pd.DataFrame)
        assert isinstance(args[2], pa.Table)
        _assert_identity(context, session, carrier)
        assert context.step_uuid == _step_uuid(result, "compute")
        assert context.source_compute_framework is PyArrowTable
        assert context.target_compute_framework is PandasDataFrame
        assert context.compute_framework_name == "PandasDataFrame"
        assert context.worker_index is None

    @pytest.mark.parametrize("output_framework", [None, "PyArrowTable"])
    def test_without_a_conversion_nothing_fires(self, output_framework: str | None) -> None:
        extender = _CapturingExtender()

        _prepare("ft_arrow_value", extender, output_framework=output_framework).run(
            parallelization_modes={ParallelizationMode.SYNC}
        )

        assert extender.captured == []

    def test_a_raising_extender_fails_a_sync_run(self) -> None:
        session = _prepare("ft_arrow_value", _RaisingExtender(), output_framework="PandasDataFrame")

        with pytest.raises(Exception, match=_DENIED):
            session.run(parallelization_modes={ParallelizationMode.SYNC})

    @pytest.mark.timeout(30)
    def test_a_raising_extender_fails_a_multiprocessing_run_in_the_parent(self, flight_server: Any) -> None:
        session = _prepare(
            "ft_arrow_value",
            _RaisingExtender(),
            ParallelizationMode.MULTIPROCESSING,
            output_framework="PandasDataFrame",
        )

        with pytest.raises(Exception, match=_DENIED):
            session.run(parallelization_modes={ParallelizationMode.MULTIPROCESSING}, flight_server=flight_server)
