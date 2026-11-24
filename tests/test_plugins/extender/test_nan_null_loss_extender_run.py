"""NanNullLossExtender end to end: transfers onto Pandas and the output_framework conversion."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.function_extender.base_implementations.nan_null_loss.nan_null_loss_extender import (
    NanNullLoss,
    NanNullLossError,
    NanNullLossExtender,
)

_NAN = float("nan")
_BOTH = [1.0, _NAN, None]


class NlArrowSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"nl_both"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"nl_both": pa.array(_BOTH, type=pa.float64())})


class NlDictSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"nl_dict_both"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"nl_dict_both": list(_BOTH)}


class NlPandasConsumer(FeatureGroup):
    SUFFIX = "__pd"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(str(feature_name).removesuffix(self.SUFFIX))}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"nl_both" + cls.SUFFIX, "nl_dict_both" + cls.SUFFIX}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        data[name] = data[name.removesuffix(cls.SUFFIX)]
        return data


class NlCollectingExtender(NanNullLossExtender):
    """Collects the records of the process that ran the hop."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[NanNullLoss] = []

    def report(self, loss: NanNullLoss) -> None:
        self.records.append(loss)


_GROUPS = {NlArrowSource, NlDictSource, NlPandasConsumer}
_SOURCES = {
    "pyarrow": ("nl_both__pd", "nl_both", "PyArrowTable"),
    "python_dict": ("nl_dict_both__pd", "nl_dict_both", "PythonDictFramework"),
}
_THREAD_MODES = [ParallelizationMode.SYNC, ParallelizationMode.THREADING]


def _prepare(feature: str, extender: Any, mode: ParallelizationMode, **kwargs: Any) -> Any:
    return mloda.prepare(
        [feature],
        compute_frameworks=[PyArrowTable, PandasDataFrame, PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups(_GROUPS),
        parallelization_modes={mode},
        function_extender={extender},
        **kwargs,
    )


def _step_uuid(plan: Any, kind: str) -> Any:
    (step,) = [s for s in plan if s.step_kind == kind]
    return step.step_uuid


@pytest.mark.parametrize("source", sorted(_SOURCES))
class TestTransferObserving:
    @pytest.mark.parametrize("mode", _THREAD_MODES)
    def test_a_both_values_column_yields_one_record_paired_with_the_transform_step(
        self, source: str, mode: ParallelizationMode
    ) -> None:
        feature, column, source_framework = _SOURCES[source]
        extender = NlCollectingExtender()

        result = _prepare(feature, extender, mode).run(parallelization_modes={mode})

        assert extender.records == [
            NanNullLoss(
                step_uuid=_step_uuid(result.plan, "transform"),
                column=column,
                source_framework=source_framework,
                target_framework="PandasDataFrame",
            )
        ]


@pytest.mark.parametrize("source", sorted(_SOURCES))
class TestTransferRefusing:
    @pytest.mark.parametrize("mode", _THREAD_MODES)
    def test_refuse_raises_naming_the_step_and_the_column(self, source: str, mode: ParallelizationMode) -> None:
        feature, column, _ = _SOURCES[source]
        session = _prepare(feature, NanNullLossExtender(refuse=True), mode)

        with pytest.raises(NanNullLossError) as excinfo:
            session.run(parallelization_modes={mode})

        message = str(excinfo.value)
        for part in (str(_step_uuid(session.resolved_plan(), "transform")), column, "PandasDataFrame"):
            assert part in message, message


class TestMultiprocessingRefusing:
    # Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
    @pytest.mark.timeout(30)
    def test_refuse_fails_a_multiprocessing_run_naming_the_column(self, flight_server: Any) -> None:
        mode = ParallelizationMode.MULTIPROCESSING
        session = _prepare("nl_both__pd", NanNullLossExtender(refuse=True), mode)

        with pytest.raises(Exception, match="hold both NaN and null") as excinfo:
            session.run(parallelization_modes={mode}, flight_server=flight_server)

        assert "nl_both" in str(excinfo.value)


class TestOutputFramework:
    @staticmethod
    def _prepare_output(extender: Any, **kwargs: Any) -> Any:
        return mloda.prepare(
            ["nl_both"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({NlArrowSource}),
            function_extender={extender},
            **kwargs,
        )

    def test_the_conversion_record_pairs_with_the_compute_step(self) -> None:
        extender = NlCollectingExtender()

        result = self._prepare_output(extender, output_framework="PandasDataFrame").run()

        assert isinstance(result[0], pd.DataFrame)
        assert extender.records == [
            NanNullLoss(
                step_uuid=_step_uuid(result.plan, "compute"),
                column="nl_both",
                source_framework="PyArrowTable",
                target_framework="PandasDataFrame",
            )
        ]

    def test_refuse_raises_naming_the_column(self) -> None:
        session = self._prepare_output(NanNullLossExtender(refuse=True), output_framework="PandasDataFrame")

        with pytest.raises(NanNullLossError) as excinfo:
            session.run()

        message = str(excinfo.value)
        for part in (str(_step_uuid(session.resolved_plan(), "compute")), "PandasDataFrame", "nl_both"):
            assert part in message, message

    def test_no_output_framework_never_reports(self) -> None:
        extender = NlCollectingExtender()

        self._prepare_output(extender).run()

        assert extender.records == []
