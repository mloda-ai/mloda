"""Run-level NaN/null loss check: RunResult.nan_null_losses and the refuse_nan_null_loss option."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.nan_null_loss import NanNullLoss, NanNullLossError
from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

_NAN = float("nan")
_VALUES: dict[str, list[float | None]] = {
    "nl_both": [1.0, _NAN, None],
    "nl_nan_only": [1.0, _NAN, 2.0],
    "nl_null_only": [1.0, None, 2.0],
}


class NlArrowSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(_VALUES)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({name: pa.array(_VALUES[name], type=pa.float64()) for name in features.get_all_names()})


class _NlConsumer(FeatureGroup):
    SUFFIX = ""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(str(feature_name).removesuffix(self.SUFFIX))}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {name + cls.SUFFIX for name in _VALUES}


class NlPandasConsumer(_NlConsumer):
    SUFFIX = "__pd"

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        data[name] = data[name.removesuffix(cls.SUFFIX)]
        return data


class NlSqliteConsumer(_NlConsumer):
    SUFFIX = "__sq"

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        return data.select(_raw_sql=f"*, {name.removesuffix(cls.SUFFIX)} AS {name}")


class NlPolarsConsumer(_NlConsumer):
    SUFFIX = "__pl"

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PolarsDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        return data.with_columns(data[name.removesuffix(cls.SUFFIX)].alias(name))


_GROUPS = {NlArrowSource, NlPandasConsumer, NlSqliteConsumer, NlPolarsConsumer}


def _run(feature: str, *, mode: ParallelizationMode = ParallelizationMode.SYNC, **kwargs: Any) -> Any:
    return mloda.run_all(
        [feature],
        compute_frameworks=[PyArrowTable, PandasDataFrame, SqliteFramework, PolarsDataFrame],
        plugin_collector=PluginCollector.enabled_feature_groups(_GROUPS),
        parallelization_modes={mode},
        **kwargs,
    )


def _step_uuid(result: Any, kind: str) -> Any:
    (step,) = [s for s in result.plan if s.step_kind == kind]
    return step.step_uuid


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
class TestTransferReport:
    @pytest.mark.parametrize("suffix, target", [("__pd", "PandasDataFrame")], ids=["pandas"])
    def test_a_both_values_column_yields_one_record_paired_with_the_transform_step(
        self, mode: ParallelizationMode, suffix: str, target: str
    ) -> None:
        result = _run("nl_both" + suffix, mode=mode)

        assert result.nan_null_losses == (
            NanNullLoss(
                step_uuid=_step_uuid(result, "transform"),
                column="nl_both",
                source_framework="PyArrowTable",
                target_framework=target,
            ),
        )
        assert isinstance(result.nan_null_losses, tuple)

    @pytest.mark.parametrize("feature", ["nl_nan_only__pd", "nl_null_only__pd"])
    @pytest.mark.parametrize("refuse", [False, True])
    def test_clean_data_reports_nothing_and_is_not_refused(
        self, mode: ParallelizationMode, feature: str, refuse: bool
    ) -> None:
        assert _run(feature, mode=mode, refuse_nan_null_loss=refuse).nan_null_losses == ()

    def test_a_pair_that_does_not_fold_nan_into_null_never_reports(self, mode: ParallelizationMode) -> None:
        result = _run("nl_both__pl", mode=mode, refuse_nan_null_loss=True)

        assert result.nan_null_losses == ()


class TestRefuse:
    def test_refuse_raises_naming_the_step_and_the_column(self) -> None:
        with pytest.raises(NanNullLossError) as excinfo:
            _run("nl_both__pd", refuse_nan_null_loss=True)

        message = str(excinfo.value)
        for part in ("NlArrowSource", "NlPandasConsumer", "PyArrowTable", "PandasDataFrame", "nl_both"):
            assert part in message, message

    def test_without_refuse_the_same_run_succeeds(self) -> None:
        assert len(_run("nl_both__pd").nan_null_losses) == 1


class TestOutputFramework:
    def _run_output(self, **kwargs: Any) -> Any:
        return mloda.run_all(
            ["nl_both"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({NlArrowSource}),
            output_framework="PandasDataFrame",
            **kwargs,
        )

    def test_the_conversion_record_pairs_with_the_compute_step(self) -> None:
        result = self._run_output()

        assert result.nan_null_losses == (
            NanNullLoss(
                step_uuid=_step_uuid(result, "compute"),
                column="nl_both",
                source_framework="PyArrowTable",
                target_framework="PandasDataFrame",
            ),
        )

    def test_refuse_raises_naming_the_group_and_the_column(self) -> None:
        with pytest.raises(NanNullLossError) as excinfo:
            self._run_output(refuse_nan_null_loss=True)

        message = str(excinfo.value)
        for part in ("NlArrowSource", "PandasDataFrame", "nl_both"):
            assert part in message, message

    def test_no_output_framework_never_reports(self) -> None:
        result = mloda.run_all(
            ["nl_both"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({NlArrowSource}),
            refuse_nan_null_loss=True,
        )

        assert result.nan_null_losses == ()


class TestMultiprocessingNanNullLoss:
    # Spawning workers and moving data over the flight server exceeds the suite-wide timeout budget.
    @pytest.mark.timeout(30)
    def test_records_cross_the_worker_boundary(self, flight_server: Any) -> None:
        result = _run("nl_both__pd", mode=ParallelizationMode.MULTIPROCESSING, flight_server=flight_server)

        assert result.nan_null_losses == (
            NanNullLoss(
                step_uuid=_step_uuid(result, "transform"),
                column="nl_both",
                source_framework="PyArrowTable",
                target_framework="PandasDataFrame",
            ),
        )
