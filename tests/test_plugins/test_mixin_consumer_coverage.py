"""Every concrete mloda_plugins implementation of a covered family must have a consumer of its contract mixin.

Plugins come from PluginLoader, which skips modules missing an optional dependency. Consumers come from listed modules
imported explicitly (no collection-order dependence); one skipped for a missing dependency still counts."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from mloda._testing.compute_framework.connection_suite import ConnectionEndToEndSuite, ConnectionPickSuite
from mloda._testing.compute_framework.dtype_suite import DataTypeValidatorSuite, DtypeExtractionSuite
from mloda._testing.compute_framework.filter_suite import FilterEngineSuite
from mloda._testing.compute_framework.frame_suite import AppendColumnsSuite, EmptyResultSuite
from mloda._testing.compute_framework.mask_suite import MaskEngineSuite
from mloda._testing.compute_framework.merge_suite import AsofMergeEngineSuite, MergeEngineSuite
from mloda._testing.compute_framework.output_schema_suite import OutputSchemaSuite
from mloda._testing.compute_framework.relation_suite import SqlRelationWindowSuite
from mloda._testing.compute_framework.run_all_suite import (
    AsofRunAllSuite,
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    SiblingDictRunAllSuite,
)
from mloda._testing.compute_framework_shape import applicable_aspects, serves_asof_merge, serves_connection
from mloda._testing.source.read_db_suite import ReadDBFGTestMixin
from mloda._testing.source.read_document_suite import ReadDocumentFGTestMixin
from mloda._testing.source.read_file_suite import ReadFileFGTestMixin
from mloda._testing.source.source_suite import SourceFeatureGroupTestMixin
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.implementation import loaded_implementations
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.provider import ReadDBFG, ReadDocumentFG, ReadFileFG, SourceFeatureGroup
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_relation import SqlBaseRelation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.aggregated_calculate_test_mixin import (
    AggregatedCalculateTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.aggregated_zero_row_test_mixin import (
    AggregatedZeroRowTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_missing_value_feature_group.missing_value_zero_row_test_mixin import (
    MissingValueZeroRowTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.time_window_zero_row_test_mixin import (
    TimeWindowZeroRowTestMixin,
)

_INPUT_DATA = "tests.test_plugins.feature_group.input_data"
_EXPERIMENTAL = "tests.test_plugins.feature_group.experimental"
_FRAMEWORKS = "tests.test_plugins.compute_framework.base_implementations"
_RUN_ALL_HOST_MODULES = (
    f"{_FRAMEWORKS}.duckdb.test_duckdb_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.iceberg.test_iceberg_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.pandas.test_pandas_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.polars.test_polars_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.pyarrow.test_pyarrow_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.python_dict.test_python_dict_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.spark.test_spark_allow_empty_result_run_all",
    f"{_FRAMEWORKS}.sqlite.test_sqlite_allow_empty_result_run_all",
)


_AGGREGATED_CONSUMERS = (
    f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_aggregated_feature_group",
    f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_polars_lazy_aggregated_feature_group",
    f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_pyarrow_aggregated_feature_group",
    f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_python_dict_aggregated_feature_group",
)


@dataclass(frozen=True)
class MixinContract:
    """A plugin base, the mixin its implementations must join, and where the consumers live.

    With `aspect` set, only implementations whose `applicable_aspects` include it need a consumer, and
    `resolver` maps a consumer's `attribute` value to the implementation it covers; `applies` filters likewise."""

    base: type[Any]
    mixin: type[Any]
    attribute: str
    consumer_modules: tuple[str, ...]
    resolver: Callable[[type[Any]], type[Any]] | None = None
    aspect: str | None = None
    applies: Callable[[type[Any]], bool] | None = None


def _framework_of_adapter(adapter_class: type[Any]) -> type[Any]:
    framework: type[Any] = adapter_class().compute_framework()
    return framework


def _serves_sql_relation(cls: type[Any]) -> bool:
    if not cls.is_available():
        return False
    data_framework = cls.expected_data_framework()
    return isinstance(data_framework, type) and issubclass(data_framework, SqlBaseRelation)


CONTRACTS: tuple[MixinContract, ...] = (
    MixinContract(
        AggregatedFeatureGroup,
        AggregatedZeroRowTestMixin,
        "feature_group_class",
        _AGGREGATED_CONSUMERS,
    ),
    MixinContract(AggregatedFeatureGroup, AggregatedCalculateTestMixin, "feature_group_class", _AGGREGATED_CONSUMERS),
    MixinContract(
        TimeWindowFeatureGroup,
        TimeWindowZeroRowTestMixin,
        "feature_group_class",
        (
            f"{_EXPERIMENTAL}.test_time_window_feature_group.test_pandas_time_window_feature_group",
            f"{_EXPERIMENTAL}.test_time_window_feature_group.test_pyarrow_time_window_feature_group",
        ),
    ),
    MixinContract(
        MissingValueFeatureGroup,
        MissingValueZeroRowTestMixin,
        "feature_group_class",
        (
            f"{_EXPERIMENTAL}.test_missing_value_feature_group.test_pandas_missing_value_feature_group",
            f"{_EXPERIMENTAL}.test_missing_value_feature_group.test_pyarrow_missing_value_feature_group",
            f"{_EXPERIMENTAL}.test_missing_value_feature_group.test_python_dict_missing_value_feature_group",
        ),
    ),
    MixinContract(
        ComputeFramework,
        MaskEngineSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_mask_engine",
            f"{_FRAMEWORKS}.pandas.test_pandas_mask_engine",
            f"{_FRAMEWORKS}.polars.test_polars_expr_mask_engine",
            f"{_FRAMEWORKS}.polars.test_polars_mask_engine",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_mask_engine",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_mask_engine",
            f"{_FRAMEWORKS}.spark.test_spark_mask_engine",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_mask_engine",
        ),
        resolver=_framework_of_adapter,
        aspect="mask",
    ),
    MixinContract(
        ComputeFramework,
        FilterEngineSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_filter_engine",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_filter_engine",
            f"{_FRAMEWORKS}.pandas.test_pandas_filter_engine",
            f"{_FRAMEWORKS}.polars.test_polars_filter_engine",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_filter_engine",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_filter_engine",
            f"{_FRAMEWORKS}.spark.test_spark_filter_engine",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_filter_engine",
        ),
        resolver=_framework_of_adapter,
        aspect="filter",
    ),
    MixinContract(
        ComputeFramework,
        MergeEngineSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_merge_engine",
            f"{_FRAMEWORKS}.pandas.test_pandas_merge_engine",
            f"{_FRAMEWORKS}.polars.test_polars_lazy_merge_engine",
            f"{_FRAMEWORKS}.polars.test_polars_merge_engine",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_merge_engine",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_merge_engine",
            f"{_FRAMEWORKS}.spark.test_spark_merge_engine",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_merge_engine",
        ),
        resolver=_framework_of_adapter,
        aspect="merge",
    ),
    MixinContract(
        ComputeFramework,
        AsofMergeEngineSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_asof_merge_engine",
            f"{_FRAMEWORKS}.pandas.test_pandas_asof_merge_engine",
            f"{_FRAMEWORKS}.polars.test_polars_asof_merge_engine",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_asof_merge_engine",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_asof_merge_engine",
            f"{_FRAMEWORKS}.spark.test_spark_asof_merge_engine",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_asof_merge_engine",
        ),
        resolver=_framework_of_adapter,
        applies=serves_asof_merge,
    ),
    MixinContract(
        ComputeFramework,
        OutputSchemaSuite,
        "adapter_class",
        (
            "tests.test_plugins.compute_framework.test_output_schema_hook",
            f"{_FRAMEWORKS}.spark.test_spark_framework",
        ),
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        DtypeExtractionSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_framework",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_framework",
            f"{_FRAMEWORKS}.pandas.test_pandas_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_lazy_dataframe",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_table",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_framework",
            f"{_FRAMEWORKS}.spark.test_spark_framework",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_framework",
        ),
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        DataTypeValidatorSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_framework",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_framework",
            f"{_FRAMEWORKS}.pandas.test_pandas_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_lazy_dataframe",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_table",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_framework",
            f"{_FRAMEWORKS}.spark.test_spark_framework",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_framework",
        ),
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        EmptyResultSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_framework",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_framework",
            f"{_FRAMEWORKS}.pandas.test_pandas_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_lazy_dataframe",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_table",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_allow_empty_result",
            f"{_FRAMEWORKS}.spark.test_spark_framework",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_framework",
        ),
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        AppendColumnsSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_framework",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_framework",
            f"{_FRAMEWORKS}.pandas.test_pandas_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_dataframe",
            f"{_FRAMEWORKS}.polars.test_polars_lazy_dataframe",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_table",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_allow_empty_result",
            f"{_FRAMEWORKS}.spark.test_spark_framework",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_framework",
        ),
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        EmptyResultRunAllSuite,
        "adapter_class",
        _RUN_ALL_HOST_MODULES,
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        SiblingDictRunAllSuite,
        "adapter_class",
        _RUN_ALL_HOST_MODULES,
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        AsofRunAllSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_asof_run_all",
            f"{_FRAMEWORKS}.pandas.test_pandas_asof_run_all",
            f"{_FRAMEWORKS}.polars.test_polars_asof_run_all",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_asof_run_all",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_asof_run_all",
            f"{_FRAMEWORKS}.spark.test_spark_asof_run_all",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_asof_run_all",
        ),
        resolver=_framework_of_adapter,
        applies=serves_asof_merge,
    ),
    MixinContract(
        ComputeFramework,
        MissingColumnRunAllSuite,
        "adapter_class",
        _RUN_ALL_HOST_MODULES,
        resolver=_framework_of_adapter,
    ),
    MixinContract(
        ComputeFramework,
        ConnectionPickSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_framework",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_framework",
            f"{_FRAMEWORKS}.spark.test_spark_framework",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_framework",
        ),
        resolver=_framework_of_adapter,
        applies=serves_connection,
    ),
    MixinContract(
        ComputeFramework,
        ConnectionEndToEndSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_tfs_connection",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_tfs_connection",
            f"{_FRAMEWORKS}.spark.test_spark_tfs_connection",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_tfs_connection",
        ),
        resolver=_framework_of_adapter,
        applies=serves_connection,
    ),
    MixinContract(
        ComputeFramework,
        SqlRelationWindowSuite,
        "adapter_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_relation",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_relation",
        ),
        resolver=_framework_of_adapter,
        applies=_serves_sql_relation,
    ),
    MixinContract(
        SourceFeatureGroup,
        SourceFeatureGroupTestMixin,
        "feature_group_class",
        (
            f"{_INPUT_DATA}.test_read_dbs.test_duckdb_fg",
            f"{_INPUT_DATA}.test_read_dbs.test_sqlite_fg",
            f"{_INPUT_DATA}.test_read_files.test_implementations",
            f"{_INPUT_DATA}.test_read_files.test_json_document_reader",
            f"{_INPUT_DATA}.test_read_files.test_markdown_document_reader",
            f"{_INPUT_DATA}.test_read_files.test_text_document_reader",
            f"{_INPUT_DATA}.test_read_files.test_yaml_document_reader",
            "tests.test_core.test_abstract_plugins.test_components.test_input_data.test_toy_file_format_feature_group",
            "tests.test_core.test_abstract_plugins.test_components.test_input_data."
            "test_toy_source_feature_group_contract",
        ),
    ),
    MixinContract(
        ReadFileFG,
        ReadFileFGTestMixin,
        "feature_group_class",
        (
            f"{_INPUT_DATA}.test_read_files.test_implementations",
            "tests.test_core.test_abstract_plugins.test_components.test_input_data.test_toy_file_format_feature_group",
        ),
    ),
    MixinContract(ReadDBFG, ReadDBFGTestMixin, "feature_group_class", (f"{_INPUT_DATA}.test_read_dbs.test_sqlite_fg",)),
    MixinContract(
        ReadDocumentFG,
        ReadDocumentFGTestMixin,
        "feature_group_class",
        (
            f"{_INPUT_DATA}.test_read_files.test_json_document_reader",
            f"{_INPUT_DATA}.test_read_files.test_markdown_document_reader",
            f"{_INPUT_DATA}.test_read_files.test_text_document_reader",
            f"{_INPUT_DATA}.test_read_files.test_yaml_document_reader",
        ),
    ),
)


def _qualified_name(cls: type[Any]) -> str:
    """Compared by name, not identity: a reloaded module leaves a stale duplicate class behind."""
    return f"{cls.__module__}.{cls.__qualname__}"


def concrete_plugin_classes(
    base: type[Any], aspect: str | None = None, applies: Callable[[type[Any]], bool] | None = None
) -> set[str]:
    """Concrete subclasses of base defined under mloda_plugins; test-local subclasses fall out by module."""
    PluginLoader.all()
    if applies is not None:
        return {
            _qualified_name(cls)
            for cls in get_all_subclasses(base)
            if cls.__module__.startswith("mloda_plugins.") and not inspect.isabstract(cls) and applies(cls)
        }
    if aspect is not None:
        return {
            _qualified_name(cls)
            for cls in get_all_subclasses(base)
            if cls.__module__.startswith("mloda_plugins.")
            and not inspect.isabstract(cls)
            and aspect in applicable_aspects(cls)
        }
    if issubclass(base, OperationFeatureGroup):
        return {
            _qualified_name(impl)
            for impl in loaded_implementations()
            if issubclass(impl.feature_group, base) and impl.__module__.startswith("mloda_plugins.")
        }
    return {
        _qualified_name(cls)
        for cls in get_all_subclasses(base)
        if cls.__module__.startswith("mloda_plugins.") and not inspect.isabstract(cls)
    }


def covered_classes(contract: MixinContract) -> set[str]:
    """Classes named by the contract attribute of the mixin consumers defined in the listed modules."""
    modules = {importlib.import_module(name).__name__ for name in contract.consumer_modules}
    resolve = contract.resolver or (lambda value: value)
    return {
        _qualified_name(resolve(getattr(consumer, contract.attribute)))
        for consumer in get_all_subclasses(contract.mixin)
        if consumer.__module__ in modules and hasattr(consumer, contract.attribute)
    }


@pytest.mark.parametrize("contract", CONTRACTS, ids=[contract.mixin.__name__ for contract in CONTRACTS])
def test_every_concrete_implementation_has_a_mixin_consumer(contract: MixinContract) -> None:
    implementations = concrete_plugin_classes(contract.base, contract.aspect, contract.applies)
    assert implementations, f"no concrete {contract.base.__name__} found under mloda_plugins: the plugin import broke"
    uncovered = sorted(implementations - covered_classes(contract))
    assert uncovered == [], (
        f"{contract.base.__name__} implementations without a {contract.mixin.__name__} consumer: {uncovered}. "
        f"Add a test class inheriting {contract.mixin.__name__} with {contract.attribute} set to the class, "
        f"in a module listed in its MixinContract in {__name__}."
    )
