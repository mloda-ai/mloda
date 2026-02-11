"""Tests for column ordering in identify_naming_convention() and mlodaAPI."""

from typing import Any, Optional, Set
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.core.engine import Engine
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.data_lifecycle_manager import DataLifecycleManager
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, DataCreator, BaseInputData
from mloda.user import FeatureName, ParallelizationMode, mloda, PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

# Optional imports for plugin-specific tests
try:
    import duckdb
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DuckDBFramework = None  # type: ignore[assignment, misc]

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
    from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
except ImportError:
    pl = None  # type: ignore[assignment]
    PolarsDataFrame = None  # type: ignore[assignment, misc]
    PolarsLazyDataFrame = None  # type: ignore[assignment, misc]


# --- Test Helpers ---


class ConcreteComputeFramework(ComputeFramework):
    """Concrete implementation of ComputeFramework for testing."""

    @classmethod
    def expected_data_framework(cls) -> type:
        return dict

    @classmethod
    def set_column(cls, data: Any, column_name: str, column_data: Any) -> Any:
        data[column_name] = column_data
        return data

    @classmethod
    def select_data(cls, data: Any, column_names: Set[str]) -> Any:
        return {k: v for k, v in data.items() if k in column_names}

    @classmethod
    def merge(cls, data1: Any, data2: Any) -> Any:
        return {**data1, **data2}


def create_compute_framework() -> ConcreteComputeFramework:
    return ConcreteComputeFramework(
        mode=ParallelizationMode.SYNC,
        children_if_root=frozenset(),
        uuid=uuid4(),
    )


class SimpleTestFeature(FeatureGroup):
    """Simple test feature for API tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"SimpleTestFeature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"SimpleTestFeature": [1, 2, 3]}

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"SimpleTestFeature"}


# --- Core identify_naming_convention Tests ---


class TestIdentifyNamingConvention:
    """Test identify_naming_convention() ordering functionality."""

    def test_default_returns_set(self) -> None:
        cfw = create_compute_framework()
        result = cfw.identify_naming_convention({FeatureName("A"), FeatureName("B")}, {"A", "B", "C"})
        assert isinstance(result, set)
        assert result == {"A", "B"}

    def test_alphabetical_returns_sorted_list(self) -> None:
        cfw = create_compute_framework()
        result = cfw.identify_naming_convention(
            {FeatureName("Zebra"), FeatureName("Apple"), FeatureName("Mango")},
            {"Zebra", "Apple", "Mango"},
            ordering="alphabetical",
        )
        assert result == ["Apple", "Mango", "Zebra"]

    def test_alphabetical_with_sub_columns(self) -> None:
        cfw = create_compute_framework()
        result = cfw.identify_naming_convention(
            {FeatureName("Temp")}, {"Temp~min", "Temp~max", "Temp~mean"}, ordering="alphabetical"
        )
        assert result == ["Temp~max", "Temp~mean", "Temp~min"]

    def test_request_order_returns_list(self) -> None:
        cfw = create_compute_framework()
        result = cfw.identify_naming_convention(
            {FeatureName("A"), FeatureName("B")}, {"A", "B"}, ordering="request_order"
        )
        assert isinstance(result, list)
        assert set(result) == {"A", "B"}

    def test_sub_columns_grouped_alphabetical(self) -> None:
        cfw = create_compute_framework()
        result = cfw.identify_naming_convention(
            {FeatureName("Temp"), FeatureName("Hum")},
            {"Temp~mean", "Temp~max", "Hum~mean", "Hum~max"},
            ordering="alphabetical",
        )
        assert result == ["Hum~max", "Hum~mean", "Temp~max", "Temp~mean"]

    def test_invalid_ordering_raises(self) -> None:
        cfw = create_compute_framework()
        with pytest.raises(ValueError):
            cfw.identify_naming_convention({FeatureName("A")}, {"A"}, ordering="invalid")


# --- API Parameter Tests ---


class TestApiColumnOrderingParameter:
    """Test mlodaAPI accepts column_ordering parameter."""

    def test_run_all_accepts_alphabetical(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({SimpleTestFeature})
        result = mloda.run_all(
            [Feature(name="SimpleTestFeature")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
            column_ordering="alphabetical",
        )
        assert result is not None

    def test_run_all_accepts_request_order(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({SimpleTestFeature})
        result = mloda.run_all(
            [Feature(name="SimpleTestFeature")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
            column_ordering="request_order",
        )
        assert result is not None

    def test_run_all_rejects_invalid(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({SimpleTestFeature})
        with pytest.raises(ValueError):
            mloda.run_all(
                [Feature(name="SimpleTestFeature")],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
                column_ordering="invalid",
            )

    def test_init_accepts_column_ordering(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({SimpleTestFeature})
        api = mloda(
            [Feature(name="SimpleTestFeature")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
            column_ordering="alphabetical",
        )
        assert api.column_ordering == "alphabetical"


# --- Threading Tests ---


class TestColumnOrderingThreading:
    """Test column_ordering is threaded through the system."""

    def test_data_lifecycle_manager_accepts_column_ordering(self) -> None:
        dlm = DataLifecycleManager(column_ordering="alphabetical")
        assert dlm.column_ordering == "alphabetical"

    def test_data_lifecycle_manager_defaults_to_none(self) -> None:
        dlm = DataLifecycleManager()
        assert dlm.column_ordering is None

    def test_get_result_data_passes_column_ordering(self) -> None:
        dlm = DataLifecycleManager(column_ordering="alphabetical")
        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.data = {"col": [1, 2, 3]}
        mock_cfw.select_data_by_column_names.return_value = {"col": [1, 2, 3]}

        dlm.get_result_data(mock_cfw, {FeatureName("col")})
        mock_cfw.select_data_by_column_names.assert_called_once_with(
            mock_cfw.data, {FeatureName("col")}, column_ordering="alphabetical"
        )

    def test_compute_framework_select_accepts_column_ordering(self) -> None:
        cfw = create_compute_framework()
        result = cfw.select_data_by_column_names({}, set(), column_ordering="alphabetical")
        assert result == {}


# --- Full Chain Tests ---


class TestColumnOrderingFullChain:
    """Test column_ordering flows through the complete chain."""

    def test_engine_accepts_column_ordering(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({SimpleTestFeature})
        engine = Engine(
            Features([Feature(name="SimpleTestFeature")]),
            {PandasDataFrame},
            None,
            plugin_collector=plugin_collector,
            column_ordering="alphabetical",
        )
        assert engine.column_ordering == "alphabetical"

    def test_execution_orchestrator_accepts_column_ordering(self) -> None:
        mock_plan = Mock(spec=ExecutionPlan)
        mock_plan.execution_plan = []
        orchestrator = ExecutionOrchestrator(mock_plan, column_ordering="alphabetical")
        assert orchestrator.data_lifecycle_manager.column_ordering == "alphabetical"

    def test_full_chain_flows_to_data_lifecycle_manager(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({SimpleTestFeature})
        api = mloda(
            [Feature(name="SimpleTestFeature")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
            column_ordering="alphabetical",
        )
        assert api.engine is not None
        assert api.engine.column_ordering == "alphabetical"


# --- Compute Framework Tests ---


class TestComputeFrameworksColumnOrdering:
    """Test each compute framework accepts column_ordering."""

    def test_pandas_accepts_column_ordering(self) -> None:
        cfw = PandasDataFrame(ParallelizationMode.SYNC, frozenset(), uuid4())
        data = pd.DataFrame({"Z": [1], "A": [2], "M": [3]})
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="alphabetical")
        assert list(result.columns) == ["A", "M", "Z"]

    def test_pandas_request_order(self) -> None:
        cfw = PandasDataFrame(ParallelizationMode.SYNC, frozenset(), uuid4())
        data = pd.DataFrame({"Z": [1], "A": [2], "M": [3]})
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="request_order")
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Z", "A", "M"}

    def test_pyarrow_accepts_column_ordering(self) -> None:
        cfw = PyArrowTable(ParallelizationMode.SYNC, frozenset(), uuid4())
        data = pa.table({"Z": [1], "A": [2], "M": [3]})
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="alphabetical")
        assert result.column_names == ["A", "M", "Z"]

    @pytest.mark.skipif(pl is None, reason="Polars is not installed")
    def test_polars_accepts_column_ordering(self) -> None:
        cfw = PolarsDataFrame(ParallelizationMode.SYNC, frozenset(), uuid4())
        data = pl.DataFrame({"Z": [1], "A": [2], "M": [3]})
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="alphabetical")
        assert result.columns == ["A", "M", "Z"]

    @pytest.mark.skipif(pl is None, reason="Polars is not installed")
    def test_polars_lazy_accepts_column_ordering(self) -> None:
        cfw = PolarsLazyDataFrame(ParallelizationMode.SYNC, frozenset(), uuid4())
        data = pl.DataFrame({"Z": [1], "A": [2], "M": [3]}).lazy()
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="alphabetical")
        # PolarsLazyDataFrame.select_data_by_column_names() returns collected DataFrame
        assert result.columns == ["A", "M", "Z"]

    @pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed")
    def test_duckdb_accepts_column_ordering(self) -> None:
        cfw = DuckDBFramework(ParallelizationMode.SYNC, frozenset(), uuid4())
        conn = duckdb.connect(":memory:")
        # DuckDB expects a DuckDBPyRelation
        data = conn.sql("SELECT 1 as Z, 2 as A, 3 as M")
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="alphabetical")
        assert list(result.columns) == ["A", "M", "Z"]

    def test_python_dict_accepts_column_ordering(self) -> None:
        cfw = PythonDictFramework(ParallelizationMode.SYNC, frozenset(), uuid4())
        # PythonDictFramework expects List[Dict] format (rows)
        data = [{"Z": 1, "A": 2, "M": 3}]
        features = {FeatureName("Z"), FeatureName("A"), FeatureName("M")}
        result = cfw.select_data_by_column_names(data, features, column_ordering="alphabetical")
        assert list(result[0].keys()) == ["A", "M", "Z"]


# --- End-to-End Test Features ---


class MultiFeatureGroup(FeatureGroup):
    """Test feature group that supports 5 features (FeatureA through FeatureE)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"FeatureA", "FeatureB", "FeatureC", "FeatureD", "FeatureE"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result = {}
        requested_names = features.get_all_names()
        feature_values = {
            "FeatureA": [1, 2, 3],
            "FeatureB": [4, 5, 6],
            "FeatureC": [7, 8, 9],
            "FeatureD": [10, 11, 12],
            "FeatureE": [13, 14, 15],
        }
        for name in requested_names:
            if name in feature_values:
                result[name] = feature_values[name]
        return result

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"FeatureA", "FeatureB", "FeatureC", "FeatureD", "FeatureE"}


# --- End-to-End Column Ordering Tests ---


class TestEndToEndColumnOrdering:
    """Test column ordering through the full mloda.run_all() API."""

    def test_alphabetical_ordering_returns_columns_in_alphabetical_order(self) -> None:
        """Request features in non-alphabetical order, verify result is alphabetically sorted."""
        plugin_collector = PluginCollector.enabled_feature_groups({MultiFeatureGroup})

        # Request features in non-alphabetical order: E, C, A, D, B
        result = mloda.run_all(
            [
                Feature(name="FeatureE"),
                Feature(name="FeatureC"),
                Feature(name="FeatureA"),
                Feature(name="FeatureD"),
                Feature(name="FeatureB"),
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
            column_ordering="alphabetical",
        )

        # Result should be a list with one DataFrame (one compute framework)
        assert len(result) == 1
        df = result[0]

        # Columns should be in alphabetical order regardless of request order
        assert list(df.columns) == ["FeatureA", "FeatureB", "FeatureC", "FeatureD", "FeatureE"]
