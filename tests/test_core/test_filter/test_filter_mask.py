"""Tests for the FilterMask orchestrator: error handling, setup wiring, and integration.

Per-framework mask engine tests live in the framework-specific test directories and
use the shared FilterMaskEngineTestMixin (see filter_mask_engine_test_mixin.py).
"""

from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet, BaseInputData
from mloda.provider import FilterMask, BaseFilterMaskEngine
from mloda.user import Feature, Features, GlobalFilter, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_mask_engine import (
    PyArrowFilterMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_mask_engine import (
    PandasFilterMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_mask_engine import (
    PythonDictFilterMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_mask_engine import (
    DuckDBFilterMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_filter_mask_engine import (
    SqliteFilterMaskEngine,
)
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_mask_engine import (
        PolarsFilterMaskEngine,
    )
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
except ImportError:
    pl = None  # type: ignore[assignment]
    PolarsFilterMaskEngine = None  # type: ignore[assignment, misc]
    PolarsDataFrame = None  # type: ignore[assignment, misc]

try:
    from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
    from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_mask_engine import (
        SparkFilterMaskEngine,
    )

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkFramework = None  # type: ignore[assignment, misc]
    SparkFilterMaskEngine = None  # type: ignore[assignment, misc]


def _make_features(
    filters: set[SingleFilter] | None,
    engine: type[BaseFilterMaskEngine],
) -> FeatureSet:
    """Create a FeatureSet with pre-resolved mask engine and filters."""
    fs = FeatureSet()
    fs.filters = filters
    fs.filter_mask_engine = engine
    return fs


# ---------------------------------------------------------------------------
# 1. Error tests
# ---------------------------------------------------------------------------


class TestFilterMaskErrors:
    def test_missing_engine_raises(self) -> None:
        """When filter_mask_engine is not set, raise TypeError with clear message."""
        sf = SingleFilter("col", "equal", {"value": "x"})
        features = FeatureSet()
        features.filters = {sf}
        with pytest.raises(TypeError, match="No filter_mask_engine set on features"):
            FilterMask.build("not a table", features, column="col")

    def test_unsupported_filter_type_regex(self) -> None:
        sf = SingleFilter("status", "regex", {"value": "act.*"})
        features = _make_features({sf}, PyArrowFilterMaskEngine)
        table = pa.table({"status": ["active", "inactive"]})
        with pytest.raises(ValueError, match="not supported for mask building"):
            FilterMask.build(table, features, column="status")

    def test_unsupported_filter_type_custom(self) -> None:
        sf = SingleFilter("status", "some_custom_type", {"value": "x"})
        features = _make_features({sf}, PyArrowFilterMaskEngine)
        table = pa.table({"status": ["active", "inactive"]})
        with pytest.raises(ValueError, match="not supported for mask building"):
            FilterMask.build(table, features, column="status")


# ---------------------------------------------------------------------------
# 2. Setup wiring tests (all frameworks)
# ---------------------------------------------------------------------------


class TestFilterMaskSetupWiring:
    def test_pyarrow_framework_provides_mask_engine(self) -> None:
        assert PyArrowTable.filter_mask_engine() is PyArrowFilterMaskEngine

    def test_pandas_framework_provides_mask_engine(self) -> None:
        assert PandasDataFrame.filter_mask_engine() is PandasFilterMaskEngine

    def test_python_dict_framework_provides_mask_engine(self) -> None:
        assert PythonDictFramework.filter_mask_engine() is PythonDictFilterMaskEngine

    def test_duckdb_framework_provides_mask_engine(self) -> None:
        assert DuckDBFramework.filter_mask_engine() is DuckDBFilterMaskEngine

    def test_sqlite_framework_provides_mask_engine(self) -> None:
        assert SqliteFramework.filter_mask_engine() is SqliteFilterMaskEngine

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_framework_provides_mask_engine(self) -> None:
        assert PolarsDataFrame.filter_mask_engine() is PolarsFilterMaskEngine

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="pyspark not installed")
    def test_spark_framework_provides_mask_engine(self) -> None:
        assert SparkFramework.filter_mask_engine() is SparkFilterMaskEngine

    def test_set_filter_engine_wires_mask_engine(self) -> None:
        """set_filter_engine() sets filter_mask_engine on the FeatureSet."""
        cfw = PyArrowTable(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
        )
        features = FeatureSet()
        assert features.filter_mask_engine is None
        cfw.set_filter_engine(features)
        assert features.filter_mask_engine is PyArrowFilterMaskEngine


# ---------------------------------------------------------------------------
# 3. Integration test with MlodaTestRunner
# ---------------------------------------------------------------------------


class FilterMaskFeatureGroup(FeatureGroup):
    """FeatureGroup that uses FilterMask.build() to apply inline filters."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B"])
        status = pa.array(["active", "inactive", "active", "inactive"])
        value = pa.array([10, 20, 30, 40])

        mask = FilterMask.build(pa.table({"status": status}), features, column="status")
        masked_value = pc.if_else(mask, value, None)

        result = []
        for i in range(len(region)):
            region_i = region[i].as_py()
            total = 0
            for j in range(len(region)):
                if region[j].as_py() == region_i and masked_value[j].is_valid:
                    total += masked_value[j].as_py()
            result.append(total)

        return pa.table({cls.get_class_name(): result, "status": status})


class FilterMaskNoFilterFeatureGroup(FeatureGroup):
    """FilterMask.build() with no global filter returns all-True mask.

    When no filters target 'status', all values contribute to the region sum.
    Region A: 10 + 20 = 30, Region B: 30 + 40 = 70. Result: [30, 30, 70, 70].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B"])
        status = pa.array(["active", "inactive", "active", "inactive"])
        value = pa.array([10, 20, 30, 40])

        mask = FilterMask.build(pa.table({"status": status}), features, column="status")
        masked_value = pc.if_else(mask, value, None)

        result = []
        for i in range(len(region)):
            region_i = region[i].as_py()
            total = 0
            for j in range(len(region)):
                if region[j].as_py() == region_i and masked_value[j].is_valid:
                    total += masked_value[j].as_py()
            result.append(total)

        return pa.table({cls.get_class_name(): result, "status": status})


class FilterMaskRangeFeatureGroup(FeatureGroup):
    """FilterMask.build() with range filter on numeric 'value' column.

    Data: region=[A,A,B,B], value=[10,20,30,40], weight=[1,2,3,4].
    With filter value in [15, 35]: mask=[F,T,T,F], masked weights=[None,2,3,None].
    Region A: 2, Region B: 3. Result: [2, 2, 3, 3].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B"])
        value = pa.array([10, 20, 30, 40])
        weight = pa.array([1, 2, 3, 4])

        mask = FilterMask.build(pa.table({"value": value}), features, column="value")
        masked_weight = pc.if_else(mask, weight, None)

        result = []
        for i in range(len(region)):
            region_i = region[i].as_py()
            total = 0
            for j in range(len(region)):
                if region[j].as_py() == region_i and masked_weight[j].is_valid:
                    total += masked_weight[j].as_py()
            result.append(total)

        return pa.table({cls.get_class_name(): result, "value": value})


class FilterMaskCategoricalFeatureGroup(FeatureGroup):
    """FilterMask.build() with categorical_inclusion filter.

    Data: region=[A,A,B,B,B], status=[active,pending,inactive,active,pending],
          value=[10,20,30,40,50].
    With filter status in ("active","pending"): mask=[T,T,F,T,T],
    masked values=[10,20,None,40,50]. Region A: 30, Region B: 90.
    Result: [30, 30, 90, 90, 90].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B", "B"])
        status = pa.array(["active", "pending", "inactive", "active", "pending"])
        value = pa.array([10, 20, 30, 40, 50])

        mask = FilterMask.build(pa.table({"status": status}), features, column="status")
        masked_value = pc.if_else(mask, value, None)

        result = []
        for i in range(len(region)):
            region_i = region[i].as_py()
            total = 0
            for j in range(len(region)):
                if region[j].as_py() == region_i and masked_value[j].is_valid:
                    total += masked_value[j].as_py()
            result.append(total)

        return pa.table({cls.get_class_name(): result, "status": status})


class FilterMaskMultiFilterFeatureGroup(FeatureGroup):
    """FilterMask.build() AND-combines min and max filters on same column.

    Data: region=[A,A,B,B,B], value=[5,15,25,35,45], weight=[1,2,3,4,5].
    With filter value >= 10 AND value <= 35: mask=[F,T,T,T,F],
    masked weights=[None,2,3,4,None]. Region A: 2, Region B: 7.
    Result: [2, 2, 7, 7, 7].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B", "B"])
        value = pa.array([5, 15, 25, 35, 45])
        weight = pa.array([1, 2, 3, 4, 5])

        mask = FilterMask.build(pa.table({"value": value}), features, column="value")
        masked_weight = pc.if_else(mask, weight, None)

        result = []
        for i in range(len(region)):
            region_i = region[i].as_py()
            total = 0
            for j in range(len(region)):
                if region[j].as_py() == region_i and masked_weight[j].is_valid:
                    total += masked_weight[j].as_py()
            result.append(total)

        return pa.table({cls.get_class_name(): result, "value": value})


@PARALLELIZATION_MODES_SYNC_THREADING
class TestFilterMaskIntegration:
    def test_filter_mask_produces_same_result_as_manual_boilerplate(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """FilterMaskFeatureGroup should produce [10, 10, 30, 30]."""
        feature_name = "FilterMaskFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])
        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [10, 10, 30, 30]

    def test_filter_mask_row_count_preserved_with_equal_filter(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """Explicit row count assertion: all 4 rows preserved, not eliminated."""
        feature_name = "FilterMaskFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])
        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [10, 10, 30, 30]
            assert len(data[feature_name]) == 4

    def test_filter_mask_no_filter_produces_all_true_mask(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """When no GlobalFilter is set, all values contribute to the region sum.

        Region A: 10 + 20 = 30, Region B: 30 + 40 = 70. Result: [30, 30, 70, 70].
        """
        feature_name = "FilterMaskNoFilterFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [30, 30, 70, 70]
            assert len(data[feature_name]) == 4

    def test_filter_mask_range_filter(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FilterMask with range filter masks values outside [15, 35].

        Mask: [F, T, T, F]. Masked weights: [None, 2, 3, None].
        Region A: 2, Region B: 3. Result: [2, 2, 3, 3].
        """
        feature_name = "FilterMaskRangeFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])
        global_filter = GlobalFilter()
        global_filter.add_filter("value", "range", {"min": 15, "max": 35})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [2, 2, 3, 3]
            assert len(data[feature_name]) == 4

    def test_filter_mask_categorical_inclusion(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FilterMask with categorical_inclusion masks out 'inactive'.

        Mask: [T, T, F, T, T]. Masked values: [10, 20, None, 40, 50].
        Region A: 30, Region B: 90. Result: [30, 30, 90, 90, 90].
        """
        feature_name = "FilterMaskCategoricalFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])
        global_filter = GlobalFilter()
        global_filter.add_filter("status", "categorical_inclusion", {"values": ("active", "pending")})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [30, 30, 90, 90, 90]
            assert len(data[feature_name]) == 5

    def test_filter_mask_multiple_filters_and_combined(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """FilterMask AND-combines min and max filters on same column.

        Filters: value >= 10 AND value <= 35.
        Mask: [F, T, T, T, F]. Masked weights: [None, 2, 3, 4, None].
        Region A: 2, Region B: 7. Result: [2, 2, 7, 7, 7].
        """
        feature_name = "FilterMaskMultiFilterFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])
        global_filter = GlobalFilter()
        global_filter.add_filter("value", "min", {"value": 10})
        global_filter.add_filter("value", "max", {"value": 35})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [2, 2, 7, 7, 7]
            assert len(data[feature_name]) == 5
