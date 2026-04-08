"""Tests for mask engine: setup wiring and integration with FeatureGroups.

Per-framework mask engine tests live in the framework-specific test directories and
use the shared MaskEngineTestMixin (see mask_engine_test_mixin.py).
"""

from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet, BaseInputData
from mloda.user import Feature, Features, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import (
    PyArrowMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_mask_engine import (
    DuckDBMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_mask_engine import (
    SqliteMaskEngine,
)
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.polars_mask_engine import (
        PolarsMaskEngine,
    )
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
except ImportError:
    pl = None  # type: ignore[assignment]
    PolarsMaskEngine = None  # type: ignore[assignment, misc]
    PolarsDataFrame = None  # type: ignore[assignment, misc]

try:
    from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
    from mloda_plugins.compute_framework.base_implementations.spark.spark_mask_engine import (
        SparkMaskEngine,
    )

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkFramework = None  # type: ignore[assignment, misc]
    SparkMaskEngine = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# 1. Setup wiring tests (all frameworks)
# ---------------------------------------------------------------------------


class TestMaskEngineSetupWiring:
    def test_pyarrow_framework_provides_mask_engine(self) -> None:
        assert PyArrowTable.mask_engine() is PyArrowMaskEngine

    def test_pandas_framework_provides_mask_engine(self) -> None:
        assert PandasDataFrame.mask_engine() is PandasMaskEngine

    def test_python_dict_framework_provides_mask_engine(self) -> None:
        assert PythonDictFramework.mask_engine() is PythonDictMaskEngine

    def test_duckdb_framework_provides_mask_engine(self) -> None:
        assert DuckDBFramework.mask_engine() is DuckDBMaskEngine

    def test_sqlite_framework_provides_mask_engine(self) -> None:
        assert SqliteFramework.mask_engine() is SqliteMaskEngine

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_framework_provides_mask_engine(self) -> None:
        assert PolarsDataFrame.mask_engine() is PolarsMaskEngine

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="pyspark not installed")
    def test_spark_framework_provides_mask_engine(self) -> None:
        assert SparkFramework.mask_engine() is SparkMaskEngine

    def test_set_mask_engine_wires_mask_engine(self) -> None:
        """set_mask_engine() sets mask_engine on the FeatureSet."""
        cfw = PyArrowTable(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
        )
        features = FeatureSet()
        assert features.mask_engine is None
        cfw.set_mask_engine(features)
        assert features.mask_engine is PyArrowMaskEngine


# ---------------------------------------------------------------------------
# 2. Integration test with MlodaTestRunner
# ---------------------------------------------------------------------------


class MaskEngineFeatureGroup(FeatureGroup):
    """FeatureGroup that uses the mask engine to apply inline masking."""

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

        assert features.mask_engine is not None
        engine = features.mask_engine
        mask = engine.equal(pa.table({"status": status}), "status", "active")
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


class MaskEngineNoMaskFeatureGroup(FeatureGroup):
    """Mask engine with all_true returns all-True mask.

    All values contribute to the region sum.
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

        assert features.mask_engine is not None
        engine = features.mask_engine
        mask = engine.all_true(pa.table({"status": status}))
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


class MaskEngineRangeFeatureGroup(FeatureGroup):
    """Mask engine with range mask via combine(greater_equal, less_equal).

    Data: region=[A,A,B,B], value=[10,20,30,40], weight=[1,2,3,4].
    With mask value in [15, 35]: mask=[F,T,T,F], masked weights=[None,2,3,None].
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

        assert features.mask_engine is not None
        engine = features.mask_engine
        table = pa.table({"value": value})
        mask_min = engine.greater_equal(table, "value", 15)
        mask_max = engine.less_equal(table, "value", 35)
        mask = engine.combine(mask_min, mask_max)
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


class MaskEngineCategoricalFeatureGroup(FeatureGroup):
    """Mask engine with categorical mask via is_in.

    Data: region=[A,A,B,B,B], status=[active,pending,inactive,active,pending],
          value=[10,20,30,40,50].
    With mask status in ("active","pending"): mask=[T,T,F,T,T],
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

        assert features.mask_engine is not None
        engine = features.mask_engine
        mask = engine.is_in(pa.table({"status": status}), "status", ("active", "pending"))
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


class MaskEngineMultiMaskFeatureGroup(FeatureGroup):
    """Mask engine with AND-combined masks on same column.

    Data: region=[A,A,B,B,B], value=[5,15,25,35,45], weight=[1,2,3,4,5].
    With mask value >= 10 AND value <= 35: mask=[F,T,T,T,F],
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

        assert features.mask_engine is not None
        engine = features.mask_engine
        table = pa.table({"value": value})
        mask_min = engine.greater_equal(table, "value", 10)
        mask_max = engine.less_equal(table, "value", 35)
        mask = engine.combine(mask_min, mask_max)
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
class TestMaskEngineIntegration:
    def test_mask_engine_equal_produces_correct_result(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """MaskEngineFeatureGroup should produce [10, 10, 30, 30]."""
        feature_name = "MaskEngineFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [10, 10, 30, 30]

    def test_mask_engine_row_count_preserved(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Explicit row count assertion: all 4 rows preserved, not eliminated."""
        feature_name = "MaskEngineFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [10, 10, 30, 30]
            assert len(data[feature_name]) == 4

    def test_mask_engine_all_true(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """all_true mask: all values contribute. Region A: 30, B: 70."""
        feature_name = "MaskEngineNoMaskFeatureGroup"
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

    def test_mask_engine_range(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Range mask via combine(greater_equal, less_equal). Result: [2, 2, 3, 3]."""
        feature_name = "MaskEngineRangeFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [2, 2, 3, 3]
            assert len(data[feature_name]) == 4

    def test_mask_engine_categorical(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Categorical mask via is_in. Result: [30, 30, 90, 90, 90]."""
        feature_name = "MaskEngineCategoricalFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [30, 30, 90, 90, 90]
            assert len(data[feature_name]) == 5

    def test_mask_engine_multi_mask_combined(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """AND-combined masks. Result: [2, 2, 7, 7, 7]."""
        feature_name = "MaskEngineMultiMaskFeatureGroup"
        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
        )

        for res in result.results:
            data = res.to_pydict()
            assert data[feature_name] == [2, 2, 7, 7, 7]
            assert len(data[feature_name]) == 5
