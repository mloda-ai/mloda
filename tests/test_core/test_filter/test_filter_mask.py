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
