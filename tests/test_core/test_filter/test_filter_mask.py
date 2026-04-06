"""Tests for the FilterMask builder utility.

FilterMask.build() replaces the boilerplate of manually iterating over SingleFilter
objects and constructing boolean masks in calculate_feature(). It accepts a
framework-native data container, an optional set of SingleFilter objects, and a
column name, and returns a framework-native boolean mask.

These tests cover PyArrow, Pandas, Polars, and plain Python dict inputs, as well
as error handling and an integration test with MlodaTestRunner.
"""

from typing import Any, Optional

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from mloda.core.filter.filter_mask import FilterMask
from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet, BaseInputData
from mloda.user import Feature, Features, GlobalFilter, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


# ---------------------------------------------------------------------------
# 1. PyArrow tests
# ---------------------------------------------------------------------------


class TestPyArrowFilterMask:
    """FilterMask.build() with PyArrow tables."""

    def _table(self) -> pa.Table:
        return pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def test_pyarrow_equal_filter(self) -> None:
        """Equal filter on 'status' with value 'active' selects rows 0 and 2."""
        sf = SingleFilter("status", "equal", {"value": "active"})
        mask = FilterMask.build(self._table(), {sf}, column="status")
        assert mask.to_pylist() == [True, False, True, False]

    def test_pyarrow_min_filter(self) -> None:
        """Min filter on 'value' with value 20 selects rows where value >= 20."""
        sf = SingleFilter("value", "min", {"value": 20})
        mask = FilterMask.build(self._table(), {sf}, column="value")
        assert mask.to_pylist() == [False, True, True, True]

    def test_pyarrow_max_filter_simple(self) -> None:
        """Max filter on 'value' with {'value': 30} selects rows where value <= 30."""
        sf = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._table(), {sf}, column="value")
        assert mask.to_pylist() == [True, True, True, False]

    def test_pyarrow_max_filter_exclusive(self) -> None:
        """Max filter with max_exclusive=True uses strict less-than comparison."""
        sf = SingleFilter("value", "max", {"max": 30, "max_exclusive": True})
        mask = FilterMask.build(self._table(), {sf}, column="value")
        assert mask.to_pylist() == [True, True, False, False]

    def test_pyarrow_range_filter(self) -> None:
        """Range filter on 'value' with min=15, max=35 selects rows in [15, 35]."""
        sf = SingleFilter("value", "range", {"min": 15, "max": 35})
        mask = FilterMask.build(self._table(), {sf}, column="value")
        assert mask.to_pylist() == [False, True, True, False]

    def test_pyarrow_range_filter_exclusive(self) -> None:
        """Range filter with max_exclusive=True: min=15, max=35. 30 < 35 so still True."""
        sf = SingleFilter("value", "range", {"min": 15, "max": 35, "max_exclusive": True})
        mask = FilterMask.build(self._table(), {sf}, column="value")
        assert mask.to_pylist() == [False, True, True, False]

    def test_pyarrow_categorical_inclusion(self) -> None:
        """Categorical inclusion on 'status' with values=['active'] selects active rows."""
        sf = SingleFilter("status", "categorical_inclusion", {"values": ["active"]})
        mask = FilterMask.build(self._table(), {sf}, column="status")
        assert mask.to_pylist() == [True, False, True, False]

    def test_pyarrow_no_matching_filters(self) -> None:
        """Filters for column 'other' should produce all-True mask when building for 'status'."""
        sf = SingleFilter("other", "equal", {"value": "x"})
        mask = FilterMask.build(self._table(), {sf}, column="status")
        assert mask.to_pylist() == [True, True, True, True]

    def test_pyarrow_none_filters(self) -> None:
        """None filters produce an all-True mask."""
        mask = FilterMask.build(self._table(), None, column="status")
        assert mask.to_pylist() == [True, True, True, True]

    def test_pyarrow_empty_filters(self) -> None:
        """Empty filter set produces an all-True mask."""
        mask = FilterMask.build(self._table(), set(), column="status")
        assert mask.to_pylist() == [True, True, True, True]

    def test_pyarrow_multiple_filters_same_column(self) -> None:
        """Two filters on 'value': min>=20 AND max<=30 selects rows 1 and 2."""
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._table(), {sf_min, sf_max}, column="value")
        assert mask.to_pylist() == [False, True, True, False]


# ---------------------------------------------------------------------------
# 2. Pandas tests
# ---------------------------------------------------------------------------


class TestPandasFilterMask:
    """FilterMask.build() with Pandas DataFrames."""

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def test_pandas_equal_filter(self) -> None:
        """Equal filter on 'status' with value 'active' selects rows 0 and 2."""
        sf = SingleFilter("status", "equal", {"value": "active"})
        mask = FilterMask.build(self._df(), {sf}, column="status")
        assert list(mask) == [True, False, True, False]

    def test_pandas_min_filter(self) -> None:
        """Min filter on 'value' with value 20 selects rows where value >= 20."""
        sf = SingleFilter("value", "min", {"value": 20})
        mask = FilterMask.build(self._df(), {sf}, column="value")
        assert list(mask) == [False, True, True, True]

    def test_pandas_max_filter_simple(self) -> None:
        """Max filter on 'value' with {'value': 30} selects rows where value <= 30."""
        sf = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._df(), {sf}, column="value")
        assert list(mask) == [True, True, True, False]

    def test_pandas_categorical_inclusion(self) -> None:
        """Categorical inclusion on 'status' with values=['active'] selects active rows."""
        sf = SingleFilter("status", "categorical_inclusion", {"values": ["active"]})
        mask = FilterMask.build(self._df(), {sf}, column="status")
        assert list(mask) == [True, False, True, False]

    def test_pandas_none_filters(self) -> None:
        """None filters produce an all-True mask."""
        mask = FilterMask.build(self._df(), None, column="status")
        assert list(mask) == [True, True, True, True]

    def test_pandas_multiple_filters_same_column(self) -> None:
        """Two filters on 'value': min>=20 AND max<=30 selects rows 1 and 2."""
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._df(), {sf_min, sf_max}, column="value")
        assert list(mask) == [False, True, True, False]


# ---------------------------------------------------------------------------
# 3. Polars tests
# ---------------------------------------------------------------------------


class TestPolarsFilterMask:
    """FilterMask.build() with Polars DataFrames."""

    def _df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def test_polars_equal_filter(self) -> None:
        """Equal filter on 'status' with value 'active' selects rows 0 and 2."""
        sf = SingleFilter("status", "equal", {"value": "active"})
        mask = FilterMask.build(self._df(), {sf}, column="status")
        assert list(mask) == [True, False, True, False]

    def test_polars_min_filter(self) -> None:
        """Min filter on 'value' with value 20 selects rows where value >= 20."""
        sf = SingleFilter("value", "min", {"value": 20})
        mask = FilterMask.build(self._df(), {sf}, column="value")
        assert list(mask) == [False, True, True, True]

    def test_polars_max_filter_simple(self) -> None:
        """Max filter on 'value' with {'value': 30} selects rows where value <= 30."""
        sf = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._df(), {sf}, column="value")
        assert list(mask) == [True, True, True, False]

    def test_polars_categorical_inclusion(self) -> None:
        """Categorical inclusion on 'status' with values=['active'] selects active rows."""
        sf = SingleFilter("status", "categorical_inclusion", {"values": ["active"]})
        mask = FilterMask.build(self._df(), {sf}, column="status")
        assert list(mask) == [True, False, True, False]

    def test_polars_none_filters(self) -> None:
        """None filters produce an all-True mask."""
        mask = FilterMask.build(self._df(), None, column="status")
        assert list(mask) == [True, True, True, True]

    def test_polars_multiple_filters_same_column(self) -> None:
        """Two filters on 'value': min>=20 AND max<=30 selects rows 1 and 2."""
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._df(), {sf_min, sf_max}, column="value")
        assert list(mask) == [False, True, True, False]


# ---------------------------------------------------------------------------
# 4. PythonDict tests
# ---------------------------------------------------------------------------


class TestPythonDictFilterMask:
    """FilterMask.build() with list[dict] data."""

    def _data(self) -> list[dict[str, Any]]:
        return [
            {"status": "active", "value": 10},
            {"status": "inactive", "value": 20},
            {"status": "active", "value": 30},
            {"status": "inactive", "value": 40},
        ]

    def test_python_dict_equal_filter(self) -> None:
        """Equal filter on 'status' with value 'active' selects rows 0 and 2."""
        sf = SingleFilter("status", "equal", {"value": "active"})
        mask = FilterMask.build(self._data(), {sf}, column="status")
        assert list(mask) == [True, False, True, False]

    def test_python_dict_min_filter(self) -> None:
        """Min filter on 'value' with value 20 selects rows where value >= 20."""
        sf = SingleFilter("value", "min", {"value": 20})
        mask = FilterMask.build(self._data(), {sf}, column="value")
        assert list(mask) == [False, True, True, True]

    def test_python_dict_none_filters(self) -> None:
        """None filters produce an all-True mask."""
        mask = FilterMask.build(self._data(), None, column="status")
        assert list(mask) == [True, True, True, True]

    def test_python_dict_multiple_filters(self) -> None:
        """Two filters on 'value': min>=20 AND max<=30 selects rows 1 and 2."""
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        mask = FilterMask.build(self._data(), {sf_min, sf_max}, column="value")
        assert list(mask) == [False, True, True, False]


# ---------------------------------------------------------------------------
# 5. Error tests
# ---------------------------------------------------------------------------


class TestFilterMaskErrors:
    """Error handling in FilterMask.build()."""

    def test_unsupported_data_type(self) -> None:
        """Passing a raw string as data should raise ValueError."""
        sf = SingleFilter("col", "equal", {"value": "x"})
        with pytest.raises(ValueError):
            FilterMask.build("not a table", {sf}, column="col")

    def test_unsupported_filter_type(self) -> None:
        """Passing a SingleFilter with filter_type='regex' should raise ValueError."""
        sf = SingleFilter("status", "regex", {"value": "act.*"})
        table = pa.table({"status": ["active", "inactive"]})
        with pytest.raises(ValueError):
            FilterMask.build(table, {sf}, column="status")


# ---------------------------------------------------------------------------
# 6. Integration test with MlodaTestRunner
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

        mask = FilterMask.build(pa.table({"status": status}), features.filters, column="status")
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
    """Integration: FilterMask.build() inside a FeatureGroup matches manual boilerplate."""

    def test_filter_mask_produces_same_result_as_manual_boilerplate(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """FilterMaskFeatureGroup using FilterMask.build() should produce [10, 10, 30, 30].

        This is the same result as InlineMaskViaFeatureGroup which uses the manual
        boilerplate loop. The two approaches must be equivalent.
        """
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
