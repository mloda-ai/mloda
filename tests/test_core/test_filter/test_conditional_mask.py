from typing import Any, Optional, Set, Type

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import BaseFilterEngine
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.user import Feature
from mloda.user import Features
from mloda.user import GlobalFilter
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


class InlineMaskFilterEngine(PyArrowFilterEngine):
    """Filter engine that disables final (post-calculation) filtering.

    By returning False from final_filters(), the engine delegates filtering
    responsibility to the FeatureGroup's calculate_feature method, enabling
    inline conditional masking instead of row elimination.
    """

    @classmethod
    def final_filters(cls) -> bool:
        return False


class PyArrowTableInlineFilter(PyArrowTable):
    """Compute framework using inline filtering instead of final filtering."""

    @classmethod
    def filter_engine(cls) -> Type[BaseFilterEngine]:
        return InlineMaskFilterEngine


class ConditionalMaskExample(FeatureGroup):
    """Example FeatureGroup demonstrating inline filter usage for conditional masking.

    Creates source data with region, status, and value columns. Reads filters
    from features.filters to build a mask, replaces non-matching values with
    NULL (preserving all rows), groups by region, sums the masked values, and
    broadcasts the grouped sums back to all rows.

    Expected source data:
        region=["A","A","B","B"], status=["active","inactive","active","inactive"], value=[10,20,30,40]

    With filter status=="active", the masked values become [10, None, 30, None].
    Grouped sum by region: A=10, B=30.
    Broadcast back: [10, 10, 30, 30].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {PyArrowTableInlineFilter}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B"])
        status = pa.array(["active", "inactive", "active", "inactive"])
        value = pa.array([10, 20, 30, 40])

        if features.filters is None:
            raise ValueError("No filters provided")

        mask = pa.array([True] * 4)
        for f in features.filters:
            if f.filter_feature.name == "status" and f.filter_type == "equal":
                mask = pc.and_(mask, pc.equal(status, pa.scalar(f.parameter.value)))

        masked_value = pc.if_else(mask, value, pa.scalar(None, type=pa.int64()))

        table = pa.table({"region": region, "masked_value": masked_value})
        grouped = table.group_by("region").aggregate([("masked_value", "sum")])

        region_to_sum = {}
        for i in range(grouped.num_rows):
            region_to_sum[grouped.column("region")[i].as_py()] = grouped.column("masked_value_sum")[i].as_py()

        result_array = pa.array([region_to_sum[r.as_py()] for r in region])
        return pa.table({cls.get_class_name(): result_array})


class RawMaskExample(FeatureGroup):
    """Variant FeatureGroup that returns the raw masked column before aggregation.

    Used to verify that masking produces NULLs in the correct positions rather
    than eliminating rows. With filter status=="active", the expected result is
    [10, None, 30, None].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {PyArrowTableInlineFilter}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        status = pa.array(["active", "inactive", "active", "inactive"])
        value = pa.array([10, 20, 30, 40])

        if features.filters is None:
            raise ValueError("No filters provided")

        mask = pa.array([True] * 4)
        for f in features.filters:
            if f.filter_feature.name == "status" and f.filter_type == "equal":
                mask = pc.and_(mask, pc.equal(status, pa.scalar(f.parameter.value)))

        masked_value = pc.if_else(mask, value, pa.scalar(None, type=pa.int64()))
        return pa.table({cls.get_class_name(): masked_value})


class FilterPresenceChecker(FeatureGroup):
    """FeatureGroup that verifies filters are received in calculate_feature.

    Raises ValueError if features.filters is None or empty, confirming that
    inline filter engines correctly pass filter information to the FeatureGroup.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {PyArrowTableInlineFilter}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.filters is None or len(features.filters) == 0:
            raise ValueError("Expected filter not found")

        found = False
        for f in features.filters:
            if f.filter_type == "equal" and f.parameter.value == "active":
                found = True

        if not found:
            raise ValueError("Expected filter not found")

        return pa.table({cls.get_class_name(): [1, 1, 1, 1]})


@PARALLELIZATION_MODES_SYNC_THREADING
class TestConditionalMask:
    def test_inline_filter_conditional_masking(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """Inline filter produces grouped sums broadcast to all rows.

        With status=="active" filter applied as a conditional mask:
        - Region A active values sum to 10
        - Region B active values sum to 30
        - All 4 rows preserved with broadcast sums: [10, 10, 30, 30]
        """
        features = Features([Feature(name="ConditionalMaskExample", initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        run_result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTableInlineFilter},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        assert len(run_result.results) == 1
        result = run_result.results[0].to_pydict()
        assert result == {"ConditionalMaskExample": [10, 10, 30, 30]}

    def test_all_rows_preserved(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """Inline filtering preserves all rows (no row elimination).

        The output must contain exactly 4 rows, matching the original data size,
        because masking replaces non-matching values with NULL instead of
        removing rows.
        """
        features = Features([Feature(name="ConditionalMaskExample", initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        run_result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTableInlineFilter},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        assert len(run_result.results) == 1
        result = run_result.results[0].to_pydict()
        assert len(result["ConditionalMaskExample"]) == 4

    def test_filters_received_in_calculate_feature(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """Verify that features.filters is populated when using inline filtering.

        The FilterPresenceChecker FeatureGroup validates that the filter engine
        correctly passes filter information through features.filters. When
        implemented, it should confirm filters are present and raise ValueError
        if they are missing.
        """
        features = Features([Feature(name="FilterPresenceChecker", initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        run_result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTableInlineFilter},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        assert len(run_result.results) == 1
        result = run_result.results[0].to_pydict()
        assert "FilterPresenceChecker" in result

    def test_masking_produces_null_not_elimination(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """Masking sets non-matching values to NULL rather than removing rows.

        The RawMaskExample returns the masked column before aggregation.
        With status=="active", rows at index 1 and 3 (inactive) should be NULL,
        producing [10, None, 30, None].
        """
        features = Features([Feature(name="RawMaskExample", initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        run_result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTableInlineFilter},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        assert len(run_result.results) == 1
        result = run_result.results[0].to_pydict()
        assert result == {"RawMaskExample": [10, None, 30, None]}
