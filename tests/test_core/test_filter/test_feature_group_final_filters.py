"""Tests that a FeatureGroup can control whether filters are applied inline vs. final.

Currently, `final_filters()` lives on the FilterEngine (tied to the ComputeFramework).
If a FeatureGroup wants inline filtering, it must create boilerplate subclasses of both
the FilterEngine and ComputeFramework. These tests prove that a FeatureGroup can
override `final_filters()` directly, without any custom compute framework or filter
engine subclasses.

Expected failures:
- FeatureGroup does not yet expose a `final_filters()` method.
- `run_final_filter()` does not yet consult the FeatureGroup for this decision.
"""

from typing import Any, Optional, Set, Type

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.provider import BaseInputData
from mloda.user import Feature, Features, GlobalFilter, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


class InlineMaskViaFeatureGroup(FeatureGroup):
    """A FeatureGroup that handles filters inline instead of relying on final filtering.

    Uses the STANDARD PyArrowTable compute framework. No custom FilterEngine or
    ComputeFramework subclass is needed.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        """Signal that this FeatureGroup handles filters inline, not as a final step."""
        return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "B", "B"])
        status = pa.array(["active", "inactive", "active", "inactive"])
        value = pa.array([10, 20, 30, 40])

        # Read filters from features and apply as inline mask
        mask = pa.array([True, True, True, True])
        if features.filters is not None:
            for single_filter in features.filters:
                if single_filter.name == "status":
                    filter_value = single_filter.parameter.value
                    mask = pc.and_(mask, pc.equal(status, filter_value))

        # Mask non-matching values to null, then sum by region and broadcast back
        masked_value = pc.if_else(mask, value, None)

        # Group-by sum: for each row, sum all values in the same region
        result = []
        for i in range(len(region)):
            region_i = region[i].as_py()
            total = 0
            for j in range(len(region)):
                if region[j].as_py() == region_i and masked_value[j].is_valid:
                    total += masked_value[j].as_py()
            result.append(total)

        return pa.table({
            cls.get_class_name(): result,
            "status": status,
        })


class RegularFeatureGroupForFilterTest(FeatureGroup):
    """A regular FeatureGroup that does NOT override final_filters().

    Filters should still be applied as a final step by the framework.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({
            cls.get_class_name(): [10, 20, 30, 40],
            "status": ["active", "inactive", "active", "inactive"],
        })


@PARALLELIZATION_MODES_SYNC_THREADING
class TestFeatureGroupFinalFilters:
    """Tests that FeatureGroup.final_filters() controls inline vs. final filter application."""

    def test_inline_filter_without_custom_compute_framework(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        """A FeatureGroup with final_filters() returning False should handle filters inline.

        InlineMaskViaFeatureGroup overrides final_filters() -> False, meaning the
        framework should NOT apply filters as a final step. Instead, the FeatureGroup
        reads features.filters in calculate_feature() and builds a mask itself.

        With filter status=="active", the inline mask zeros out inactive rows, sums
        by region, and broadcasts back: [10, 10, 30, 30].

        The status column should NOT be eliminated by final filtering because the
        FeatureGroup declared final_filters() -> False.
        """
        feature_name = "InlineMaskViaFeatureGroup"

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

    def test_regular_feature_group_still_uses_final_filters(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        """A FeatureGroup without final_filters() override should still have rows eliminated.

        RegularFeatureGroupForFilterTest does NOT override final_filters(), so the
        framework's default behavior applies: the PyArrowFilterEngine (which returns
        final_filters() -> True) filters rows after calculate_feature().

        With filter status=="active", only the two "active" rows should remain.
        """
        feature_name = "RegularFeatureGroupForFilterTest"

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
            # Final filtering should have eliminated the "inactive" rows
            assert data[feature_name] == [10, 30]

    def test_mixed_final_and_inline_filters_in_same_run(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        """Both inline and final filter modes should work correctly in the same run_all() call.

        When requesting InlineMaskViaFeatureGroup (final_filters() -> False) and
        RegularFeatureGroupForFilterTest (default final_filters() -> True) together
        with a single GlobalFilter status=="active":

        - InlineMaskViaFeatureGroup handles the filter inline via masking, preserving
          all 4 rows with masked aggregation: [10, 10, 30, 30].
        - RegularFeatureGroupForFilterTest relies on the framework's final filtering,
          which eliminates non-matching rows: [10, 30].

        This proves both filter modes operate independently and correctly within the
        same API call.
        """
        inline_feature_name = "InlineMaskViaFeatureGroup"
        final_feature_name = "RegularFeatureGroupForFilterTest"

        features = Features([
            Feature(name=inline_feature_name, initial_requested_data=True),
            Feature(name=final_feature_name, initial_requested_data=True),
        ])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        results_by_feature = {}
        for res in result.results:
            data = res.to_pydict()
            if inline_feature_name in data:
                results_by_feature[inline_feature_name] = data[inline_feature_name]
            if final_feature_name in data:
                results_by_feature[final_feature_name] = data[final_feature_name]

        # Inline FeatureGroup preserves all 4 rows with masked aggregation
        assert results_by_feature[inline_feature_name] == [10, 10, 30, 30]
        # Regular FeatureGroup has non-matching rows eliminated by final filtering
        assert results_by_feature[final_feature_name] == [10, 30]
