"""Tests that a FeatureGroup can control whether filters are applied inline vs. final.

``FeatureGroup.final_filters()`` controls post-calculation row elimination only.
``features.filters`` is always available inside ``calculate_feature()`` regardless
of what ``final_filters()`` returns, so inline filter reading and row elimination
are independent concerns.

The matrix of (FG final_filters, Engine final_filters, reads inline) combinations:

  FG=False, Engine=True,  inline=yes -- test_inline_filter_without_custom_compute_framework
  FG=None,  Engine=True,  inline=no  -- test_regular_feature_group_still_uses_final_filters
  FG=False, Engine=False, inline=yes -- test_fg_skip_with_inline_engine
  FG=True,  Engine=False, inline=no  -- test_fg_force_final_overrides_inline_engine
  FG=True,  Engine=True,  inline=no  -- test_fg_force_final_with_final_engine
  FG=True,  Engine=True,  inline=yes -- test_inline_mask_with_final_elimination

Validation (filter column must be present when row elimination applies):
  FG=True,  drops filter column       -- test_dropped_filter_column_raises_error
  FG=None,  drops filter column       -- test_default_fg_drops_filter_column_raises_error
"""

from typing import Any, Optional

import pytest
import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import BaseFilterEngine, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.provider import BaseInputData
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine
from mloda.user import Feature, Features, GlobalFilter, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


class InlineFilterEngine(PyArrowFilterEngine):
    """Filter engine that returns final_filters()=False, simulating Iceberg-like behavior."""

    @classmethod
    def final_filters(cls) -> bool:
        return False


class PyArrowTableInlineEngine(PyArrowTable):
    """PyArrowTable variant with an inline (non-final) filter engine."""

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        return InlineFilterEngine


class InlineMaskViaFeatureGroup(FeatureGroup):
    """A FeatureGroup that handles filters inline instead of relying on final filtering.

    Uses the STANDARD PyArrowTable compute framework. No custom FilterEngine or
    ComputeFramework subclass is needed.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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

        return pa.table(
            {
                cls.get_class_name(): result,
                "status": status,
            }
        )


class RegularFeatureGroupForFilterTest(FeatureGroup):
    """A regular FeatureGroup that does NOT override final_filters().

    Filters should still be applied as a final step by the framework.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "status": ["active", "inactive", "active", "inactive"],
            }
        )


class InlineMaskOnInlineEngine(InlineMaskViaFeatureGroup):
    """FG says False, Engine says False. Both agree: skip final filtering.

    Reuses InlineMaskViaFeatureGroup's calculate_feature (inline masking) but
    runs on PyArrowTableInlineEngine whose filter engine also returns False.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTableInlineEngine}

    @classmethod
    def final_filters(cls) -> bool:
        return False


class ForceFinalOnInlineEngine(FeatureGroup):
    """FG says True, Engine says False. FG overrides: force row elimination.

    The filter engine on PyArrowTableInlineEngine returns final_filters()=False,
    but this FeatureGroup explicitly returns True, so the framework should apply
    final filtering and eliminate non-matching rows.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTableInlineEngine}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "status": ["active", "inactive", "active", "inactive"],
            }
        )


class ForceFinalOnFinalEngine(FeatureGroup):
    """FG says True, Engine says True. Both agree: apply final filtering.

    Both the FeatureGroup and the PyArrowTable filter engine agree that final
    filtering should be applied. Non-matching rows should be eliminated.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "status": ["active", "inactive", "active", "inactive"],
            }
        )


class InlineMaskWithFinalElimination(FeatureGroup):
    """FG reads filters inline for masking AND returns True for row elimination.

    Proves that inline filter reading and final row elimination are independent.
    The FeatureGroup reads ``features.filters`` during ``calculate_feature()``
    to mask non-matching values before aggregation, then returns
    ``final_filters() = True`` so the framework also eliminates non-matching
    rows afterward.

    Data layout (3 active, 1 inactive across two regions):
        region: [A, A, A, B], status: [active, active, inactive, active]
        value:  [10, 5, 20, 30]

    With filter status=="active":
        Masked values:  [10, 5, None, 30]
        Region sums:    A=15, B=30
        Broadcast:      [15, 15, 15, 30]
        Row elimination: removes the inactive row -> [15, 15, 30]

    Without inline masking (raw + elimination only):
        Would be [10, 5, 30] (raw active values, no aggregation).

    The expected [15, 15, 30] is distinguishable from both the inline-only
    result [15, 15, 15, 30] and the elimination-only result [10, 5, 30].
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        region = pa.array(["A", "A", "A", "B"])
        status = pa.array(["active", "active", "inactive", "active"])
        value = pa.array([10, 5, 20, 30])

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

        return pa.table(
            {
                cls.get_class_name(): result,
                "status": status,
            }
        )


class DropsFilterColumnFeatureGroup(FeatureGroup):
    """FG returns final_filters()=True but drops the filter column from its output.

    This violates the overlap contract: row elimination needs the filter column
    to decide which rows to remove. The framework should raise a clear error.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                # "status" intentionally omitted
            }
        )


class DefaultFGDropsFilterColumn(FeatureGroup):
    """FG does NOT override final_filters() (defaults to None) but drops the filter column.

    The engine (PyArrowFilterEngine) returns True, so the framework applies
    row elimination. The missing column should be caught by the validation.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                # "status" intentionally omitted
            }
        )


class MutatesFilterColumnToInt(FeatureGroup):
    """FG returns final_filters()=True but changes filter column from string to int.

    The filter expects status=="active" (string comparison), but the FeatureGroup
    maps the column to integers. The dtype mismatch should be detected.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "status": [1, 0, 1, 0],
            }
        )


class MutatesFilterColumnToString(FeatureGroup):
    """FG returns final_filters()=True but changes filter column from numeric to string.

    The filter expects price >= 10 (numeric comparison), but the FeatureGroup
    maps the column to strings. The dtype mismatch should be detected.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "price"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "price": ["ten", "twenty", "thirty", "forty"],
            }
        )


class DefaultFGMutatesFilterColumnToInt(FeatureGroup):
    """FG does NOT override final_filters() but mutates filter column dtype.

    The engine fallback path applies row elimination. The dtype mismatch should
    be caught on this path too.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "status": [1, 0, 1, 0],
            }
        )


class CompatibleDtypeFeatureGroup(FeatureGroup):
    """FG returns final_filters()=True with compatible dtypes.

    The filter expects status=="active" (string) and the column is also string.
    This should pass without error.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                cls.get_class_name(): [10, 20, 30, 40],
                "status": ["active", "inactive", "active", "inactive"],
            }
        )


@PARALLELIZATION_MODES_SYNC_THREADING
class TestFeatureGroupFinalFilters:
    """Tests that FeatureGroup.final_filters() controls inline vs. final filter application."""

    def test_inline_filter_without_custom_compute_framework(
        self, modes: set[ParallelizationMode], flight_server: Any
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
        self, modes: set[ParallelizationMode], flight_server: Any
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
        self, modes: set[ParallelizationMode], flight_server: Any
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

        features = Features(
            [
                Feature(name=inline_feature_name, initial_requested_data=True),
                Feature(name=final_feature_name, initial_requested_data=True),
            ]
        )

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

    def test_fg_skip_with_inline_engine(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=False, Engine=False. Both agree to skip final filtering. Rows preserved.

        InlineMaskOnInlineEngine inherits inline masking from InlineMaskViaFeatureGroup
        and runs on PyArrowTableInlineEngine (whose filter engine also returns
        final_filters()=False). Since both the FeatureGroup and the engine agree that
        final filtering should be skipped, the inline mask logic preserves all 4 rows
        with masked aggregation: [10, 10, 30, 30].
        """
        feature_name = "InlineMaskOnInlineEngine"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTableInlineEngine},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            # Both FG and engine say skip: inline masking preserves all 4 rows
            assert data[feature_name] == [10, 10, 30, 30]

    def test_fg_force_final_overrides_inline_engine(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=True, Engine=False. FG overrides engine. Rows eliminated.

        ForceFinalOnInlineEngine explicitly returns final_filters()=True, while
        PyArrowTableInlineEngine's filter engine returns False. The FeatureGroup's
        decision should override the engine: final filtering is applied and
        non-matching rows are eliminated, leaving only [10, 30].
        """
        feature_name = "ForceFinalOnInlineEngine"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTableInlineEngine},
            parallelization_modes=modes,
            flight_server=flight_server,
            global_filter=global_filter,
        )

        for res in result.results:
            data = res.to_pydict()
            # FG overrides engine: rows eliminated despite engine saying skip
            assert data[feature_name] == [10, 30]

    def test_fg_force_final_with_final_engine(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=True, Engine=True. Both agree to apply final filtering. Rows eliminated.

        ForceFinalOnFinalEngine explicitly returns final_filters()=True and runs on
        standard PyArrowTable (whose PyArrowFilterEngine also returns True). Since
        both agree, final filtering is applied and non-matching rows are eliminated,
        leaving only [10, 30].
        """
        feature_name = "ForceFinalOnFinalEngine"

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
            # Both FG and engine agree: rows eliminated
            assert data[feature_name] == [10, 30]

    def test_inline_mask_with_final_elimination(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=True, Engine=True, reads inline. Masking AND elimination are independent.

        InlineMaskWithFinalElimination reads features.filters during calculation
        to mask inactive values before aggregation, AND returns final_filters()=True
        so the framework also eliminates non-matching rows afterward.

        Data: region=[A,A,A,B], status=[active,active,inactive,active], value=[10,5,20,30]
        After inline mask + region sum + broadcast: [15, 15, 15, 30]
        After row elimination (remove inactive):   [15, 15, 30]

        This result is distinguishable from:
        - inline only (no elimination):  [15, 15, 15, 30] (4 rows)
        - elimination only (no masking): [10, 5, 30] (raw active values)
        """
        feature_name = "InlineMaskWithFinalElimination"

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
            # Inline masking changed values (15 not 10/5), elimination removed inactive row
            assert data[feature_name] == [15, 15, 30]

    def test_dropped_filter_column_raises_error(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=True but filter column missing from output. Framework must raise ValueError.

        DropsFilterColumnFeatureGroup returns final_filters()=True but omits the
        "status" column from its output table. The framework's validation should
        catch this and raise a clear error instead of crashing in the filter engine.
        """
        feature_name = "DropsFilterColumnFeatureGroup"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        # ValueError from _validate_filter_columns is wrapped in a generic
        # Exception by the threading/worker layer (see thread_worker.py).
        with pytest.raises(Exception, match="missing filter column.*status.*bug in the FeatureGroup"):
            MlodaTestRunner.run_api(
                features,
                compute_frameworks={PyArrowTable},
                parallelization_modes=modes,
                flight_server=flight_server,
                global_filter=global_filter,
            )

    def test_default_fg_drops_filter_column_raises_error(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """FG=None (default), Engine=True, filter column missing. Framework must raise ValueError.

        DefaultFGDropsFilterColumn does not override final_filters(), so the
        engine fallback path applies row elimination. The validation must catch
        the missing column on this path too, not only when the FG returns True.
        """
        feature_name = "DefaultFGDropsFilterColumn"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        # ValueError from _validate_filter_columns is wrapped in a generic
        # Exception by the threading/worker layer (see thread_worker.py).
        with pytest.raises(Exception, match="missing filter column.*status.*bug in the FeatureGroup"):
            MlodaTestRunner.run_api(
                features,
                compute_frameworks={PyArrowTable},
                parallelization_modes=modes,
                flight_server=flight_server,
                global_filter=global_filter,
            )

    def test_string_filter_on_int_column_raises_error(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """FG=True, string filter on int column. Framework must raise ValueError.

        MutatesFilterColumnToInt returns final_filters()=True but maps the
        "status" column from strings to integers. The filter expects
        status=="active" (string), so the dtype mismatch should be detected.
        """
        feature_name = "MutatesFilterColumnToInt"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        with pytest.raises(Exception, match="expects string comparison but column has type"):
            MlodaTestRunner.run_api(
                features,
                compute_frameworks={PyArrowTable},
                parallelization_modes=modes,
                flight_server=flight_server,
                global_filter=global_filter,
            )

    def test_numeric_filter_on_string_column_raises_error(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """FG=True, numeric filter on string column. Framework must raise ValueError.

        MutatesFilterColumnToString returns final_filters()=True but maps the
        "price" column from numbers to strings. The filter expects
        price >= 10 (numeric), so the dtype mismatch should be detected.
        """
        feature_name = "MutatesFilterColumnToString"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("price", "min", {"min": 10})

        with pytest.raises(Exception, match="expects numeric comparison but column has type"):
            MlodaTestRunner.run_api(
                features,
                compute_frameworks={PyArrowTable},
                parallelization_modes=modes,
                flight_server=flight_server,
                global_filter=global_filter,
            )

    def test_default_fg_dtype_mismatch_raises_error(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=None (default), Engine=True, dtype mismatch. Framework must raise ValueError.

        DefaultFGMutatesFilterColumnToInt does not override final_filters(),
        so the engine fallback path applies. The dtype mismatch should be caught
        on this path too, not only when the FG returns True.
        """
        feature_name = "DefaultFGMutatesFilterColumnToInt"

        features = Features([Feature(name=feature_name, initial_requested_data=True)])

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        with pytest.raises(Exception, match="expects string comparison but column has type"):
            MlodaTestRunner.run_api(
                features,
                compute_frameworks={PyArrowTable},
                parallelization_modes=modes,
                flight_server=flight_server,
                global_filter=global_filter,
            )

    def test_compatible_dtype_passes_validation(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """FG=True with compatible dtypes. No error should be raised.

        CompatibleDtypeFeatureGroup returns final_filters()=True and the
        "status" column is string, matching the filter value type. Row
        elimination should proceed normally.
        """
        feature_name = "CompatibleDtypeFeatureGroup"

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
            assert data[feature_name] == [10, 30]
