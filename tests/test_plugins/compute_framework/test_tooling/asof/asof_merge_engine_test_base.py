"""
Base test class for ASOF merge engine testing.

This module provides a reusable ABC that implements common test logic
for point-in-time (as-of) merge operations across all compute frameworks.
"""

import math
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Optional

from mloda.user import Index
from mloda.provider import BaseMergeEngine
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

from .asof_scenarios import ASOF_SCENARIOS
from ..multi_index.test_data_converter import DataConverter


class AsofMergeEngineTestBase(ABC):
    """
    Base class for ASOF merge engine tests.

    Subclasses must implement:
    - merge_engine_class(): Return the MergeEngine class to test
    - framework_type(): Return the framework's expected data type
    - get_connection(): Return framework connection object (or None)

    This base class provides:
    - One concrete test method per canonical ASOF scenario
    - Framework-agnostic test logic using shared scenarios
    - Automatic conversion between dict and framework formats
    - Common assertion logic including null normalization
    """

    def setup_method(self) -> None:
        """Initialize the test base with a data converter before each test method."""
        self.converter = DataConverter()

    # ==================== ABSTRACT METHODS (must implement) ====================

    @classmethod
    @abstractmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the merge engine class for this framework."""
        pass

    @classmethod
    @abstractmethod
    def framework_type(cls) -> type[Any]:
        """Return the framework's expected data type (e.g., pd.DataFrame, DuckdbRelation)."""
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return framework connection object, or None if not needed."""
        pass

    # ==================== HELPER METHODS ====================

    def convert_dict_to_framework(self, data: list[dict[str, Any]]) -> Any:
        """Convert List[Dict] to this framework's native format using the converter."""
        return self.converter.to_framework(data, self.framework_type(), self.get_connection())

    def convert_framework_to_dict(self, data: Any) -> list[dict[str, Any]]:
        """Convert framework data back to List[Dict] for assertions."""
        return self.converter.from_framework(data, self.framework_type())

    def _normalize_value(self, value: Any) -> Any:
        """Normalize NaN/NULL to Python None for cross-backend comparison."""
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    def _run_asof_test(self, scenario_key: str) -> None:
        """
        Run an ASOF merge test using a predefined scenario.

        Converts left/right dicts to the framework format, builds AsOfJoinConfig
        from cfg_kwargs + time columns, runs merge_asof, converts result back to
        list[dict], sorts by sort_columns, and asserts row count, column presence,
        and rv column values (with null normalization).
        """
        scenario = ASOF_SCENARIOS[scenario_key]

        left_data = self.convert_dict_to_framework(scenario["left"])
        right_data = self.convert_dict_to_framework(scenario["right"])

        left_index = Index(scenario["left_index"])
        right_index = Index(scenario["right_index"])

        cfg = AsOfJoinConfig(
            left_time_column=scenario["left_time_column"],
            right_time_column=scenario["right_time_column"],
            **scenario["cfg_kwargs"],
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        result = engine.merge_asof(left_data, right_data, left_index, right_index, cfg)

        result_dicts = self.convert_framework_to_dict(result)

        # Sort deterministically before asserting
        sort_columns = scenario["sort_columns"]
        result_dicts.sort(key=lambda row: tuple((row.get(c) is None, row.get(c)) for c in sort_columns))

        self._assert_row_count(result_dicts, scenario["expected_rows"])
        self._assert_left_rows_preserved(scenario["left"], result_dicts)
        if scenario.get("exact_columns"):
            self._assert_exact_columns(result_dicts, scenario["expected_columns"])
        else:
            self._assert_columns_exist(result_dicts, scenario["expected_columns"])
        self._assert_rv_values(result_dicts, scenario["expected_rv"])

    def _assert_row_count(self, result: list[dict[str, Any]], expected: int) -> None:
        """Assert that result has expected number of rows."""
        actual = len(result)
        assert actual == expected, f"Expected {expected} rows, got {actual}"

    def _assert_left_rows_preserved(self, left: list[dict[str, Any]], result: list[dict[str, Any]]) -> None:
        """Assert ASOF is a LEFT join: every left row survives exactly once.

        The multiset (Counter) of the left value column ``lv`` must be identical in the
        scenario's left input and the merged output, after null normalization.
        """
        expected = Counter(self._normalize_value(row.get("lv")) for row in left)
        actual = Counter(self._normalize_value(row.get("lv")) for row in result)
        assert actual == expected, (
            f"Left rows not preserved: lv multiset mismatch. expected {dict(expected)}, got {dict(actual)}"
        )

    def _assert_columns_exist(self, result: list[dict[str, Any]], expected_columns: list[str]) -> None:
        """Assert that all expected columns exist in result."""
        if not result:
            return
        actual_columns = set(result[0].keys())
        for col in expected_columns:
            assert col in actual_columns, f"Expected column '{col}' not found. Available: {actual_columns}"

    def _assert_exact_columns(self, result: list[dict[str, Any]], expected_columns: list[str]) -> None:
        """Assert that the result's column set equals expected_columns exactly (order-independent)."""
        if not result:
            return
        actual_columns = set(result[0].keys())
        assert actual_columns == set(expected_columns), (
            f"Column set mismatch: got {sorted(actual_columns)}, expected {sorted(expected_columns)}"
        )

    def _assert_rv_values(self, result: list[dict[str, Any]], expected_rv: list[Any]) -> None:
        """Assert that the 'rv' column matches expected values after null normalization."""
        actual_rv = [self._normalize_value(row.get("rv")) for row in result]
        normalized_expected = [self._normalize_value(v) for v in expected_rv]
        assert actual_rv == normalized_expected, f"rv mismatch: got {actual_rv}, expected {normalized_expected}"

    # ==================== TEST METHODS (one per scenario) ====================

    def test_backward_single_key(self) -> None:
        """Vector A: backward, single by-key. Right rows shuffled to prove internal sorting."""
        self._run_asof_test("backward_single_key")

    def test_forward_single_key(self) -> None:
        """Vector B: forward; row (k=1, t=20) has no right_time >= 20 -> null."""
        self._run_asof_test("forward_single_key")

    def test_allow_exact_matches_true(self) -> None:
        """Vector C: backward + allow_exact_matches=True -> exact-time row matched (rv=99)."""
        self._run_asof_test("exact_matches_true")

    def test_allow_exact_matches_false(self) -> None:
        """Vector C: backward + allow_exact_matches=False -> exact excluded, prior row (rv=1)."""
        self._run_asof_test("exact_matches_false")

    def test_tolerance_numeric(self) -> None:
        """Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null."""
        self._run_asof_test("tolerance_numeric")

    def test_tolerance_none(self) -> None:
        """Vector D: backward, tolerance=None -> both rows match (rv=7, 7)."""
        self._run_asof_test("tolerance_none")

    def test_multi_by_key(self) -> None:
        """Vector E: multi by-key (k1, k2), backward."""
        self._run_asof_test("multi_by_key")

    def test_differing_names(self) -> None:
        """Vector G: differing by-key and time-column names; both by-key columns survive."""
        self._run_asof_test("differing_names")

    def test_no_sortedness_warning_with_by_groups(self) -> None:
        """An ASOF join with `by` groups must not emit a 'Sortedness' warning.

        Polars' ``join_asof`` warns ("Sortedness of columns cannot be checked when
        'by' groups provided") whenever ``by_left``/``by_right`` are passed; the fix
        is to pass ``check_sortedness=False``. The warning is framework-specific
        (only polars emits it), so we record all warnings and assert none mention
        "Sortedness" rather than turning every UserWarning into an error -- this keeps
        the test framework-safe for pandas/duckdb that inherit this base class.

        For the lazy polars engine the warning is deferred until the LazyFrame is
        collected, so the capture must wrap the whole scenario run (including the
        conversion/collect inside ``_run_asof_test``), not just the merge_asof call.
        """
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            self._run_asof_test("multi_by_key")

        offending = [str(w.message) for w in recorded if "Sortedness" in str(w.message)]
        assert not offending, f"Unexpected 'Sortedness' warning(s) emitted: {offending}"
