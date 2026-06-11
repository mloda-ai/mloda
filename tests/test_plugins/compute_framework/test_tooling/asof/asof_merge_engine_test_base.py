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

import pytest

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

    # ==================== OVERRIDABLE HOOKS ====================

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """Exception types expected when opt-in time-column coercion hits an unparseable value.

        Coercion fails HARD: engines raise their native backend error for values that are
        not ISO-8601, so subclasses override this where the backend error is not a plain
        ValueError (e.g. polars ComputeError, duckdb.Error).
        """
        return (ValueError,)

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

    def _normalize_time_value(self, value: Any) -> Any:
        """Normalize NULL-like time values (None, NaN, NaT) to Python None.

        Extends _normalize_value with a self-inequality check: NaT (pandas/numpy)
        compares unequal to itself, just like NaN, so it counts as null here.
        """
        normalized = self._normalize_value(value)
        if normalized is None:
            return None
        if normalized != normalized:
            return None
        return normalized

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

    def test_string_time_column_raises_clear_error(self) -> None:
        """A non-ordered (string/object) time column must raise a clear ValueError.

        Forwarding ISO-date strings straight to the backend yields a cryptic error
        (e.g. pandas' "both sides must have numeric dtype"). The guard instead raises a
        uniform ValueError that names the offending time column ('t') and states the
        datetime/numeric/timedelta requirement, consistently across every engine that
        inherits this base class.
        """
        left_data = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-01", "lv": 1}])
        right_data = self.convert_dict_to_framework([{"k": "a", "t": "2025-05-01", "rv": 1.0}])

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        # Require the quoted column name 't' AND the requirement wording. A bare
        # ``t.*(datetime|numeric)`` regex would be too loose: pandas' low-level MergeError
        # ("...both sides must have numeric dtype") is itself a ValueError and would satisfy
        # it. The clear guard message names the column, so we match on "'t'".
        with pytest.raises(ValueError, match=r"'t'.*(datetime|numeric)"):
            engine.merge_asof(left_data, right_data, Index(("k",)), Index(("k",)), cfg)

    def test_coerce_time_columns_opt_in_iso_strings(self) -> None:
        """With coerce_time_columns=True, ISO-8601 string time columns are coerced and the
        as-of join SUCCEEDS instead of raising the #509 guard ValueError.

        The coerced 't' column's resulting type is engine-specific (datetime, timestamp,
        julianday float, ...), so we assert only on lv/rv values, not on the time type.
        """
        left_data = self.convert_dict_to_framework(
            [{"k": "a", "t": "2025-06-03", "lv": 1}, {"k": "a", "t": "2025-06-05", "lv": 2}]
        )
        right_data = self.convert_dict_to_framework(
            [{"k": "a", "t": "2025-06-01", "rv": 1.0}, {"k": "a", "t": "2025-06-04", "rv": 2.0}]
        )

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        result = engine.merge_asof(left_data, right_data, Index(("k",)), Index(("k",)), cfg)
        result_dicts = self.convert_framework_to_dict(result)
        result_dicts.sort(key=lambda row: row["lv"])

        assert [row["lv"] for row in result_dicts] == [1, 2], "Every left row must survive the as-of join"
        actual_rv = [self._normalize_value(row["rv"]) for row in result_dicts]
        assert actual_rv == [1.0, 2.0], f"rv mismatch after coercion: got {actual_rv}, expected [1.0, 2.0]"

    def test_coerce_unparseable_string_raises(self) -> None:
        """Coercion fails HARD: a value that is not ISO-8601 must raise, never silently
        become null/NaT and produce a wrong match.

        Lazy engines (polars lazy, duckdb) surface the backend error only at
        materialization, so the conversion back to dicts is inside the raises block too.
        """
        left_data = self.convert_dict_to_framework(
            [{"k": "a", "t": "2025-06-03", "lv": 1}, {"k": "a", "t": "2025-06-05", "lv": 2}]
        )
        right_data = self.convert_dict_to_framework(
            [{"k": "a", "t": "2025-06-01", "rv": 1.0}, {"k": "a", "t": "not-a-date", "rv": 9.0}]
        )

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        with pytest.raises(self.coercion_error_types()):
            result = engine.merge_asof(left_data, right_data, Index(("k",)), Index(("k",)), cfg)
            self.convert_framework_to_dict(result)

    def test_coerce_mixed_format_raises(self) -> None:
        """A non-ISO format ('06/03/2025') mixed into otherwise ISO strings must raise.

        Same lazy-engine rule as test_coerce_unparseable_string_raises: the conversion
        back to dicts stays inside the raises block.
        """
        left_data = self.convert_dict_to_framework(
            [{"k": "a", "t": "2025-06-03", "lv": 1}, {"k": "a", "t": "2025-06-05", "lv": 2}]
        )
        right_data = self.convert_dict_to_framework(
            [{"k": "a", "t": "2025-06-01", "rv": 1.0}, {"k": "a", "t": "06/03/2025", "rv": 2.0}]
        )

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        with pytest.raises(self.coercion_error_types()):
            result = engine.merge_asof(left_data, right_data, Index(("k",)), Index(("k",)), cfg)
            self.convert_framework_to_dict(result)

    def test_coerce_preserves_nulls_and_transforms_strings(self) -> None:
        """validate_asof_time_columns returns the (left, right) pair; with coercion on, ISO
        strings become non-string ordered values while pre-existing nulls stay null.

        Calls the hook DIRECTLY (no merge) and unpacks the returned tuple, which also pins
        the new return contract (the old implementation returned None).
        """
        rows: list[dict[str, Any]] = [{"k": "a", "t": "2025-06-01", "lv": 1}, {"k": "a", "t": None, "lv": 2}]
        left_data = self.convert_dict_to_framework(rows)
        right_data = self.convert_dict_to_framework(rows)

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            coerce_time_columns=True,
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        left_out, right_out = engine.validate_asof_time_columns(left_data, right_data, cfg)
        assert right_out is not None

        left_dicts = self.convert_framework_to_dict(left_out)
        by_lv = {row["lv"]: row for row in left_dicts}

        coerced = by_lv[1]["t"]
        assert self._normalize_time_value(coerced) is not None, "Coerced ISO string must not become null"
        assert not isinstance(coerced, str), f"'t' must no longer be a string after coercion, got {coerced!r}"

        assert self._normalize_time_value(by_lv[2]["t"]) is None, "Pre-existing null must stay null after coercion"

    def test_coerce_flag_noop_on_ordered_columns(self) -> None:
        """coerce_time_columns=True is a no-op when the time columns are already ordered
        (numeric here): the join succeeds exactly as without the flag."""
        left_data = self.convert_dict_to_framework([{"k": 1, "t": 10, "lv": 100}])
        right_data = self.convert_dict_to_framework([{"k": 1, "t": 5, "rv": 1.0}])

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        result = engine.merge_asof(left_data, right_data, Index(("k",)), Index(("k",)), cfg)
        result_dicts = self.convert_framework_to_dict(result)

        assert len(result_dicts) == 1
        actual_rv = [self._normalize_value(row["rv"]) for row in result_dicts]
        assert actual_rv == [1.0], f"rv mismatch with no-op coercion flag: got {actual_rv}, expected [1.0]"
