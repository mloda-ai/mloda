"""Conformance suites for BaseMergeEngine implementations (equi, multi-index and as-of joins), run through an adapter."""

from __future__ import annotations

import collections
import math
import warnings
from collections.abc import Callable
from typing import Any, TypedDict

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.compute_framework_shape import serves_asof_merge
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig, JoinSpec, JoinType, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureGroup

Rows = list[dict[str, Any]]


def make_merge_link(
    jointype: JoinType,
    left_index: Index,
    right_index: Index,
    asof_config: AsOfJoinConfig | None = None,
) -> Link:
    """Minimal Link carrying jointype, indexes and an optional asof_config."""
    return Link(
        jointype,
        JoinSpec(FeatureGroup, left_index),
        JoinSpec(FeatureGroup, right_index),
        asof_config=asof_config,
    )


class MergeScenario(TypedDict):
    description: str
    left: TypedTestData
    right: TypedTestData
    index: tuple[str, ...]
    expected_rows: int
    expected_columns: list[str]


class MergeConformanceScenario(TypedDict):
    description: str
    left: TypedTestData
    right: TypedTestData
    left_index: tuple[str, ...]
    right_index: tuple[str, ...]
    expected: dict[str, Rows]


class _AsofScenarioRequired(TypedDict):
    description: str
    left: TypedTestData
    right: TypedTestData
    left_index: tuple[str, ...]
    right_index: tuple[str, ...]
    left_time_column: str
    right_time_column: str
    cfg_kwargs: dict[str, Any]
    sort_columns: list[str]
    expected_rows: int
    expected_columns: list[str]
    expected_rv: list[Any]


class AsofScenario(_AsofScenarioRequired, total=False):
    """exact_columns: the result's column set must equal expected_columns, not merely contain it."""

    exact_columns: bool


_I = DataType.INT64
_S = DataType.STRING
_D = DataType.DOUBLE

PLAIN_LEFT = TypedTestData({"idx": (_I, [1, 3]), "col1": (_S, ["a", "b"])})
PLAIN_RIGHT = TypedTestData({"idx": (_I, [1, 2]), "col2": (_S, ["x", "z"])})
PLAIN_COLUMNS = {"idx", "col1", "col2"}

SCENARIOS: dict[str, MergeScenario] = {
    "inner_basic": {
        "description": "Basic INNER join with all rows matching on 2-column index",
        "left": TypedTestData(
            {"id": (_I, [1, 2, 3]), "category": (_S, ["A", "B", "C"]), "left_value": (_I, [10, 20, 30])}
        ),
        "right": TypedTestData(
            {"id": (_I, [1, 2, 3]), "category": (_S, ["A", "B", "C"]), "right_value": (_I, [100, 200, 300])}
        ),
        "index": ("id", "category"),
        "expected_rows": 3,
        "expected_columns": ["id", "category", "left_value", "right_value"],
    },
    "left_with_unmatched": {
        "description": "LEFT join with some unmatched rows from left side",
        "left": TypedTestData(
            {"id": (_I, [1, 2, 3, 4]), "category": (_S, ["A", "B", "C", "D"]), "left_value": (_I, [10, 20, 30, 40])}
        ),
        "right": TypedTestData(
            {"id": (_I, [1, 2, 5]), "category": (_S, ["A", "B", "E"]), "right_value": (_I, [100, 200, 500])}
        ),
        "index": ("id", "category"),
        "expected_rows": 4,
        "expected_columns": ["id", "category", "left_value", "right_value"],
    },
    "outer_both_sides": {
        "description": "OUTER join with unmatched rows on both sides",
        "left": TypedTestData(
            {"id": (_I, [1, 2, 3]), "category": (_S, ["A", "B", "C"]), "left_value": (_I, [10, 20, 30])}
        ),
        "right": TypedTestData(
            {"id": (_I, [1, 4, 5]), "category": (_S, ["A", "D", "E"]), "right_value": (_I, [100, 400, 500])}
        ),
        "index": ("id", "category"),
        "expected_rows": 5,
        "expected_columns": ["id", "category", "left_value", "right_value"],
    },
    "append_basic": {
        "description": "APPEND operation concatenating all rows",
        "left": TypedTestData({"id": (_I, [1, 2]), "category": (_S, ["A", "B"]), "value": (_I, [10, 20])}),
        "right": TypedTestData({"id": (_I, [3, 4]), "category": (_S, ["C", "D"]), "value": (_I, [30, 40])}),
        "index": ("id", "category"),
        "expected_rows": 4,
        "expected_columns": ["id", "category", "value"],
    },
    "union_with_duplicates": {
        "description": "UNION operation removing exact duplicate rows",
        "left": TypedTestData({"id": (_I, [1, 2, 3]), "category": (_S, ["A", "B", "C"]), "value": (_I, [10, 20, 30])}),
        "right": TypedTestData({"id": (_I, [1, 4, 5]), "category": (_S, ["A", "D", "E"]), "value": (_I, [10, 40, 50])}),
        "index": ("id", "category"),
        "expected_rows": 5,
        "expected_columns": ["id", "category", "value"],
    },
}

JOIN_TYPE_NAMES: tuple[str, ...] = ("INNER", "LEFT", "RIGHT", "OUTER")

MERGE_CONFORMANCE_SCENARIOS: dict[str, MergeConformanceScenario] = {
    "same_key": {
        "description": "Shared key name 'k' on both sides; left keys {1,2,3}, right keys {1,2,4}.",
        "left": TypedTestData({"k": (_I, [1, 2, 3]), "lv": (_S, ["a", "b", "c"])}),
        "right": TypedTestData({"k": (_I, [1, 2, 4]), "rv": (_S, ["x", "y", "z"])}),
        "left_index": ("k",),
        "right_index": ("k",),
        "expected": {
            "INNER": [{"k": 1, "lv": "a", "rv": "x"}, {"k": 2, "lv": "b", "rv": "y"}],
            "LEFT": [
                {"k": 1, "lv": "a", "rv": "x"},
                {"k": 2, "lv": "b", "rv": "y"},
                {"k": 3, "lv": "c", "rv": None},
            ],
            "RIGHT": [
                {"k": 1, "lv": "a", "rv": "x"},
                {"k": 2, "lv": "b", "rv": "y"},
                {"k": 4, "lv": None, "rv": "z"},
            ],
            "OUTER": [
                {"k": 1, "lv": "a", "rv": "x"},
                {"k": 2, "lv": "b", "rv": "y"},
                {"k": 3, "lv": "c", "rv": None},
                {"k": 4, "lv": None, "rv": "z"},
            ],
        },
    },
    "diff_key": {
        "description": "Differing key names 'lk' vs 'rk'; both key columns must survive in the output.",
        "left": TypedTestData({"lk": (_I, [1, 2, 3]), "lv": (_S, ["a", "b", "c"])}),
        "right": TypedTestData({"rk": (_I, [1, 2, 4]), "rv": (_S, ["x", "y", "z"])}),
        "left_index": ("lk",),
        "right_index": ("rk",),
        "expected": {
            "INNER": [
                {"lk": 1, "lv": "a", "rk": 1, "rv": "x"},
                {"lk": 2, "lv": "b", "rk": 2, "rv": "y"},
            ],
            "LEFT": [
                {"lk": 1, "lv": "a", "rk": 1, "rv": "x"},
                {"lk": 2, "lv": "b", "rk": 2, "rv": "y"},
                {"lk": 3, "lv": "c", "rk": None, "rv": None},
            ],
            "RIGHT": [
                {"lk": 1, "lv": "a", "rk": 1, "rv": "x"},
                {"lk": 2, "lv": "b", "rk": 2, "rv": "y"},
                {"lk": None, "lv": None, "rk": 4, "rv": "z"},
            ],
            "OUTER": [
                {"lk": 1, "lv": "a", "rk": 1, "rv": "x"},
                {"lk": 2, "lv": "b", "rk": 2, "rv": "y"},
                {"lk": 3, "lv": "c", "rk": None, "rv": None},
                {"lk": None, "lv": None, "rk": 4, "rv": "z"},
            ],
        },
    },
}

_ASOF_LEFT = TypedTestData({"k": (_I, [1, 1, 2]), "t": (_I, [10, 20, 15]), "lv": (_I, [100, 200, 300])})
_ASOF_RIGHT_SORTED = TypedTestData({"k": (_I, [1, 1, 2, 2]), "t": (_I, [5, 18, 5, 30]), "rv": (_I, [1, 2, 3, 4])})
_ASOF_EXACT_RIGHT = TypedTestData({"k": (_I, [1, 1]), "t": (_I, [10, 5]), "rv": (_I, [99, 1])})
_ASOF_TOL_LEFT = TypedTestData({"k": (_I, [1, 1]), "t": (_I, [10, 100]), "lv": (_I, [1, 2])})
_ASOF_TOL_RIGHT = TypedTestData({"k": (_I, [1]), "t": (_I, [8]), "rv": (_I, [7])})
_ASOF_COLUMNS = ["k", "t", "lv", "rv"]

ASOF_SCENARIOS: dict[str, AsofScenario] = {
    "backward_single_key": {
        "description": "Vector A: backward, single by-key. Right rows shuffled to prove internal sorting.",
        "left": _ASOF_LEFT,
        "right": TypedTestData({"k": (_I, [2, 1, 1, 2]), "t": (_I, [30, 5, 18, 5]), "rv": (_I, [4, 1, 2, 3])}),
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward"},
        "sort_columns": ["k", "t"],
        "expected_rows": 3,
        "expected_columns": _ASOF_COLUMNS,
        "expected_rv": [1, 2, 3],
    },
    "forward_single_key": {
        "description": "Vector B: forward; row (k=1, t=20) has no right_time >= 20 -> null.",
        "left": _ASOF_LEFT,
        "right": _ASOF_RIGHT_SORTED,
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "forward"},
        "sort_columns": ["k", "t"],
        "expected_rows": 3,
        "expected_columns": _ASOF_COLUMNS,
        "expected_rv": [2, None, 4],
    },
    "exact_matches_true": {
        "description": "Vector C: backward + allow_exact_matches=True -> exact-time row matched (rv=99).",
        "left": TypedTestData({"k": (_I, [1]), "t": (_I, [10]), "lv": (_I, [100])}),
        "right": _ASOF_EXACT_RIGHT,
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "allow_exact_matches": True},
        "sort_columns": ["k", "t"],
        "expected_rows": 1,
        "expected_columns": _ASOF_COLUMNS,
        "expected_rv": [99],
    },
    "exact_matches_false": {
        "description": "Vector C: backward + allow_exact_matches=False -> exact excluded, prior row (rv=1).",
        "left": TypedTestData({"k": (_I, [1]), "t": (_I, [10]), "lv": (_I, [100])}),
        "right": _ASOF_EXACT_RIGHT,
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "allow_exact_matches": False},
        "sort_columns": ["k", "t"],
        "expected_rows": 1,
        "expected_columns": _ASOF_COLUMNS,
        "expected_rv": [1],
    },
    "tolerance_numeric": {
        "description": "Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null.",
        "left": _ASOF_TOL_LEFT,
        "right": _ASOF_TOL_RIGHT,
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "tolerance": 5},
        "sort_columns": ["k", "t"],
        "expected_rows": 2,
        "expected_columns": _ASOF_COLUMNS,
        "expected_rv": [7, None],
    },
    "tolerance_none": {
        "description": "Vector D: backward, tolerance=None -> both rows match (rv=7, rv=7).",
        "left": _ASOF_TOL_LEFT,
        "right": _ASOF_TOL_RIGHT,
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "tolerance": None},
        "sort_columns": ["k", "t"],
        "expected_rows": 2,
        "expected_columns": _ASOF_COLUMNS,
        "expected_rv": [7, 7],
    },
    "multi_by_key": {
        "description": "Vector E: multi by-key (k1, k2), backward.",
        "left": TypedTestData({"k1": (_I, [1, 1]), "k2": (_S, ["a", "b"]), "t": (_I, [10, 10]), "lv": (_I, [1, 2])}),
        "right": TypedTestData({"k1": (_I, [1, 1]), "k2": (_S, ["a", "b"]), "t": (_I, [5, 5]), "rv": (_I, [10, 20])}),
        "left_index": ("k1", "k2"),
        "right_index": ("k1", "k2"),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward"},
        "sort_columns": ["k1", "k2"],
        "expected_rows": 2,
        "expected_columns": ["k1", "k2", "t", "lv", "rv"],
        "expected_rv": [10, 20],
    },
    "differing_names": {
        "description": ("Differing by-key and time-column names, backward; both by-key columns must survive."),
        "left": TypedTestData({"lk": (_I, [1, 1, 2]), "lt": (_I, [10, 20, 15]), "lv": (_I, [100, 200, 300])}),
        "right": TypedTestData({"rk": (_I, [1, 1, 2, 2]), "rt": (_I, [5, 18, 5, 30]), "rv": (_I, [1, 2, 3, 4])}),
        "left_index": ("lk",),
        "right_index": ("rk",),
        "left_time_column": "lt",
        "right_time_column": "rt",
        "cfg_kwargs": {"direction": "backward"},
        "sort_columns": ["lk", "lt"],
        "expected_rows": 3,
        "expected_columns": ["lk", "lt", "lv", "rk", "rt", "rv"],
        "expected_rv": [1, 2, 3],
        "exact_columns": True,
    },
}


def _norm(value: Any) -> Any:
    """None stays None; float NaN collapses to None so nullable-int widening never matters (numeric/string data only)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _multiset(rows: Rows) -> collections.Counter[frozenset[tuple[str, Any]]]:
    """Order- and column-order-independent, null-normalized bag of rows (1 == 1.0)."""
    return collections.Counter(frozenset((col, _norm(val)) for col, val in row.items()) for row in rows)


def _has_null(rows: Rows, column: str) -> bool:
    return any(_norm(row.get(column)) is None for row in rows)


class MergeEngineSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the host's merge engine."""

    aspect = "merge"

    @pytest.fixture
    def merge_engine(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> Any:
        return framework.merge_engine()(adapter.connection)

    def _plain(self, adapter: FrameworkAdapter, merge_engine: Any, join_type: JoinType, expected_rows: int) -> None:
        index = Index(("idx",))
        left = self.frame(adapter, "plain_left", PLAIN_LEFT)
        right = self.frame(adapter, "plain_right", PLAIN_RIGHT)
        rows = self.rows(adapter, merge_engine.merge(left, right, make_merge_link(join_type, index, index)))
        assert len(rows) == expected_rows, f"Expected {expected_rows} rows, got {len(rows)}"
        assert set(rows[0]) == PLAIN_COLUMNS, f"Unexpected columns {sorted(rows[0])}"

    def test_merge_inner(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        self._plain(adapter, merge_engine, JoinType.INNER, 1)

    def test_merge_left(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        self._plain(adapter, merge_engine, JoinType.LEFT, 2)

    def test_merge_right(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        self._plain(adapter, merge_engine, JoinType.RIGHT, 2)

    def test_merge_full_outer(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        self._plain(adapter, merge_engine, JoinType.OUTER, 3)

    def test_merge_append(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        self._plain(adapter, merge_engine, JoinType.APPEND, 4)

    def test_merge_union(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        self._plain(adapter, merge_engine, JoinType.UNION, 4)

    def _run_merge_test(
        self,
        adapter: FrameworkAdapter,
        merge_engine: Any,
        scenario_key: str,
        merge_method: str,
        additional_assertions: Callable[[Rows, MergeScenario], None],
    ) -> None:
        scenario = SCENARIOS[scenario_key]
        left = self.frame(adapter, f"{scenario_key}_left", scenario["left"])
        right = self.frame(adapter, f"{scenario_key}_right", scenario["right"])
        index = Index(scenario["index"])

        rows = self.rows(adapter, getattr(merge_engine, merge_method)(left, right, index, index))

        expected = scenario["expected_rows"]
        assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"
        actual_columns = set(rows[0]) if rows else set()
        for col in scenario["expected_columns"]:
            assert col in actual_columns, f"Expected column '{col}' not found in result. Available: {actual_columns}"
        additional_assertions(rows, scenario)

    def test_merge_inner_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        def checks(rows: Rows, scenario: MergeScenario) -> None:
            for row in rows:
                assert row.get("left_value") is not None, "INNER join should not have null left values"
                assert row.get("right_value") is not None, "INNER join should not have null right values"

        self._run_merge_test(adapter, merge_engine, "inner_basic", "merge_inner", checks)

    def test_merge_left_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        def checks(rows: Rows, scenario: MergeScenario) -> None:
            for row in rows:
                assert row.get("left_value") is not None, "LEFT join should have all left values"
            assert _has_null(rows, "right_value"), "LEFT join should have some null right values"

        self._run_merge_test(adapter, merge_engine, "left_with_unmatched", "merge_left", checks)

    def test_merge_full_outer_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        def checks(rows: Rows, scenario: MergeScenario) -> None:
            assert _has_null(rows, "left_value"), "OUTER join should have some null left values"
            assert _has_null(rows, "right_value"), "OUTER join should have some null right values"

        self._run_merge_test(adapter, merge_engine, "outer_both_sides", "merge_full_outer", checks)

    def test_merge_append_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        def checks(rows: Rows, scenario: MergeScenario) -> None:
            for row in rows:
                assert row.get("value") is not None, "APPEND should not have null values"

        self._run_merge_test(adapter, merge_engine, "append_basic", "merge_append", checks)

    def test_merge_union_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        def checks(rows: Rows, scenario: MergeScenario) -> None:
            for row in rows:
                assert row.get("value") is not None, "UNION should not have null values"
            total = scenario["left"].row_count + scenario["right"].row_count
            assert len(rows) < total, "UNION should remove duplicates"

        self._run_merge_test(adapter, merge_engine, "union_with_duplicates", "merge_union", checks)

    def test_mismatched_index_lengths_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        left = self.frame(adapter, "mismatch_left", TypedTestData({"a": (_I, [1]), "b": (_I, [2]), "val": (_I, [10])}))
        right = self.frame(adapter, "mismatch_right", TypedTestData({"x": (_I, [1]), "val": (_I, [20])}))

        with pytest.raises(ValueError):
            merge_engine.merge(left, right, make_merge_link(JoinType.INNER, Index(("a", "b")), Index(("x",))))

    @pytest.mark.parametrize("scenario_key", list(MERGE_CONFORMANCE_SCENARIOS))
    @pytest.mark.parametrize("join_name", JOIN_TYPE_NAMES)
    def test_merge_conformance(
        self, adapter: FrameworkAdapter, merge_engine: Any, scenario_key: str, join_name: str
    ) -> None:
        """Single-key equi-joins give one canonical result, compared as a null-normalized bag."""
        scenario = MERGE_CONFORMANCE_SCENARIOS[scenario_key]
        left = self.frame(adapter, f"{scenario_key}_left", scenario["left"])
        right = self.frame(adapter, f"{scenario_key}_right", scenario["right"])
        link = make_merge_link(JoinType[join_name], Index(scenario["left_index"]), Index(scenario["right_index"]))

        actual = self.rows(adapter, merge_engine.merge(left, right, link))

        expected = scenario["expected"][join_name]
        assert _multiset(actual) == _multiset(expected), (
            f"Merge conformance mismatch [{scenario_key} / {join_name}]:\n  expected={expected}\n  actual={actual}"
        )


class AsofMergeEngineSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the host's as-of join; hosts without merge_asof are not served."""

    aspect = "asof merge"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return serves_asof_merge(framework)

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """Exception types raised when opt-in time-column coercion hits a value that is not ISO-8601."""
        return (ValueError,)

    @pytest.fixture
    def merge_engine(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> Any:
        return framework.merge_engine()(adapter.connection)

    def _normalize_time_value(self, value: Any) -> Any:
        """Normalize None, NaN and NaT (unequal to itself) to None."""
        normalized = _norm(value)
        if normalized is None:
            return None
        if normalized != normalized:
            return None
        return normalized

    def _run_asof_test(self, adapter: FrameworkAdapter, merge_engine: Any, scenario_key: str) -> None:
        scenario = ASOF_SCENARIOS[scenario_key]
        left = self.frame(adapter, f"{scenario_key}_left", scenario["left"])
        right = self.frame(adapter, f"{scenario_key}_right", scenario["right"])
        cfg = AsOfJoinConfig(
            left_time_column=scenario["left_time_column"],
            right_time_column=scenario["right_time_column"],
            **scenario["cfg_kwargs"],
        )

        result = merge_engine.merge_asof(
            left, right, Index(scenario["left_index"]), Index(scenario["right_index"]), cfg
        )
        rows = self.rows(adapter, result)

        sort_columns = scenario["sort_columns"]
        rows.sort(key=lambda row: tuple((row.get(c) is None, row.get(c)) for c in sort_columns))

        expected_rows = scenario["expected_rows"]
        assert len(rows) == expected_rows, f"Expected {expected_rows} rows, got {len(rows)}"
        self._assert_left_rows_preserved(scenario["left"].to_python_dict()["lv"], rows)
        if scenario.get("exact_columns"):
            self._assert_exact_columns(rows, scenario["expected_columns"])
        else:
            self._assert_columns_exist(rows, scenario["expected_columns"])
        actual_rv = [_norm(row.get("rv")) for row in rows]
        normalized_expected = [_norm(v) for v in scenario["expected_rv"]]
        assert actual_rv == normalized_expected, f"rv mismatch: got {actual_rv}, expected {normalized_expected}"

    def _assert_left_rows_preserved(self, left_lv: list[Any], rows: Rows) -> None:
        """ASOF is a LEFT join: the multiset of lv is identical in the left input and the output."""
        expected = collections.Counter(_norm(v) for v in left_lv)
        actual = collections.Counter(_norm(row.get("lv")) for row in rows)
        assert actual == expected, (
            f"Left rows not preserved: lv multiset mismatch. expected {dict(expected)}, got {dict(actual)}"
        )

    def _assert_columns_exist(self, rows: Rows, expected_columns: list[str]) -> None:
        if not rows:
            return
        actual_columns = set(rows[0])
        for col in expected_columns:
            assert col in actual_columns, f"Expected column '{col}' not found. Available: {actual_columns}"

    def _assert_exact_columns(self, rows: Rows, expected_columns: list[str]) -> None:
        if not rows:
            return
        actual_columns = set(rows[0])
        assert actual_columns == set(expected_columns), (
            f"Column set mismatch: got {sorted(actual_columns)}, expected {sorted(expected_columns)}"
        )

    def test_backward_single_key(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Backward, single by-key, right rows shuffled."""
        self._run_asof_test(adapter, merge_engine, "backward_single_key")

    def test_forward_single_key(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Forward; a row with no later right time is null."""
        self._run_asof_test(adapter, merge_engine, "forward_single_key")

    def test_allow_exact_matches_true(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Exact-time row matches with allow_exact_matches=True."""
        self._run_asof_test(adapter, merge_engine, "exact_matches_true")

    def test_allow_exact_matches_false(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Exact-time row is excluded with allow_exact_matches=False."""
        self._run_asof_test(adapter, merge_engine, "exact_matches_false")

    def test_tolerance_numeric(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """A gap beyond the tolerance is null."""
        self._run_asof_test(adapter, merge_engine, "tolerance_numeric")

    def test_tolerance_none(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Without tolerance both rows match."""
        self._run_asof_test(adapter, merge_engine, "tolerance_none")

    def test_multi_by_key(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Multi by-key, backward."""
        self._run_asof_test(adapter, merge_engine, "multi_by_key")

    def test_differing_names(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Differing by-key and time-column names."""
        self._run_asof_test(adapter, merge_engine, "differing_names")

    def test_no_sortedness_warning_with_by_groups(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """An ASOF join with `by` groups must not emit a 'Sortedness' warning (polars warns, lazily at collect)."""
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            self._run_asof_test(adapter, merge_engine, "multi_by_key")

        offending = [str(w.message) for w in recorded if "Sortedness" in str(w.message)]
        assert not offending, f"Unexpected 'Sortedness' warning(s) emitted: {offending}"

    def test_string_time_column_raises_clear_error(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """A non-ordered (string) time column must raise a ValueError naming the column and the requirement."""
        left = self.frame(
            adapter,
            "string_time_left",
            TypedTestData({"k": (_S, ["a"]), "t": (_S, ["2025-06-01"]), "lv": (_I, [1])}),
        )
        right = self.frame(
            adapter,
            "string_time_right",
            TypedTestData({"k": (_S, ["a"]), "t": (_S, ["2025-05-01"]), "rv": (_D, [1.0])}),
        )
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")

        # The quoted column name rules out pandas' own MergeError ("both sides must have numeric dtype").
        with pytest.raises(ValueError, match=r"'t'.*(datetime|numeric)"):
            merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def _iso_frames(self, adapter: FrameworkAdapter, right_t: list[str], right_rv: list[float]) -> tuple[Any, Any]:
        left = TypedTestData({"k": (_S, ["a", "a"]), "t": (_S, ["2025-06-03", "2025-06-05"]), "lv": (_I, [1, 2])})
        right = TypedTestData({"k": (_S, ["a", "a"]), "t": (_S, right_t), "rv": (_D, right_rv)})
        return self.frame(adapter, "iso_left", left), self.frame(adapter, "iso_right", right)

    @staticmethod
    def _coerce_cfg() -> AsOfJoinConfig:
        return AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", coerce_time_columns=True
        )

    def test_coerce_time_columns_opt_in_iso_strings(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """With coerce_time_columns=True, ISO-8601 string time columns are coerced and the join succeeds."""
        left, right = self._iso_frames(adapter, ["2025-06-01", "2025-06-04"], [1.0, 2.0])

        result = merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), self._coerce_cfg())
        rows = self.rows(adapter, result)
        rows.sort(key=lambda row: row["lv"])

        assert [row["lv"] for row in rows] == [1, 2], "Every left row must survive the as-of join"
        actual_rv = [_norm(row["rv"]) for row in rows]
        assert actual_rv == [1.0, 2.0], f"rv mismatch after coercion: got {actual_rv}, expected [1.0, 2.0]"

    def test_coerce_unparseable_string_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Coercion fails HARD: a value that is not ISO-8601 must raise (lazy engines raise on read)."""
        left, right = self._iso_frames(adapter, ["2025-06-01", "not-a-date"], [1.0, 9.0])

        with pytest.raises(self.coercion_error_types()):
            result = merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), self._coerce_cfg())
            self.rows(adapter, result)

    def test_coerce_mixed_format_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """A non-ISO format ('06/03/2025') mixed into ISO strings must raise (lazy engines raise on read)."""
        left, right = self._iso_frames(adapter, ["2025-06-01", "06/03/2025"], [1.0, 2.0])

        with pytest.raises(self.coercion_error_types()):
            result = merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), self._coerce_cfg())
            self.rows(adapter, result)

    def test_coerce_preserves_nulls_and_transforms_strings(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """validate_asof_time_columns returns the pair; with coercion on, ISO strings become non-strings, nulls stay."""
        data = TypedTestData({"k": (_S, ["a", "a"]), "t": (_S, ["2025-06-01", None]), "lv": (_I, [1, 2])})
        left = self.frame(adapter, "nulls_left", data)
        right = self.frame(adapter, "nulls_right", data)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", coerce_time_columns=True)

        left_out, right_out = merge_engine.validate_asof_time_columns(left, right, cfg)
        assert right_out is not None

        by_lv = {row["lv"]: row for row in self.rows(adapter, left_out)}

        coerced = by_lv[1]["t"]
        assert self._normalize_time_value(coerced) is not None, "Coerced ISO string must not become null"
        assert not isinstance(coerced, str), f"'t' must no longer be a string after coercion, got {coerced!r}"

        assert self._normalize_time_value(by_lv[2]["t"]) is None, "Pre-existing null must stay null after coercion"

    def test_coerce_flag_noop_on_ordered_columns(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """coerce_time_columns=True is a no-op when the time columns are already ordered."""
        left = self.frame(adapter, "noop_left", TypedTestData({"k": (_I, [1]), "t": (_I, [10]), "lv": (_I, [100])}))
        right = self.frame(adapter, "noop_right", TypedTestData({"k": (_I, [1]), "t": (_I, [5]), "rv": (_D, [1.0])}))

        result = merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), self._coerce_cfg())
        rows = self.rows(adapter, result)

        assert len(rows) == 1
        actual_rv = [_norm(row["rv"]) for row in rows]
        assert actual_rv == [1.0], f"rv mismatch with no-op coercion flag: got {actual_rv}, expected [1.0]"
