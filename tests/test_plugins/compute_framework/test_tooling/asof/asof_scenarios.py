"""
Reusable test scenarios for ASOF (point-in-time) merge operations.

These scenarios are framework-agnostic and can be used by any compute framework
test by converting the dict format to the framework's native format.
Expected values are copied verbatim from the canonical pandas reference tests.
"""

from typing import Any, TypedDict


class AsofScenario(TypedDict):
    """Type definition for an ASOF merge test scenario."""

    description: str
    left: list[dict[str, Any]]
    right: list[dict[str, Any]]
    left_index: tuple[str, ...]
    right_index: tuple[str, ...]
    left_time_column: str
    right_time_column: str
    cfg_kwargs: dict[str, Any]
    sort_columns: list[str]
    expected_rows: int
    expected_columns: list[str]
    expected_rv: list[Any]


# Canonical ASOF test scenarios. Data and expected values copied from the pandas
# reference tests so semantics are preserved across all backends.
ASOF_SCENARIOS: dict[str, AsofScenario] = {
    "backward_single_key": {
        "description": "Vector A: backward, single by-key. Right rows shuffled to prove internal sorting.",
        "left": [{"k": 1, "t": 10, "lv": 100}, {"k": 1, "t": 20, "lv": 200}, {"k": 2, "t": 15, "lv": 300}],
        # right is intentionally shuffled to verify the engine sorts internally
        "right": [
            {"k": 2, "t": 30, "rv": 4},
            {"k": 1, "t": 5, "rv": 1},
            {"k": 1, "t": 18, "rv": 2},
            {"k": 2, "t": 5, "rv": 3},
        ],
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward"},
        "sort_columns": ["k", "t"],
        "expected_rows": 3,
        "expected_columns": ["k", "t", "lv", "rv"],
        "expected_rv": [1, 2, 3],
    },
    "forward_single_key": {
        "description": "Vector B: forward; row (k=1, t=20) has no right_time >= 20 -> null.",
        "left": [{"k": 1, "t": 10, "lv": 100}, {"k": 1, "t": 20, "lv": 200}, {"k": 2, "t": 15, "lv": 300}],
        "right": [
            {"k": 1, "t": 5, "rv": 1},
            {"k": 1, "t": 18, "rv": 2},
            {"k": 2, "t": 5, "rv": 3},
            {"k": 2, "t": 30, "rv": 4},
        ],
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "forward"},
        "sort_columns": ["k", "t"],
        "expected_rows": 3,
        "expected_columns": ["k", "t", "lv", "rv"],
        "expected_rv": [2, None, 4],
    },
    "exact_matches_true": {
        "description": "Vector C: backward + allow_exact_matches=True -> exact-time row matched (rv=99).",
        "left": [{"k": 1, "t": 10, "lv": 100}],
        "right": [{"k": 1, "t": 10, "rv": 99}, {"k": 1, "t": 5, "rv": 1}],
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "allow_exact_matches": True},
        "sort_columns": ["k", "t"],
        "expected_rows": 1,
        "expected_columns": ["k", "t", "lv", "rv"],
        "expected_rv": [99],
    },
    "exact_matches_false": {
        "description": "Vector C: backward + allow_exact_matches=False -> exact excluded, prior row (rv=1).",
        "left": [{"k": 1, "t": 10, "lv": 100}],
        "right": [{"k": 1, "t": 10, "rv": 99}, {"k": 1, "t": 5, "rv": 1}],
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "allow_exact_matches": False},
        "sort_columns": ["k", "t"],
        "expected_rows": 1,
        "expected_columns": ["k", "t", "lv", "rv"],
        "expected_rv": [1],
    },
    "tolerance_numeric": {
        "description": "Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null.",
        "left": [{"k": 1, "t": 10, "lv": 1}, {"k": 1, "t": 100, "lv": 2}],
        "right": [{"k": 1, "t": 8, "rv": 7}],
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "tolerance": 5},
        "sort_columns": ["k", "t"],
        "expected_rows": 2,
        "expected_columns": ["k", "t", "lv", "rv"],
        "expected_rv": [7, None],
    },
    "tolerance_none": {
        "description": "Vector D: backward, tolerance=None -> both rows match (rv=7, rv=7).",
        "left": [{"k": 1, "t": 10, "lv": 1}, {"k": 1, "t": 100, "lv": 2}],
        "right": [{"k": 1, "t": 8, "rv": 7}],
        "left_index": ("k",),
        "right_index": ("k",),
        "left_time_column": "t",
        "right_time_column": "t",
        "cfg_kwargs": {"direction": "backward", "tolerance": None},
        "sort_columns": ["k", "t"],
        "expected_rows": 2,
        "expected_columns": ["k", "t", "lv", "rv"],
        "expected_rv": [7, 7],
    },
    "multi_by_key": {
        "description": "Vector E: multi by-key (k1, k2), backward.",
        "left": [{"k1": 1, "k2": "a", "t": 10, "lv": 1}, {"k1": 1, "k2": "b", "t": 10, "lv": 2}],
        "right": [{"k1": 1, "k2": "a", "t": 5, "rv": 10}, {"k1": 1, "k2": "b", "t": 5, "rv": 20}],
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
}
