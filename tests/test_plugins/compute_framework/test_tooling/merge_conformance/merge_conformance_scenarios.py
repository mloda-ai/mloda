"""Canonical cross-framework merge-engine conformance scenarios.

Every merge engine must produce the same result for the same input. The expected
frames below are the cross-framework contract, verified against pandas as the
reference. They are compared as an order-independent, null-normalized multiset of
rows (see ``test_merge_conformance``), so column/row order and 1 vs 1.0 do not matter.

Scope: this suite covers SINGLE-key equi-joins (same and differing key names);
multi-column differing-key-name joins are a known cross-framework divergence (pyarrow,
polars) that is NOT yet covered or fixed here.
"""

from typing import Any, TypedDict


class MergeConformanceScenario(TypedDict):
    """A single equi-merge scenario plus its expected frame per join type."""

    description: str
    left: list[dict[str, Any]]
    right: list[dict[str, Any]]
    left_index: tuple[str, ...]
    right_index: tuple[str, ...]
    expected: dict[str, list[dict[str, Any]]]


# The four equi-join types every engine must implement identically.
JOIN_TYPE_NAMES: tuple[str, ...] = ("INNER", "LEFT", "RIGHT", "OUTER")


MERGE_CONFORMANCE_SCENARIOS: dict[str, MergeConformanceScenario] = {
    "same_key": {
        "description": "Shared key name 'k' on both sides; left keys {1,2,3}, right keys {1,2,4}.",
        "left": [{"k": 1, "lv": "a"}, {"k": 2, "lv": "b"}, {"k": 3, "lv": "c"}],
        "right": [{"k": 1, "rv": "x"}, {"k": 2, "rv": "y"}, {"k": 4, "rv": "z"}],
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
        "left": [{"lk": 1, "lv": "a"}, {"lk": 2, "lv": "b"}, {"lk": 3, "lv": "c"}],
        "right": [{"rk": 1, "rv": "x"}, {"rk": 2, "rv": "y"}, {"rk": 4, "rv": "z"}],
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
