"""Row-order contract for the eager and lazy polars merge engines (issue #2252).

inner/left keep left-frame order, right keeps right-frame order, full keeps left order then
unmatched right rows in right order. Frames are large and shuffled so an unordered join cannot pass.
"""

import random
from typing import Any

import pytest

from mloda.provider import BaseMergeEngine
from mloda.user import Index, JoinType
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_ROWS = 400
_OVERLAP = 200

_rng = random.Random(2252)  # nosec B311
_POOL = _rng.sample(range(100_000), 2 * _ROWS - _OVERLAP)
LEFT_KEYS: list[int] = _POOL[:_ROWS]
RIGHT_KEYS: list[int] = _POOL[_ROWS - _OVERLAP :]
_rng.shuffle(RIGHT_KEYS)

Pair = tuple[str | None, str | None]


def _lv(key: int) -> str:
    return f"l{key}"


def _rv(key: int) -> str:
    return f"r{key}"


def _expected_pairs(join_name: str) -> list[Pair]:
    """Explicit (lv, rv) sequence each join type must produce, built without polars."""
    left_set, right_set = set(LEFT_KEYS), set(RIGHT_KEYS)
    left_rows: list[Pair] = [(_lv(k), _rv(k) if k in right_set else None) for k in LEFT_KEYS]
    if join_name == "INNER":
        return [(_lv(k), _rv(k)) for k in LEFT_KEYS if k in right_set]
    if join_name == "LEFT":
        return left_rows
    if join_name == "RIGHT":
        return [(_lv(k) if k in left_set else None, _rv(k)) for k in RIGHT_KEYS]
    return left_rows + [(None, _rv(k)) for k in RIGHT_KEYS if k not in left_set]


def _key_columns(keys: list[int], names: tuple[str, ...]) -> dict[str, list[int]]:
    if len(names) == 1:
        return {names[0]: keys}
    return {names[0]: [k % 13 for k in keys], names[1]: [k // 13 for k in keys]}


KEY_MODES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "same_key": (("idx",), ("idx",)),
    "diff_key": (("lk",), ("rk",)),
    "multi_key": (("k1", "k2"), ("k1", "k2")),
}

ENGINES: dict[str, tuple[type[BaseMergeEngine], bool]] = {
    "eager": (PolarsMergeEngine, False),
    "lazy": (PolarsLazyMergeEngine, True),
}


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsMergeRowOrder:
    """Eager and lazy polars merges return rows in the documented, deterministic order."""

    @pytest.mark.parametrize("key_mode", list(KEY_MODES))
    @pytest.mark.parametrize("join_name", ["INNER", "LEFT", "RIGHT", "OUTER"])
    @pytest.mark.parametrize("engine_name", list(ENGINES))
    def test_join_row_order(self, engine_name: str, join_name: str, key_mode: str) -> None:
        engine_cls, lazy = ENGINES[engine_name]
        left_names, right_names = KEY_MODES[key_mode]

        left: Any = pl.DataFrame({**_key_columns(LEFT_KEYS, left_names), "lv": [_lv(k) for k in LEFT_KEYS]})
        right: Any = pl.DataFrame({**_key_columns(RIGHT_KEYS, right_names), "rv": [_rv(k) for k in RIGHT_KEYS]})
        if lazy:
            left, right = left.lazy(), right.lazy()

        link = make_merge_link(JoinType[join_name], Index(left_names), Index(right_names))
        result = engine_cls().merge(left, right, link)
        if lazy:
            assert isinstance(result, pl.LazyFrame)
            result = result.collect()

        actual: list[Pair] = list(zip(result["lv"].to_list(), result["rv"].to_list()))
        expected = _expected_pairs(join_name)

        assert sorted(actual, key=repr) == sorted(expected, key=repr), "row content differs, not just order"
        mismatches = [i for i, (a, e) in enumerate(zip(actual, expected)) if a != e]
        assert not mismatches, (
            f"{engine_name} {join_name} ({key_mode}) row order differs at {len(mismatches)}/{len(expected)} "
            f"positions; first at {mismatches[0]}: got {actual[mismatches[0]]}, expected {expected[mismatches[0]]}"
        )
