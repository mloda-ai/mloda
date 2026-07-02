"""Strict timezone-awareness contract for temporal range/min/max filters (epic #518, Phase 4).

Range / min / max filters receive NATIVE ``datetime`` bounds (PR #437). When the
bound's tz-awareness differs from the temporal column's (tz-aware column vs tz-naive
bound, or the reverse), the filter must raise a clear ``ValueError`` mentioning
timezone via ``ComparisonContract.require_compatible`` instead of comparing wrong or
surfacing a cryptic backend error.

These tests drive Phase 4. They exercise the ``do_filter`` dispatch path (the public
entry the guard lives behind) across all four in-tree frameworks:

- NEGATIVE: mismatched tz-awareness must raise a contract ``ValueError``.
- POSITIVE: matching tz-awareness (both aware or both naive) returns the right rows.
- NARROW: a numeric range filter is untouched by the temporal guard.

Current behavior (why the negatives fail now, per framework):

- pandas:      raises ``TypeError`` ("Invalid comparison between dtype=... and datetime").
- pyarrow:     raises ``pyarrow.lib.ArrowInvalid`` (a ``ValueError`` subclass) reading
               "Cannot compare timestamp with timezone to timestamp without timezone";
               it mentions "timezone" but NOT the contract wording, so the match fails.
- polars:      raises ``polars.exceptions.SchemaError`` (not a ``ValueError``).
- python_dict: raises ``TypeError`` ("can't compare offset-naive and offset-aware datetimes").

None of these is the clear, contract-level ``ValueError`` the negatives assert for,
so every negative fails today for the right reason.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

import pandas as pd

from mloda.provider import BaseFilterEngine
from mloda.user import Feature, FilterType, SingleFilter

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)

logger = logging.getLogger(__name__)

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

    POLARS_AVAILABLE = True
except ImportError:
    logger.warning("Polars is not installed. Polars framework cases will not be parametrized.")
    pl = None  # type: ignore
    PolarsFilterEngine = None  # type: ignore
    POLARS_AVAILABLE = False

try:
    import pyarrow as pa
    from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine

    PYARROW_AVAILABLE = True
except ImportError:
    logger.warning("PyArrow is not installed. PyArrow framework cases will not be parametrized.")
    pa = None  # type: ignore
    PyArrowFilterEngine = None  # type: ignore
    PYARROW_AVAILABLE = False


IDS: list[int] = [1, 2, 3, 4, 5, 6]
TS_AWARE: list[datetime] = [datetime(2023, 1, day, tzinfo=timezone.utc) for day in range(1, 7)]
TS_NAIVE: list[datetime] = [datetime(2023, 1, day) for day in range(1, 7)]
AGES: list[int] = [10, 20, 30, 40, 50, 60]

RANGE_MIN_AWARE = datetime(2023, 1, 2, tzinfo=timezone.utc)
RANGE_MAX_AWARE = datetime(2023, 1, 5, tzinfo=timezone.utc)
RANGE_MIN_NAIVE = datetime(2023, 1, 2)
RANGE_MAX_NAIVE = datetime(2023, 1, 5)
MIN_BOUND_AWARE = datetime(2023, 1, 4, tzinfo=timezone.utc)
MIN_BOUND_NAIVE = datetime(2023, 1, 4)

EXPECTED_RANGE_IDS: list[int] = [2, 3, 4]
EXPECTED_MIN_IDS: list[int] = [4, 5, 6]
EXPECTED_NUMERIC_IDS: list[int] = [2, 3, 4, 5]

# The Phase-4 guard reuses ComparisonContract.require_compatible, whose message is
# "... have incompatible timezone awareness." A bare "timezone" match would be
# satisfied today by pyarrow's ArrowInvalid (a ValueError), so the negatives pin the
# contract wording to guarantee they fail now and pass only once the guard is wired.
TZ_CONTRACT_MATCH = r"(?i)incompatible.*time[ -]?zone"


@dataclass(frozen=True)
class FrameworkSpec:
    """Framework-specific builders and accessors for the timezone-contract tests."""

    name: str
    engine: type[BaseFilterEngine]
    build_aware: Callable[[], Any]
    build_naive: Callable[[], Any]
    build_numeric: Callable[[], Any]
    get_ids: Callable[[Any], list[int]]


def _pandas_aware() -> Any:
    return pd.DataFrame({"id": IDS, "ts": pd.to_datetime(TS_AWARE, utc=True)})


def _pandas_naive() -> Any:
    return pd.DataFrame({"id": IDS, "ts": pd.to_datetime(TS_NAIVE)})


def _pandas_numeric() -> Any:
    return pd.DataFrame({"id": IDS, "age": AGES})


def _pandas_ids(result: Any) -> list[int]:
    return [int(v) for v in result["id"].tolist()]


def _pyarrow_aware() -> Any:
    return pa.table({"id": IDS, "ts": pa.array(TS_AWARE, type=pa.timestamp("us", tz="UTC"))})


def _pyarrow_naive() -> Any:
    return pa.table({"id": IDS, "ts": pa.array(TS_NAIVE, type=pa.timestamp("us"))})


def _pyarrow_numeric() -> Any:
    return pa.table({"id": IDS, "age": AGES})


def _pyarrow_ids(result: Any) -> list[int]:
    return [int(v) for v in result.column("id").to_pylist()]


def _polars_aware() -> Any:
    return pl.DataFrame(
        {"id": IDS, "ts": TS_AWARE},
        schema={"id": pl.Int64, "ts": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


def _polars_naive() -> Any:
    return pl.DataFrame(
        {"id": IDS, "ts": TS_NAIVE},
        schema={"id": pl.Int64, "ts": pl.Datetime(time_unit="us")},
    )


def _polars_numeric() -> Any:
    return pl.DataFrame({"id": IDS, "age": AGES}, schema={"id": pl.Int64, "age": pl.Int64})


def _polars_ids(result: Any) -> list[int]:
    return [int(v) for v in result["id"].to_list()]


def _python_dict_aware() -> Any:
    return {"id": list(IDS), "ts": list(TS_AWARE)}


def _python_dict_naive() -> Any:
    return {"id": list(IDS), "ts": list(TS_NAIVE)}


def _python_dict_numeric() -> Any:
    return {"id": list(IDS), "age": list(AGES)}


def _python_dict_ids(result: Any) -> list[int]:
    return [int(v) for v in result["id"]]


SPECS: list[FrameworkSpec] = [
    FrameworkSpec("pandas", PandasFilterEngine, _pandas_aware, _pandas_naive, _pandas_numeric, _pandas_ids),
    FrameworkSpec(
        "python_dict",
        PythonDictFilterEngine,
        _python_dict_aware,
        _python_dict_naive,
        _python_dict_numeric,
        _python_dict_ids,
    ),
]

if PYARROW_AVAILABLE:
    SPECS.append(
        FrameworkSpec("pyarrow", PyArrowFilterEngine, _pyarrow_aware, _pyarrow_naive, _pyarrow_numeric, _pyarrow_ids)
    )

if POLARS_AVAILABLE:
    SPECS.append(
        FrameworkSpec("polars", PolarsFilterEngine, _polars_aware, _polars_naive, _polars_numeric, _polars_ids)
    )

SPEC_IDS: list[str] = [spec.name for spec in SPECS]


def _range_filter(min_bound: Any, max_bound: Any) -> SingleFilter:
    return SingleFilter(
        Feature("ts"),
        FilterType.RANGE,
        {"min": min_bound, "max": max_bound, "max_exclusive": True},
    )


def _min_filter(bound: Any) -> SingleFilter:
    return SingleFilter(Feature("ts"), FilterType.MIN, {"value": bound})


# --------------------------------------------------------------------------- #
# NEGATIVE: mismatched tz-awareness must raise a contract ValueError.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_range_filter_tz_aware_column_naive_bound_raises(spec: FrameworkSpec) -> None:
    """Range filter: tz-aware column vs tz-naive datetime bounds must raise a tz ValueError."""
    single_filter = _range_filter(RANGE_MIN_NAIVE, RANGE_MAX_NAIVE)
    with pytest.raises(ValueError, match=TZ_CONTRACT_MATCH):
        spec.engine.do_filter(spec.build_aware(), single_filter)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_range_filter_tz_naive_column_aware_bound_raises(spec: FrameworkSpec) -> None:
    """Range filter: tz-naive column vs tz-aware datetime bounds must raise a tz ValueError."""
    single_filter = _range_filter(RANGE_MIN_AWARE, RANGE_MAX_AWARE)
    with pytest.raises(ValueError, match=TZ_CONTRACT_MATCH):
        spec.engine.do_filter(spec.build_naive(), single_filter)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_min_filter_tz_aware_column_naive_bound_raises(spec: FrameworkSpec) -> None:
    """Min filter: tz-aware column vs tz-naive datetime bound must raise a tz ValueError."""
    single_filter = _min_filter(MIN_BOUND_NAIVE)
    with pytest.raises(ValueError, match=TZ_CONTRACT_MATCH):
        spec.engine.do_filter(spec.build_aware(), single_filter)


# --------------------------------------------------------------------------- #
# POSITIVE: matching tz-awareness returns the correctly filtered rows.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_range_filter_both_tz_aware_returns_rows(spec: FrameworkSpec) -> None:
    """Range filter with matching tz-aware column and bounds returns the in-range ids."""
    single_filter = _range_filter(RANGE_MIN_AWARE, RANGE_MAX_AWARE)
    result = spec.engine.do_filter(spec.build_aware(), single_filter)
    assert sorted(spec.get_ids(result)) == EXPECTED_RANGE_IDS


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_range_filter_both_tz_naive_returns_rows(spec: FrameworkSpec) -> None:
    """Range filter with matching tz-naive column and bounds returns the in-range ids."""
    single_filter = _range_filter(RANGE_MIN_NAIVE, RANGE_MAX_NAIVE)
    result = spec.engine.do_filter(spec.build_naive(), single_filter)
    assert sorted(spec.get_ids(result)) == EXPECTED_RANGE_IDS


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_min_filter_both_tz_aware_returns_rows(spec: FrameworkSpec) -> None:
    """Min filter with matching tz-aware column and bound returns the ids at/after the bound."""
    single_filter = _min_filter(MIN_BOUND_AWARE)
    result = spec.engine.do_filter(spec.build_aware(), single_filter)
    assert sorted(spec.get_ids(result)) == EXPECTED_MIN_IDS


# --------------------------------------------------------------------------- #
# NARROW: numeric range filters are unaffected by the temporal timezone guard.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_numeric_range_filter_unaffected_by_tz_guard(spec: FrameworkSpec) -> None:
    """A numeric range filter on a numeric column raises nothing and filters normally."""
    single_filter = SingleFilter(
        Feature("age"),
        FilterType.RANGE,
        {"min": 20, "max": 50, "max_exclusive": False},
    )
    result = spec.engine.do_filter(spec.build_numeric(), single_filter)
    assert sorted(spec.get_ids(result)) == EXPECTED_NUMERIC_IDS
