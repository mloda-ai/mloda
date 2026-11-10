"""Shipped name-path feature groups pass the shape check when probed with the names users request."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

import pytest

from mloda._testing.feature_group_shape import assert_feature_group_shape
from mloda._testing.framework_adapter import PandasAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.options import Options

_ROOT = "mloda_plugins.feature_group.experimental"


def _numbers() -> TypedTestData:
    return TypedTestData({"x": (DataType.DOUBLE, [1.0, 2.0, 3.0, 4.0])})


def _numbers_with_gap() -> TypedTestData:
    return TypedTestData({"x": (DataType.DOUBLE, [1.0, None, 3.0, 4.0])})


def _text() -> TypedTestData:
    return TypedTestData({"t": (DataType.STRING, ["Hello  World", "Foo Bar", "x", "Baz"])})


def _timed() -> TypedTestData:
    days = [datetime(2024, 1, d, tzinfo=UTC) for d in (1, 2, 3, 4)]
    return TypedTestData(
        {"x": (DataType.DOUBLE, [1.0, 2.0, 3.0, 4.0]), "reference_time": (DataType.TIMESTAMP_MICROS_TZ, days)}
    )


@dataclass(frozen=True)
class Case:
    module: str
    group: str
    requests: tuple[str, ...]
    data: Any
    options: dict[str, Any]
    needs: tuple[str, ...] = ("pandas",)


CASES = {
    "aggregation": Case(
        f"{_ROOT}.aggregated_feature_group",
        "AggregatedFeatureGroup",
        ("x__sum_aggr", "x__count_aggr", "x__median_aggr"),
        _numbers,
        {},
    ),
    "missing_value": Case(
        f"{_ROOT}.data_quality.missing_value",
        "MissingValueFeatureGroup",
        ("x__mean_imputed", "x__median_imputed", "x__ffill_imputed"),
        _numbers_with_gap,
        {},
    ),
    "time_window": Case(
        f"{_ROOT}.time_window", "TimeWindowFeatureGroup", ("x__sum_2_day_window", "x__avg_2_day_window"), _timed, {}
    ),
    "text_cleaning": Case(
        f"{_ROOT}.text_cleaning",
        "TextCleaningFeatureGroup",
        ("t__cleaned_text",),
        _text,
        {"cleaning_operations": ("normalize", "remove_punctuation")},
        ("pandas", "nltk"),
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
def test_shipped_group_passes_shape_check_for_requested_names(case: Case) -> None:
    for module in case.needs:
        pytest.importorskip(module)
    group = getattr(importlib.import_module(f"{case.module}.base"), case.group)
    importlib.import_module(f"{case.module}.pandas")
    assert_feature_group_shape(
        group,
        Options(group=case.options) if case.options else None,
        requests=case.requests,
        input_data=case.data(),
        adapter=PandasAdapter(),
    )
