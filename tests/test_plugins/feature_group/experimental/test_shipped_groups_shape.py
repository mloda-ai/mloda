"""Shipped name-path feature groups pass the shape check when probed with the names users request."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

import pytest

from mloda._testing.feature_group_shape import assert_feature_group_shape, check_feature_group_shape
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import current_implementations
from tests.test_plugins.feature_group.experimental.test_operation_family_contract import (
    ADAPTERS,
    LOOP_INPUT_CASES,
    OPERATION_FAMILIES,
)


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
    requests: tuple[str, ...]
    data: Callable[[], TypedTestData]
    options: dict[str, Any]
    needs: tuple[str, ...] = ()


CASES = {
    "aggregation": Case(("x__sum_aggr", "x__count_aggr", "x__median_aggr"), _numbers, {}),
    "missing_value": Case(("x__mean_imputed", "x__median_imputed", "x__ffill_imputed"), _numbers_with_gap, {}),
    "time_window": Case(("x__sum_2_day_window", "x__avg_2_day_window"), _timed, {}),
    "text_cleaning": Case(("t__cleaned_text",), _text, {"cleaning_operations": ("normalize", "remove_punctuation")}),
    "scaling": Case(
        ("x__standard_scaled", "x__minmax_scaled", "x__robust_scaled", "x__normalizer_scaled"),
        _numbers,
        {},
        ("sklearn",),
    ),
    "encoding": Case(("t__label_encoded", "t__ordinal_encoded"), _text, {}, ("sklearn",)),
    "clustering": Case(("a__cluster_kmeans_2",), LOOP_INPUT_CASES["clustering"][0], {}, ("sklearn",)),
    "forecasting": Case(("sales__linear_forecast_7day",), LOOP_INPUT_CASES["forecasting"][0], {}, ("sklearn",)),
    "dimensionality_reduction": Case(
        ("a,b,c__pca_2d",), LOOP_INPUT_CASES["dimensionality_reduction"][0], {}, ("sklearn",)
    ),
    "node_centrality": Case(("user__degree_centrality",), LOOP_INPUT_CASES["node_centrality"][0], {}),
    "geo_distance": Case(("a&b__haversine_distance",), LOOP_INPUT_CASES["geo_distance"][0], {}),
    "sklearn_pipeline": Case(
        ("a,b__sklearn_pipeline_scaling",), LOOP_INPUT_CASES["sklearn_pipeline"][0], {}, ("sklearn",)
    ),
}

# Group defects and check limits found by this suite, pinned until fixed; divergence records cannot excuse them.
_EMPTY_SKLEARN = r"zero_rows: run on zero rows raised: Found array with 0 sample\(s\)"
KNOWN_FAILURES: dict[str, tuple[str, ...]] = {
    "clustering-PandasClustering": (_EMPTY_SKLEARN,),
    "dimensionality_reduction-PandasDimensionalityReduction": (_EMPTY_SKLEARN,),
    "sklearn_pipeline-PandasSklearnPipeline": (_EMPTY_SKLEARN,),
    "forecasting-PandasForecasting": (
        r"outputs: input column '\w+' was changed",
        r"zero_rows: run on zero rows raised: No training data available",
    ),
    "geo_distance-PandasGeoDistance": (
        r"zero_rows: run on zero rows raised: column 'a': .*needs an arrow_types override",
    ),
}

_PARAMS = [
    pytest.param(
        name,
        impl,
        id=f"{name}-{impl.__name__}",
        marks=pytest.mark.skipif(not impl.compute_framework.is_available(), reason="framework unavailable"),
    )
    for name, family in OPERATION_FAMILIES.items()
    for impl in family[1]
]


def test_every_operation_family_has_a_shape_case() -> None:
    assert set(CASES) == set(OPERATION_FAMILIES)


def test_known_failures_name_real_params() -> None:
    assert set(KNOWN_FAILURES) <= {p.id for p in _PARAMS}


@pytest.mark.parametrize(("name", "impl"), _PARAMS)
def test_shipped_group_passes_shape_check_for_requested_names(name: str, impl: type[Any]) -> None:
    case = CASES[name]
    for module in case.needs:
        pytest.importorskip(module)
    group = OPERATION_FAMILIES[name][0]
    assert group.select_implementation(impl.compute_framework, list(current_implementations())) is impl
    options = Options(group=case.options) if case.options else None
    adapter = ADAPTERS[impl.compute_framework]
    pinned = KNOWN_FAILURES.get(f"{name}-{impl.__name__}")
    if pinned is None:
        assert_feature_group_shape(group, options, requests=case.requests, input_data=case.data(), adapter=adapter)
        return
    failures = check_feature_group_shape(
        group, options, requests=case.requests, input_data=case.data(), adapter=adapter, require_run=True
    ).failures
    unexpected = [f for f in failures if not any(re.search(p, f) for p in pinned)]
    assert not unexpected, f"unexpected failures: {unexpected}"
    fixed = [p for p in pinned if not any(re.search(p, f) for f in failures)]
    assert not fixed, f"fixed, drop from KNOWN_FAILURES: {fixed}"
