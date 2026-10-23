"""build_request turns a validated feature request into mloda Features, a GlobalFilter and a canonical form."""

import json
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.api.feature_config.loader import load_features_from_config
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.filter.single_filter import SingleFilter
from mloda.interfaces.service.request import ParsedRequest, Problem, build_request

FEATURES: list[Any] = [
    "plain",
    {"name": "configured", "options": {"k": 1}},
    {"name": "derived", "in_features": ["plain"], "context_options": {"c": 2}},
]


def _parsed(request: dict[str, Any]) -> ParsedRequest:
    result = build_request(request)
    assert isinstance(result, ParsedRequest), result
    return result


def _names(features: list[Feature | str]) -> list[str]:
    return [f if isinstance(f, str) else f.name for f in features]


def test_features_match_feature_config_loader() -> None:
    parsed = _parsed({"features": FEATURES})
    expected = load_features_from_config(json.dumps(FEATURES))
    assert _names(parsed.features) == _names(expected)
    assert [type(f) for f in parsed.features] == [type(f) for f in expected]
    assert parsed.features == expected


def test_no_filters_and_no_sources() -> None:
    parsed = _parsed({"features": ["a"]})
    assert parsed.global_filter is None
    assert parsed.restrict_sources == ()


def test_restrict_sources_is_tuple() -> None:
    assert _parsed({"features": ["a"], "restrict_sources": ["s1", "s2"]}).restrict_sources == ("s1", "s2")


def test_filters_map_to_global_filter() -> None:
    request = {
        "features": ["a"],
        "filters": [
            {"feature": "a", "op": "eq", "value": 5},
            {"feature": "b", "op": "in", "values": ["x", "y"]},
            {"feature": "c", "op": "range", "min": 1, "max": 10},
            {"feature": "d", "op": "range", "max": 3, "max_exclusive": True},
        ],
    }

    global_filter = _parsed(request).global_filter

    assert isinstance(global_filter, GlobalFilter)
    assert global_filter.filters == {
        SingleFilter("a", FilterType.EQUAL, {"value": 5}),
        SingleFilter("b", FilterType.CATEGORICAL_INCLUSION, {"values": ("x", "y")}),
        SingleFilter("c", FilterType.RANGE, {"min": 1, "max": 10}),
        SingleFilter("d", FilterType.RANGE, {"max": 3, "max_exclusive": True}),
    }


def test_range_filter_carries_only_present_keys() -> None:
    global_filter = _parsed({"features": ["a"], "filters": [{"feature": "a", "op": "range", "min": 0}]}).global_filter
    assert global_filter is not None
    (single,) = global_filter.filters
    assert single.filter_type == FilterType.RANGE.value
    assert single.parameter == SingleFilter("a", FilterType.RANGE, {"min": 0}).parameter
    assert single.parameter.max_value is None


def test_normalized_is_json_serialisable() -> None:
    request = {"features": FEATURES, "filters": [{"feature": "a", "op": "eq", "value": 1}], "restrict_sources": ["s"]}
    json.dumps(_parsed(request).normalized)


def test_normalized_ignores_key_order() -> None:
    one = {
        "features": [{"name": "b", "options": {"x": 1, "y": 2}}],
        "filters": [{"feature": "a", "op": "range", "min": 1, "max": 2}],
        "restrict_sources": ["s"],
    }
    other = {
        "restrict_sources": ["s"],
        "filters": [{"max": 2, "min": 1, "op": "range", "feature": "a"}],
        "features": [{"options": {"y": 2, "x": 1}, "name": "b"}],
    }
    assert json.dumps(_parsed(one).normalized) == json.dumps(_parsed(other).normalized)


def test_normalized_differs_for_different_requests() -> None:
    one = _parsed({"features": [{"name": "b", "options": {"x": 1}}]}).normalized
    other = _parsed({"features": [{"name": "b", "options": {"x": 2}}]}).normalized
    assert one != other


def test_semantically_invalid_feature_config_is_a_problem() -> None:
    request = {"features": ["a", {"name": "b", "options": {"x": 1}, "group_options": {"y": 1}}]}

    result = build_request(request)

    assert isinstance(result, list)
    assert len(result) == 1
    problem = result[0]
    assert isinstance(problem, Problem)
    assert problem.problem == "invalid_feature_config"
    assert problem.path == "/features/1"
    assert problem.message


INLINE: dict[str, Any] = {"S": {"a": ["x", 1, None], "b": [True, 2.5, "y"]}, "T": {"c": [1]}}


def test_inline_data_defaults_to_empty() -> None:
    assert dict(_parsed({"features": ["a"]}).inline_data) == {}


def test_inline_data_is_parsed() -> None:
    parsed = _parsed({"features": ["a"], "data": INLINE})
    assert {name: dict(columns) for name, columns in parsed.inline_data.items()} == INLINE


def test_normalized_carries_data() -> None:
    normalized = _parsed({"features": ["a"], "data": INLINE}).normalized
    assert normalized["data"] == INLINE
    json.dumps(normalized)


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"S": {"a": ["s3cr3t", 2], "b": [1]}}, "/data/S"),
        ({"S": {"a": ["s3cr3t"]}, "T": {"a": [2]}}, "/data/T/a"),
    ],
)
def test_inconsistent_inline_data_is_a_problem_without_values(data: dict[str, Any], path: str) -> None:
    result = build_request({"features": ["a"], "data": data})
    assert isinstance(result, list)
    assert [(p.path, p.problem) for p in result] == [(path, "invalid_inline_data")]
    assert result[0].message
    assert "s3cr3t" not in result[0].message
