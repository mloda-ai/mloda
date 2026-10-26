"""build_request turns a validated feature request into mloda Features, a GlobalFilter and a canonical form."""

import json
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.api.feature_config.loader import load_features_from_config
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.filter.single_filter import SingleFilter
from mloda.interfaces.service.config import Settings
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


def _settings() -> Settings:
    return Settings(options={"model": "m", "retries": 3}, context_options={"timeout": 30})


def _defaulted(request: dict[str, Any], settings: Settings | None = None) -> ParsedRequest:
    result = build_request(request, settings=settings or _settings())
    assert isinstance(result, ParsedRequest), result
    return result


def _feature(parsed: ParsedRequest, index: int = 0) -> Feature:
    feature = parsed.features[index]
    assert isinstance(feature, Feature), feature
    return feature


def test_defaults_land_on_a_string_feature_as_group_and_context_options() -> None:
    feature = _feature(_defaulted({"features": ["a"]}))

    assert feature.name == "a"
    assert feature.options.group == {"model": "m", "retries": 3}
    assert feature.options.context == {"timeout": 30}


def test_config_feature_own_value_wins_over_the_default() -> None:
    request = {"features": [{"name": "a", "group_options": {"model": "own"}, "context_options": {"timeout": 5}}]}

    feature = _feature(_defaulted(request))

    assert feature.options.group == {"model": "own", "retries": 3}
    assert feature.options.context == {"timeout": 5}


def test_flat_options_combine_with_context_defaults() -> None:
    feature = _feature(_defaulted({"features": [{"name": "a", "options": {"model": "own", "extra": 1}}]}))

    assert feature.options.group == {"model": "own", "retries": 3, "extra": 1}
    assert feature.options.context == {"timeout": 30}


def test_defaults_apply_to_every_top_level_feature() -> None:
    parsed = _defaulted({"features": ["a", {"name": "b", "group_options": {"x": 1}}]})

    assert [_feature(parsed, i).options.context for i in range(2)] == [{"timeout": 30}, {"timeout": 30}]
    assert _feature(parsed, 1).options.group == {"model": "m", "retries": 3, "x": 1}


def test_normalized_request_keeps_only_the_callers_request() -> None:
    request = {"features": ["a", {"name": "b", "options": {"x": 1}}]}

    parsed = _defaulted(request)

    assert parsed.normalized == json.loads(json.dumps(request, sort_keys=True))
    assert "30" not in json.dumps(parsed.normalized)


def test_no_defaults_keeps_features_unchanged() -> None:
    assert _defaulted({"features": FEATURES}, Settings()).features == load_features_from_config(json.dumps(FEATURES))


def test_feature_group_option_wins_over_a_context_default_of_the_same_key() -> None:
    settings = Settings(context_options={"model": "x"})

    feature = _feature(_defaulted({"features": [{"name": "a", "group_options": {"model": "y"}}]}, settings))

    assert feature.options.group == {"model": "y"}
    assert "model" not in feature.options.context


def test_feature_context_option_wins_over_a_group_default_of_the_same_key() -> None:
    settings = Settings(options={"model": "x"})

    feature = _feature(_defaulted({"features": [{"name": "a", "context_options": {"model": "y"}}]}, settings))

    assert feature.options.context == {"model": "y"}
    assert "model" not in feature.options.group


@pytest.mark.parametrize("other", ["group_options", "context_options"])
def test_options_with_group_or_context_options_is_refused_with_or_without_defaults(other: str) -> None:
    request = {"features": [{"name": "a", "options": {"x": 1}, other: {"y": 1}}]}

    plain = build_request(request)
    defaulted = build_request(request, settings=_settings())

    assert isinstance(plain, list)
    assert isinstance(defaulted, list)
    assert [(p.path, p.problem) for p in defaulted] == [(p.path, p.problem) for p in plain]
