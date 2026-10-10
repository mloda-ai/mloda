"""Feature-request JSON Schema and validate_request: precise JSON pointers and schema/validator agreement."""

import json
from typing import Any

import pytest

from mloda.core.api.feature_config.models import feature_config_schema
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.request import (
    MAX_FEATURES,
    MAX_FILTERS,
    MAX_NAME_LENGTH,
    SCHEMA_ID,
    Problem,
    request_schema,
    validate_request,
)

SALES_SETTINGS = Settings(sources={"sales": SourceConfig(alias="sales")})


def _refs(node: Any) -> list[str]:
    if isinstance(node, dict):
        found = [node["$ref"]] if isinstance(node.get("$ref"), str) else []
        for value in node.values():
            found.extend(_refs(value))
        return found
    if isinstance(node, list):
        return [ref for item in node for ref in _refs(item)]
    return []


def _paths(problems: list[Problem]) -> list[str]:
    return [p.path for p in problems]


def test_constants() -> None:
    assert SCHEMA_ID == "mloda://schema/feature-request"
    assert MAX_FEATURES == 50
    assert MAX_FILTERS == 20
    assert MAX_NAME_LENGTH == 200


def test_problem_suggestion_defaults_to_none() -> None:
    assert Problem(path="", problem="invalid", message="m").suggestion is None


def test_schema_root() -> None:
    schema = request_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == SCHEMA_ID
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["features"]
    assert set(schema["properties"]) == {"features", "filters", "restrict_sources"}
    json.dumps(schema)


def test_schema_refs_are_local_and_resolvable() -> None:
    schema = request_schema()
    refs = _refs(schema)
    assert refs
    for ref in refs:
        assert ref.startswith("#/$defs/"), ref
        assert ref.removeprefix("#/$defs/") in schema["$defs"], ref


def test_schema_features_property() -> None:
    features = request_schema()["properties"]["features"]
    assert features["type"] == "array"
    assert features["minItems"] == 1
    assert features["maxItems"] == MAX_FEATURES
    variants = features["items"]["oneOf"]
    strings = [v for v in variants if v.get("type") == "string"]
    assert len(strings) == 1
    assert strings[0]["minLength"] == 1
    assert strings[0]["maxLength"] == MAX_NAME_LENGTH
    assert {"$ref": "#/$defs/featureConfig"} in variants


def test_schema_feature_config_def_mirrors_core_schema() -> None:
    feature_config = request_schema()["$defs"]["featureConfig"]
    assert set(feature_config["properties"]) == set(feature_config_schema()["properties"])
    assert feature_config["additionalProperties"] is False
    assert feature_config["required"] == ["name"]
    assert feature_config["properties"]["name"]["maxLength"] == MAX_NAME_LENGTH


def test_schema_filters_property() -> None:
    schema = request_schema()
    filters = schema["properties"]["filters"]
    assert filters["type"] == "array"
    assert filters["maxItems"] == MAX_FILTERS
    assert filters["items"] == {"$ref": "#/$defs/filter"}
    filter_def = schema["$defs"]["filter"]
    assert filter_def["additionalProperties"] is False
    assert set(filter_def["required"]) == {"feature", "op"}
    assert set(filter_def["properties"]) == {"feature", "op", "value", "values", "min", "max", "max_exclusive"}
    assert set(filter_def["properties"]["op"]["enum"]) == {"eq", "in", "range"}


def test_schema_restrict_sources_property() -> None:
    restrict = request_schema()["properties"]["restrict_sources"]
    assert restrict["type"] == "array"
    assert restrict["uniqueItems"] is True
    assert restrict["items"]["type"] == "string"


def test_valid_request_has_no_problems() -> None:
    request = {
        "features": ["a", {"name": "b", "options": {"k": 1}}],
        "filters": [{"feature": "a", "op": "eq", "value": 1}],
        "restrict_sources": ["sales"],
    }
    assert validate_request(request, SALES_SETTINGS) == []


@pytest.mark.parametrize(
    ("request_value", "path"),
    [
        pytest.param("not an object", "", id="non_object_root"),
        pytest.param({}, "/features", id="missing_features"),
        pytest.param({"features": ["a"], "optons": {}}, "/optons", id="unknown_top_level_key"),
        pytest.param({"features": ["a", "b", 3]}, "/features/2", id="wrong_item_type"),
        pytest.param({"features": [{"name": "a", "optons": {}}]}, "/features/0/optons", id="unknown_config_key"),
        pytest.param({"features": [f"f{i}" for i in range(MAX_FEATURES + 1)]}, "/features", id="too_many_features"),
        pytest.param({"features": []}, "/features", id="empty_features"),
        pytest.param({"features": ["a"], "filters": [{"feature": "a"}]}, "/filters/0", id="filter_missing_op"),
        pytest.param(
            {"features": ["a"], "filters": [{"feature": "a", "op": "in"}]}, "/filters/0", id="in_without_values"
        ),
        pytest.param(
            {"features": ["a"], "filters": [{"feature": "a", "op": "eq"}]}, "/filters/0", id="eq_without_value"
        ),
        pytest.param(
            {"features": ["a"], "filters": [{"feature": "a", "op": "range"}]}, "/filters/0", id="range_without_bounds"
        ),
    ],
)
def test_invalid_request_reports_one_problem_at_path(request_value: Any, path: str) -> None:
    problems = validate_request(request_value, Settings())
    assert _paths(problems) == [path]
    assert problems[0].problem == "invalid"
    assert problems[0].message


def test_multiple_issues_report_one_problem_each() -> None:
    request = {"features": ["a", 3, {"name": "b", "optons": {}}], "optons": 1}
    paths = set(_paths(validate_request(request, Settings())))
    assert paths == {"/features/1", "/features/2/optons", "/optons"}


def test_unknown_source_with_close_suggestion() -> None:
    problems = validate_request({"features": ["a"], "restrict_sources": ["sales", "sale"]}, SALES_SETTINGS)
    assert len(problems) == 1
    assert problems[0].path == "/restrict_sources/1"
    assert problems[0].problem == "unknown_source"
    assert problems[0].suggestion == "sales"


def test_unknown_source_without_close_match_has_no_suggestion() -> None:
    problems = validate_request({"features": ["a"], "restrict_sources": ["zzzzzz"]}, SALES_SETTINGS)
    assert len(problems) == 1
    assert problems[0].path == "/restrict_sources/0"
    assert problems[0].problem == "unknown_source"
    assert problems[0].suggestion is None


def _filter_request(**fields: Any) -> dict[str, Any]:
    return {"features": ["a"], "filters": [{"feature": "a", **fields}]}


AGREEMENT_CORPUS: list[Any] = [
    # valid
    {"features": ["a"]},
    {"features": ["a", {"name": "b"}]},
    {"features": [{"name": "b", "options": {"k": 1, "in_features": {"name": "src"}}}]},
    {"features": [{"name": "b", "in_features": ["x", "y"], "group_options": {"g": 1}, "context_options": {"c": 2}}]},
    {"features": [{"name": "b", "column_index": 0, "feature_group": "SomeGroup"}]},
    {"features": [{"name": "b", "context_options": {"c": 1}, "propagate_context_keys": ["c"]}]},
    {"features": ["x" * MAX_NAME_LENGTH]},
    {"features": [f"f{i}" for i in range(MAX_FEATURES)]},
    {"features": ["a"], "restrict_sources": []},
    {"features": ["a"], "restrict_sources": ["sales"]},
    {"features": ["a"], "filters": []},
    _filter_request(op="eq", value="x"),
    _filter_request(op="eq", value=1.5),
    _filter_request(op="eq", value=True),
    _filter_request(op="in", values=["x", 1, False]),
    _filter_request(op="range", min=1),
    _filter_request(op="range", max=10, max_exclusive=True),
    _filter_request(op="range", min=1, max=10),
    # invalid
    "features",
    ["a"],
    None,
    {},
    {"features": []},
    {"features": "a"},
    {"features": [""]},
    {"features": ["x" * (MAX_NAME_LENGTH + 1)]},
    {"features": [{"name": "x" * (MAX_NAME_LENGTH + 1)}]},
    {"features": [f"f{i}" for i in range(MAX_FEATURES + 1)]},
    {"features": [3]},
    {"features": [None]},
    {"features": [["a"]]},
    {"features": [{"options": {}}]},
    {"features": [{"name": "b", "optons": {}}]},
    {"features": [{"name": 1}]},
    {"features": [{"name": "b", "options": []}]},
    {"features": [{"name": "b", "in_features": "x"}]},
    {"features": [{"name": "b", "in_features": [1]}]},
    {"features": [{"name": "b", "column_index": 1.5}]},
    {"features": [{"name": "b", "column_index": "0"}]},
    {"features": [{"name": "b", "feature_group": ""}]},
    {"features": [{"name": "b", "propagate_context_keys": "c"}]},
    {"features": ["a"], "optons": {}},
    {"features": ["a"], "filters": {}},
    {"features": ["a"], "filters": [_filter_request(op="eq", value=1)["filters"][0]] * (MAX_FILTERS + 1)},
    {"features": ["a"], "filters": ["a"]},
    {"features": ["a"], "filters": [{"op": "eq", "value": 1}]},
    _filter_request(),
    _filter_request(op="regex", value="x"),
    _filter_request(op="eq"),
    _filter_request(op="eq", value=None),
    _filter_request(op="eq", value={"k": 1}),
    _filter_request(op="eq", value=[1]),
    _filter_request(op="in"),
    _filter_request(op="in", values="x"),
    _filter_request(op="in", values=[{"k": 1}]),
    _filter_request(op="range"),
    _filter_request(op="range", max_exclusive=True),
    _filter_request(op="range", min=1, max_exclusive="yes"),
    _filter_request(op="eq", value=1, extra=1),
    {"features": ["a"], "filters": [{"feature": 1, "op": "eq", "value": 1}]},
    {"features": ["a"], "restrict_sources": "sales"},
    {"features": ["a"], "restrict_sources": [1]},
    {"features": ["a"], "restrict_sources": ["sales", "sales"]},
]


@pytest.mark.parametrize("request_value", AGREEMENT_CORPUS, ids=[f"case{i}" for i in range(len(AGREEMENT_CORPUS))])
def test_schema_and_validator_agree(request_value: Any) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(request_schema())
    assert validator.is_valid(request_value) == (validate_request(request_value, SALES_SETTINGS) == [])


def test_agreement_corpus_covers_both_outcomes() -> None:
    outcomes = [validate_request(r, SALES_SETTINGS) == [] for r in AGREEMENT_CORPUS]
    assert sum(outcomes) >= 15
    assert len(outcomes) - sum(outcomes) >= 15


def test_schema_is_a_valid_draft_2020_12_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(request_schema())
