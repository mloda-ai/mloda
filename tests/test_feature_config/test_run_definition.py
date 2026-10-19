"""Regression tests for run definition load/dump (issue #2003)."""

import json

import pytest

from mloda.core.api.feature_config.loader import (
    dump_run_definition,
    load_features_from_config,
    load_run_definition,
)
from mloda.core.abstract_plugins.components.feature import Feature


def test_load_run_definition_object_with_features_and_links() -> None:
    config = json.dumps(
        {
            "format": 1,
            "features": ["a", {"name": "b", "context_options": {"k": "v"}}],
            "links": [{"source": "a", "target": "b"}],
        }
    )
    features, links = load_run_definition(config)
    assert len(features) == 2
    assert features[0] == "a"
    assert isinstance(features[1], Feature)
    assert features[1].name == "b"
    assert links == [{"source": "a", "target": "b"}]


def test_load_run_definition_bare_array_still_works() -> None:
    config = json.dumps(["a", "b"])
    features, links = load_run_definition(config)
    assert features == ["a", "b"]
    assert links == []


def test_load_run_definition_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="Unsupported run definition format version"):
        load_run_definition(json.dumps({"format": 2, "features": []}))


def test_load_run_definition_rejects_non_list_links() -> None:
    with pytest.raises(ValueError, match="'links' must be a JSON array"):
        load_run_definition(json.dumps({"format": 1, "features": [], "links": {}}))


def test_load_run_definition_rejects_non_list_features() -> None:
    with pytest.raises(ValueError, match="'features' must be a JSON array"):
        load_run_definition(json.dumps({"format": 1, "features": {}}))


def test_dump_run_definition_round_trips() -> None:
    features = load_features_from_config(json.dumps(["a", {"name": "b", "context_options": {"k": "v"}}]))
    links = [{"source": "a", "target": "b"}]
    dumped = dump_run_definition(features, links)
    data = json.loads(dumped)
    assert data["format"] == 1
    assert data["links"] == links
    assert data["features"][0] == "a"
    assert data["features"][1]["name"] == "b"
    assert data["features"][1]["context_options"] == {"k": "v"}

    reloaded_features, reloaded_links = load_run_definition(dumped)
    assert reloaded_links == links
    assert len(reloaded_features) == 2
    assert reloaded_features[0] == "a"
    assert isinstance(reloaded_features[1], Feature)
    assert reloaded_features[1].name == "b"


def test_dump_run_definition_without_links() -> None:
    dumped = dump_run_definition(["a"])
    data = json.loads(dumped)
    assert data["links"] == []
    assert data["features"] == ["a"]


def test_dump_run_definition_keeps_group_options_and_in_features() -> None:
    config = json.dumps(
        [{"name": "b", "in_features": ["a"], "group_options": {"window": 3}, "context_options": {"k": "v"}}]
    )
    features = load_features_from_config(config)
    dumped = dump_run_definition(features)

    item = json.loads(dumped)["features"][0]
    assert item["in_features"] == ["a"]
    assert item["group_options"] == {"window": 3}
    assert item["context_options"] == {"k": "v"}

    reloaded = load_features_from_config(json.dumps([item]))[0]
    assert reloaded.options.group == features[0].options.group
    assert reloaded.options.context == features[0].options.context


def test_dump_run_definition_writes_feature_group_by_class_name() -> None:
    dumped = dump_run_definition([Feature(name="b", feature_group="Customers")])
    assert json.loads(dumped)["features"][0]["feature_group"] == "Customers"


def test_dump_run_definition_rejects_options_that_cannot_be_written() -> None:
    # A Feature object inside the options (a nested feature) has no JSON form.
    features = [Feature(name="b", options={"nested": Feature(name="a")})]
    with pytest.raises(ValueError, match="not JSON-serialisable"):
        dump_run_definition(features)
