"""Per-feature FeatureGroup scope in the JSON config loader (issue #582).

Feature(name, feature_group=...) can disambiguate a shared column name that two
enabled feature groups both declare. The JSON config schema has no equivalent
field, so config-file users hit "Multiple feature groups found".

These tests pin the config-side field: FeatureConfig.feature_group holds a
class-name STRING (the class-object form stays Python-only), the loader forwards
it to Feature at every construction site, non-string values are rejected with
ValueError, and the schema advertises it.
"""

from typing import Any, Optional

import pytest

from mloda.core.api.feature_config.loader import load_features_from_config, process_nested_features
from mloda.core.api.feature_config.models import FeatureConfig, feature_config_schema
from mloda.core.api.feature_config.parser import parse_json
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


# ---------------------------------------------------------------------------
# Model: FeatureConfig.feature_group
# ---------------------------------------------------------------------------


def test_feature_config_accepts_feature_group_string() -> None:
    """FeatureConfig carries an optional feature_group class-name string."""
    config = FeatureConfig(name="shared_token", feature_group="ConfigScopeSourceA")

    assert config.feature_group == "ConfigScopeSourceA"


def test_feature_config_feature_group_defaults_to_none() -> None:
    """Configs that omit feature_group keep it as None."""
    config = FeatureConfig(name="shared_token")

    assert config.feature_group is None


def test_feature_config_rejects_non_string_feature_group() -> None:
    """A non-string feature_group raises ValueError in the config validation style."""
    bad_value: Any = 123

    with pytest.raises(ValueError):
        FeatureConfig(name="shared_token", feature_group=bad_value)


def test_feature_config_rejects_class_object_feature_group() -> None:
    """The class-object scope form is Python-only and not expressible in a config."""

    class NotJsonExpressible(FeatureGroup):
        """Placeholder feature group used as an invalid config value."""

    bad_value: Any = NotJsonExpressible

    with pytest.raises(ValueError):
        FeatureConfig(name="shared_token", feature_group=bad_value)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_json_populates_feature_group() -> None:
    """parse_json maps the JSON feature_group key onto FeatureConfig."""
    config_str = '[{"name": "shared_token", "feature_group": "ConfigScopeSourceA"}]'

    items = parse_json(config_str)

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, FeatureConfig)
    assert item.feature_group == "ConfigScopeSourceA"


# ---------------------------------------------------------------------------
# Loader: every Feature construction site forwards the scope
# ---------------------------------------------------------------------------


def test_loader_forwards_feature_group_plain_options_branch() -> None:
    """The plain-options branch forwards feature_group to Feature."""
    config_str = """[
        {
            "name": "shared_token",
            "feature_group": "ConfigScopeSourceA",
            "options": {"param": "value"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "ConfigScopeSourceA"
    assert result[0].options.group.get("param") == "value"


def test_loader_forwards_feature_group_in_features_branch() -> None:
    """The in_features branch forwards feature_group to Feature."""
    config_str = """[
        {
            "name": "derived_token",
            "feature_group": "ConfigScopeSourceA",
            "in_features": ["age"],
            "options": {"method": "standard"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "ConfigScopeSourceA"


def test_loader_forwards_feature_group_group_context_options_branch() -> None:
    """The group_options/context_options branch forwards feature_group to Feature."""
    config_str = """[
        {
            "name": "modern_token",
            "feature_group": "ConfigScopeSourceA",
            "group_options": {"threshold": 0.5},
            "context_options": {"metadata": "test"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "ConfigScopeSourceA"
    assert result[0].options.group.get("threshold") == 0.5
    assert result[0].options.context.get("metadata") == "test"


def test_loader_forwards_feature_group_with_column_index() -> None:
    """A scoped feature keeps the tilde column-index name and the scope."""
    config_str = """[
        {
            "name": "shared_token",
            "feature_group": "ConfigScopeSourceA",
            "column_index": 1
        }
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert result[0].name == "shared_token~1"
    assert result[0].feature_group_scope == "ConfigScopeSourceA"


def test_nested_in_features_feature_carries_feature_group() -> None:
    """The nested in_features Feature built by process_nested_features carries the scope."""
    options: dict[str, Any] = {
        "scaler_type": "minmax",
        "in_features": {
            "name": "shared_token",
            "feature_group": "ConfigScopeSourceA",
            "options": {"in_features": "age"},
        },
    }

    processed = process_nested_features(options)

    nested = processed["in_features"]
    assert isinstance(nested, Feature)
    assert nested.name == "shared_token"
    assert nested.feature_group_scope == "ConfigScopeSourceA"


def test_loader_nested_in_features_feature_carries_feature_group() -> None:
    """End of the nested path: the loader keeps the scope on the nested Feature."""
    config_str = """[
        {
            "name": "outer_feature",
            "options": {
                "scaler_type": "minmax",
                "in_features": {
                    "name": "shared_token",
                    "feature_group": "ConfigScopeSourceA",
                    "options": {"in_features": "age"}
                }
            }
        }
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    nested = result[0].options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.feature_group_scope == "ConfigScopeSourceA"


# ---------------------------------------------------------------------------
# Regression: configs without the field are unchanged
# ---------------------------------------------------------------------------


def test_configs_without_feature_group_have_no_scope() -> None:
    """Configs that omit feature_group produce features with no scope in every branch."""
    config_str = """[
        {"name": "plain", "options": {"param": "value"}},
        {"name": "with_in_features", "in_features": ["age"]},
        {
            "name": "with_group_context",
            "group_options": {"threshold": 0.5},
            "context_options": {"metadata": "test"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 3
    for item in result:
        assert isinstance(item, Feature)
        assert item.feature_group_scope is None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_feature_config_schema_includes_feature_group() -> None:
    """The schema advertises feature_group as a string property."""
    schema = feature_config_schema()

    assert "feature_group" in schema["properties"], "'feature_group' should be in schema properties"
    assert schema["properties"]["feature_group"]["type"] == "string"


# ---------------------------------------------------------------------------
# End to end: two enabled sources declaring the same shared column name
# ---------------------------------------------------------------------------


class ConfigScopeSourceA(FeatureGroup):
    """Source A: provides the shared "config_shared_token" plus "config_value_a"."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"config_shared_token", "config_value_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "config_shared_token": ["a1", "a2"],
            "config_value_a": [1, 2],
        }


class ConfigScopeSourceB(FeatureGroup):
    """Source B: also provides the shared "config_shared_token" plus "config_value_b"."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"config_shared_token", "config_value_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "config_shared_token": ["b1", "b2"],
            "config_value_b": [3, 4],
        }


def _run(config_str: str) -> list[Any]:
    features = load_features_from_config(config_str, format="json")
    return list(
        mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=PluginCollector.enabled_feature_groups({ConfigScopeSourceA, ConfigScopeSourceB}),
        )
    )


def test_end2end_unscoped_config_feature_is_ambiguous() -> None:
    """Characterization: the bare shared name without feature_group stays ambiguous."""
    config_str = '[{"name": "config_shared_token"}]'

    with pytest.raises(ValueError, match="Multiple feature groups found"):
        _run(config_str)


def test_end2end_scoped_config_feature_resolves_to_source_a() -> None:
    """The same config with feature_group resolves uniquely and returns Source A's data."""
    config_str = '[{"name": "config_shared_token", "feature_group": "ConfigScopeSourceA"}]'

    results = _run(config_str)

    values: list[str] = []
    for df in results:
        if "config_shared_token" in df.columns:
            values = list(df["config_shared_token"].values)

    assert values == ["a1", "a2"]


def test_end2end_scoped_config_feature_resolves_to_source_b() -> None:
    """Scoping the same config name to Source B returns Source B's data instead."""
    config_str = '[{"name": "config_shared_token", "feature_group": "ConfigScopeSourceB"}]'

    results = _run(config_str)

    values: list[str] = []
    for df in results:
        if "config_shared_token" in df.columns:
            values = list(df["config_shared_token"].values)

    assert values == ["b1", "b2"]
