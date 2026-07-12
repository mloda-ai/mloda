"""Per-feature FeatureGroup scope in the JSON config loader (issue #582).

Feature(name, feature_group=...) can disambiguate a shared column name that two
enabled feature groups both declare. The JSON config schema has no equivalent
field, so config-file users hit "Multiple feature groups found".

These tests pin the config-side field: FeatureConfig.feature_group holds a
non-empty class-name STRING (the class-object form stays Python-only), the loader
forwards it to Feature at every construction site (including the nested
in_features path, which bypasses FeatureConfig validation), invalid values are
rejected with ValueError, and the schema advertises it.
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
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup


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


def test_feature_config_rejects_empty_feature_group() -> None:
    """An empty feature_group is a config mistake, not a silent "no scope"."""
    with pytest.raises(ValueError, match="feature_group"):
        FeatureConfig(name="shared_token", feature_group="")


def test_feature_config_rejects_whitespace_only_feature_group() -> None:
    """A whitespace-only feature_group strips to nothing and must be rejected too."""
    with pytest.raises(ValueError, match="feature_group"):
        FeatureConfig(name="shared_token", feature_group="   ")


def test_loader_rejects_empty_feature_group() -> None:
    """The loader rejects an empty feature_group instead of dropping the scope."""
    config_str = '[{"name": "shared_token", "feature_group": ""}]'

    with pytest.raises(ValueError, match="feature_group"):
        load_features_from_config(config_str)


def test_loader_rejects_whitespace_only_feature_group() -> None:
    """The loader rejects a whitespace-only feature_group instead of dropping the scope."""
    config_str = '[{"name": "shared_token", "feature_group": "   "}]'

    with pytest.raises(ValueError, match="feature_group"):
        load_features_from_config(config_str)


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
# Loader: the nested in_features dict is validated like a FeatureConfig
#
# The nested dict never passes through FeatureConfig.__post_init__, so its
# feature_group must be validated on the nested path too. Otherwise Feature()
# raises a raw TypeError ("feature_group must be a FeatureGroup subclass, ...")
# and a config-file user gets a Python-API error out of a config error.
# ---------------------------------------------------------------------------

CONFIG_STYLE_MESSAGE = "must be a feature group class-name string"


def test_nested_in_features_rejects_non_string_feature_group() -> None:
    """A non-string nested feature_group raises the config-style ValueError, not a TypeError."""
    options: dict[str, Any] = {
        "scaler_type": "minmax",
        "in_features": {
            "name": "shared_token",
            "feature_group": 123,
            "options": {"in_features": "age"},
        },
    }

    with pytest.raises(ValueError, match=CONFIG_STYLE_MESSAGE):
        process_nested_features(options)


def test_nested_in_features_rejects_class_object_feature_group() -> None:
    """The class-object scope form stays Python-only on the nested config path as well."""

    class NotJsonExpressibleNested(FeatureGroup):
        """Placeholder feature group used as an invalid nested config value."""

    options: dict[str, Any] = {
        "in_features": {
            "name": "shared_token",
            "feature_group": NotJsonExpressibleNested,
        },
    }

    with pytest.raises(ValueError, match=CONFIG_STYLE_MESSAGE):
        process_nested_features(options)


def test_loader_nested_in_features_rejects_non_string_feature_group() -> None:
    """The same nested rejection surfaces through load_features_from_config on a JSON string."""
    config_str = """[
        {
            "name": "outer_feature",
            "options": {
                "scaler_type": "minmax",
                "in_features": {
                    "name": "shared_token",
                    "feature_group": 123,
                    "options": {"in_features": "age"}
                }
            }
        }
    ]"""

    with pytest.raises(ValueError, match=CONFIG_STYLE_MESSAGE):
        load_features_from_config(config_str)


def test_nested_in_features_rejects_empty_feature_group() -> None:
    """An empty nested feature_group is rejected instead of silently stripped to None."""
    options: dict[str, Any] = {
        "in_features": {
            "name": "shared_token",
            "feature_group": "",
        },
    }

    with pytest.raises(ValueError, match="feature_group"):
        process_nested_features(options)


def test_loader_nested_in_features_rejects_whitespace_only_feature_group() -> None:
    """A whitespace-only nested feature_group is rejected through the loader too."""
    config_str = """[
        {
            "name": "outer_feature",
            "options": {
                "in_features": {
                    "name": "shared_token",
                    "feature_group": "   "
                }
            }
        }
    ]"""

    with pytest.raises(ValueError, match="feature_group"):
        load_features_from_config(config_str)


# ---------------------------------------------------------------------------
# The root FeatureGroup base name is rejected by the config layer (issue #682)
#
# A string scope matches by ancestry, so the root base name would scope to every
# candidate. Feature() rejects it with a TypeError, but a config value must fail
# as a config error: ValueError from the config validation, like every other
# invalid scope value, on the top-level path AND the nested in_features path.
# ---------------------------------------------------------------------------

ROOT_BASE_NAME = FeatureGroup.get_class_name()


def test_feature_config_rejects_root_feature_group_base_name() -> None:
    """The root base name is a wildcard scope and must be rejected by FeatureConfig."""
    with pytest.raises(ValueError, match="feature_group"):
        FeatureConfig(name="shared_token", feature_group=ROOT_BASE_NAME)


def test_loader_rejects_root_feature_group_base_name() -> None:
    """The loader rejects the root base name with a config ValueError, not a Python TypeError."""
    config_str = '[{"name": "shared_token", "feature_group": "FeatureGroup"}]'

    with pytest.raises(ValueError, match="feature_group"):
        load_features_from_config(config_str)


def test_nested_in_features_rejects_root_feature_group_base_name() -> None:
    """The nested path bypasses FeatureConfig, so it must reject the root base name itself."""
    options: dict[str, Any] = {
        "in_features": {
            "name": "shared_token",
            "feature_group": ROOT_BASE_NAME,
        },
    }

    with pytest.raises(ValueError, match="feature_group"):
        process_nested_features(options)


def test_loader_nested_in_features_rejects_root_feature_group_base_name() -> None:
    """The same nested rejection surfaces through load_features_from_config on a JSON string."""
    config_str = """[
        {
            "name": "outer_feature",
            "options": {
                "in_features": {
                    "name": "shared_token",
                    "feature_group": "FeatureGroup"
                }
            }
        }
    ]"""

    with pytest.raises(ValueError, match="feature_group"):
        load_features_from_config(config_str)


# ---------------------------------------------------------------------------
# Misplaced scope: "feature_group" inside an option container is a hard error
#
# The scope is a TOP-LEVEL field. Written inside "options" (or the group/context
# variants) it is silently swallowed as an ordinary option: feature_group_scope
# stays None, the key poisons the option hash and the matcher inputs, and
# resolution still fails with "Multiple feature groups found" without ever
# naming the key that was ignored. JSON is the documented surface for AI agents,
# so this misplacement is likely and must fail loudly at validation time.
#
# Only TOP-LEVEL keys of each container are checked: a nested scope legitimately
# lives at options["in_features"]["feature_group"], which is a dict VALUE.
# ---------------------------------------------------------------------------

MISPLACED_KEY_TERMS = ("feature_group", "top-level")


def test_feature_config_rejects_feature_group_key_inside_options() -> None:
    """A feature_group key inside 'options' is a misplaced scope and raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="shared_token", options={"feature_group": "ConfigScopeSourceA"})

    message = str(exc_info.value)
    for term in MISPLACED_KEY_TERMS:
        assert term in message


def test_feature_config_rejects_feature_group_key_inside_group_options() -> None:
    """A feature_group key inside 'group_options' is a misplaced scope and raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="shared_token", group_options={"feature_group": "ConfigScopeSourceA"})

    message = str(exc_info.value)
    for term in MISPLACED_KEY_TERMS:
        assert term in message


def test_feature_config_rejects_feature_group_key_inside_context_options() -> None:
    """A feature_group key inside 'context_options' is a misplaced scope and raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="shared_token", context_options={"feature_group": "ConfigScopeSourceA"})

    message = str(exc_info.value)
    for term in MISPLACED_KEY_TERMS:
        assert term in message


def test_loader_rejects_misplaced_feature_group_key_in_options() -> None:
    """Through the JSON surface, a scope written inside 'options' fails loudly instead of becoming an option."""
    config_str = '[{"name": "shared_token", "options": {"feature_group": "ConfigScopeSourceA"}}]'

    with pytest.raises(ValueError) as exc_info:
        load_features_from_config(config_str, format="json")

    message = str(exc_info.value)
    for term in MISPLACED_KEY_TERMS:
        assert term in message


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


def test_feature_config_schema_feature_group_is_non_empty() -> None:
    """The schema states the non-empty constraint the validation enforces."""
    schema = feature_config_schema()

    assert schema["properties"]["feature_group"]["minLength"] == 1


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

    assert len(results) == 1
    df = results[0]
    assert "config_shared_token" in df.columns
    assert list(df["config_shared_token"].values) == ["a1", "a2"]


def test_end2end_scoped_config_feature_resolves_to_source_b() -> None:
    """Scoping the same config name to Source B returns Source B's data instead."""
    config_str = '[{"name": "config_shared_token", "feature_group": "ConfigScopeSourceB"}]'

    results = _run(config_str)

    assert len(results) == 1
    df = results[0]
    assert "config_shared_token" in df.columns
    assert list(df["config_shared_token"].values) == ["b1", "b2"]


# ---------------------------------------------------------------------------
# End to end: a config names an ABSTRACT family base and reaches the concrete
# per-framework subclass (issue #682)
#
# A config can only carry a class-name STRING. Naming the framework-agnostic
# family base "AggregatedFeatureGroup" must resolve to the concrete
# PandasAggregatedFeatureGroup for the active compute framework, so the config
# stays free of a compute-framework-specific leaf class name.
# ---------------------------------------------------------------------------


class ConfigScopeAggregationSource(FeatureGroup):
    """Source data for the aggregated-family scope test."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"config_scope_sales"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"config_scope_sales": [10, 20, 30, 40]}


def test_end2end_config_abstract_family_base_scope_resolves_to_pandas_subclass() -> None:
    """A config scoped to 'AggregatedFeatureGroup' runs on the Pandas subclass.

    The allowlist keeps the whole family accessible, which is the situation a real
    config user is in: the scoped abstract base itself, plus a rival concrete
    sibling for another framework. The base drops out because it cannot be
    instantiated, the PyArrow sibling because its framework is not enabled.
    """
    config_str = '[{"name": "config_scope_sales__mean_aggr", "feature_group": "AggregatedFeatureGroup"}]'

    features = load_features_from_config(config_str, format="json")
    results = list(
        mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    ConfigScopeAggregationSource,
                    AggregatedFeatureGroup,
                    PandasAggregatedFeatureGroup,
                    PyArrowAggregatedFeatureGroup,
                }
            ),
        )
    )

    aggregated = [df for df in results if "config_scope_sales__mean_aggr" in df.columns]
    assert len(aggregated) == 1
    assert aggregated[0]["config_scope_sales__mean_aggr"].iloc[0] == 25.0
