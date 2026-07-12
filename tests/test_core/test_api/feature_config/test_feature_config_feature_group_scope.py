"""Config-loader support for the per-feature FeatureGroup resolution scope (issue #582).

PR #575 (issue #508) added a resolution-only scope on ``Feature``:

    Feature("subject_token", feature_group=SomeSource)
    Feature("subject_token", feature_group="SomeSource")

It is reachable only through the Python constructor. These tests pin the JSON
config equivalent: a ``feature_group`` field on ``FeatureConfig`` (STRING form
only, because JSON cannot carry a class object) that
``load_features_from_config`` forwards to ``Feature(feature_group=...)`` on every
branch, so ``feature.feature_group_scope`` carries the scope.

Normalization stays owned by ``Feature._set_feature_group_scope``: an empty or
whitespace-only string becomes None, a padded name is stripped.
"""

import json
from typing import Any, Optional

import pytest

from mloda.core.api.feature_config.loader import load_features_from_config, process_nested_features
from mloda.core.api.feature_config.models import FeatureConfig
from mloda.core.api.feature_config.parser import parse_json
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class Source582A(FeatureGroup):
    """Source A: provides the shared "subject_token" plus "scoped_value_a"."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "scoped_value_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2", "s3", "s4"],
            "scoped_value_a": [10, 20, 30, 40],
        }


class Source582B(FeatureGroup):
    """Source B: also provides the shared "subject_token" plus "scoped_value_b"."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "scoped_value_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["b1", "b2"],
            "scoped_value_b": [1, 2],
        }


# ---------------------------------------------------------------------------
# Model contract: FeatureConfig carries an optional string feature_group
# ---------------------------------------------------------------------------


def test_feature_config_feature_group_defaults_to_none() -> None:
    """Without the field, FeatureConfig.feature_group is None."""
    config = FeatureConfig(name="subject_token")

    assert config.feature_group is None


def test_feature_config_accepts_feature_group_string() -> None:
    """A class-name string is accepted and stored on FeatureConfig.feature_group."""
    config = FeatureConfig(name="subject_token", feature_group="Source582A")

    assert config.feature_group == "Source582A"


def test_feature_config_rejects_non_string_feature_group() -> None:
    """A non-string, non-None feature_group raises ValueError (config validation style)."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="subject_token", feature_group=123)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "feature_group" in message
    # Guard against a false pass while the field does not yet exist: the current
    # "unexpected keyword argument 'feature_group'" TypeError must NOT satisfy this.
    assert "unexpected keyword" not in message


def test_feature_config_rejects_feature_group_class_object() -> None:
    """The class-object form stays Python-only: a class in config raises ValueError.

    JSON cannot carry a class, so the config field is string-only. Passing a
    FeatureGroup class through the dataclass must be rejected loudly rather than
    creating a config that no config file could ever express.
    """
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="subject_token", feature_group=Source582A)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "feature_group" in message
    assert "unexpected keyword" not in message


# ---------------------------------------------------------------------------
# Parser contract: the JSON "feature_group" key survives parse_json
# ---------------------------------------------------------------------------


def test_parse_json_accepts_feature_group_field() -> None:
    """parse_json maps the JSON "feature_group" key onto FeatureConfig.feature_group."""
    config_str = """[
        {"name": "subject_token", "feature_group": "Source582A"}
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].feature_group == "Source582A"


def test_parse_json_without_feature_group_keeps_none() -> None:
    """A config item without the field parses to feature_group None (no regression)."""
    config_str = """[
        {"name": "subject_token", "options": {"param": "value"}}
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].feature_group is None


# ---------------------------------------------------------------------------
# Loader contract: the scope is forwarded on EVERY branch of the loader
# ---------------------------------------------------------------------------


def test_loader_forwards_scope_on_plain_options_branch() -> None:
    """The plain 'options' branch forwards feature_group to Feature.feature_group_scope."""
    config_str = """[
        {"name": "subject_token", "options": {"param": "value"}, "feature_group": "Source582A"}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "Source582A"
    assert result[0].options.group.get("param") == "value"


def test_loader_forwards_scope_on_in_features_branch() -> None:
    """The 'in_features' branch forwards feature_group to Feature.feature_group_scope."""
    config_str = """[
        {
            "name": "scale__subject_token",
            "in_features": ["subject_token"],
            "options": {"method": "standard"},
            "feature_group": "Source582A"
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "Source582A"
    assert result[0].options.context.get(DefaultOptionKeys.in_features) == frozenset({"subject_token"})


def test_loader_forwards_scope_on_group_context_options_branch() -> None:
    """The 'group_options'/'context_options' branch forwards feature_group too."""
    config_str = """[
        {
            "name": "subject_token",
            "group_options": {"threshold": 0.5},
            "context_options": {"metadata": "test"},
            "feature_group": "Source582A"
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "Source582A"
    assert result[0].options.group.get("threshold") == 0.5
    assert result[0].options.context.get("metadata") == "test"


def test_loader_forwards_scope_with_column_index() -> None:
    """The scope survives the column_index tilde-suffix path."""
    config_str = """[
        {"name": "subject_token", "column_index": 1, "feature_group": "Source582A"}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name == "subject_token~1"
    assert result[0].feature_group_scope == "Source582A"


def test_process_nested_features_forwards_scope_to_nested_feature() -> None:
    """A nested in_features dict scopes its OWN Feature via its own "feature_group" key."""
    options: dict[str, Any] = {
        "in_features": {"name": "subject_token", "feature_group": "Source582A"},
    }

    processed = process_nested_features(options)

    nested = processed["in_features"]
    assert isinstance(nested, Feature)
    assert nested.name == "subject_token"
    assert nested.feature_group_scope == "Source582A"


def test_loader_forwards_scope_to_nested_in_features_feature() -> None:
    """Through the loader, a nested in_features Feature carries the nested dict's scope."""
    config_str = """[
        {
            "name": "outer_feature",
            "options": {
                "in_features": {"name": "subject_token", "feature_group": "Source582A"}
            }
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    nested = result[0].options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.feature_group_scope == "Source582A"


def test_loader_scope_does_not_pollute_options() -> None:
    """The scope is resolution-only: it must not leak into options.group or options.context."""
    config_str = """[
        {"name": "subject_token", "options": {"param": "value"}, "feature_group": "Source582A"}
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert "feature_group" not in result[0].options.group
    assert "feature_group" not in result[0].options.context


def test_loader_without_feature_group_keeps_scope_none() -> None:
    """No regression: configs without the field produce features with scope None on all branches."""
    config_str = """[
        "plain_string",
        {"name": "plain_options", "options": {"param": "value"}},
        {"name": "with_in_features", "in_features": ["subject_token"]},
        {"name": "with_group_context", "group_options": {"a": 1}, "context_options": {"b": 2}}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 4
    for item in result[1:]:
        assert isinstance(item, Feature)
        assert item.feature_group_scope is None


def test_loader_empty_feature_group_string_normalizes_to_none() -> None:
    """An empty or whitespace-only scope normalizes to None, exactly as Feature does."""
    config_str = """[
        {"name": "empty_scope", "feature_group": ""},
        {"name": "blank_scope", "feature_group": "   "}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2
    for item in result:
        assert isinstance(item, Feature)
        assert item.feature_group_scope is None


def test_loader_padded_feature_group_string_is_stripped() -> None:
    """A padded class-name scope is stored stripped, exactly as Feature does."""
    config_str = """[
        {"name": "subject_token", "feature_group": " Source582A "}
    ]"""

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert result[0].feature_group_scope == "Source582A"


# ---------------------------------------------------------------------------
# End to end through mloda.run_all: the config field disambiguates a shared key
# ---------------------------------------------------------------------------


def test_config_without_feature_group_is_ambiguous() -> None:
    """Characterization: a config request for the shared key is ambiguous without a scope."""
    config_str = json.dumps([{"name": "subject_token"}])

    features = load_features_from_config(config_str, format="json")

    with pytest.raises(ValueError, match="Multiple feature groups found"):
        mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=PluginCollector.enabled_feature_groups({Source582A, Source582B}),
        )


def test_config_with_feature_group_resolves_to_scoped_source() -> None:
    """The config scope resolves the shared key to exactly one source, and run_all returns its data."""
    config_str = json.dumps([{"name": "subject_token", "feature_group": Source582A.get_class_name()}])

    features = load_features_from_config(config_str, format="json")

    results = mloda.run_all(
        features,
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.enabled_feature_groups({Source582A, Source582B}),
    )

    values: list[str] = []
    for result in results:
        if "subject_token" in result.columns:
            values = list(result["subject_token"])

    assert values == ["s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Misplaced scope: "feature_group" inside an option container is a hard error
#
# The scope is a TOP-LEVEL field. Writing it inside "options" (or the
# group/context variants) instead silently turns it into an ordinary option:
# the scope stays None, the key poisons the option hash, and resolution still
# fails with "Multiple feature groups found" without ever mentioning the key
# that was ignored. JSON is the documented surface for AI agents, so this
# misplacement is likely and must fail loudly at config-validation time.
# ---------------------------------------------------------------------------


def test_feature_config_rejects_feature_group_key_inside_options() -> None:
    """A "feature_group" key inside 'options' is a misplaced scope and raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="subject_token", options={"feature_group": "Source582A"})

    message = str(exc_info.value)
    assert "feature_group" in message
    assert "top-level" in message


def test_feature_config_rejects_feature_group_key_inside_group_options() -> None:
    """A "feature_group" key inside 'group_options' is a misplaced scope and raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="subject_token", group_options={"feature_group": "Source582A"})

    message = str(exc_info.value)
    assert "feature_group" in message
    assert "top-level" in message


def test_feature_config_rejects_feature_group_key_inside_context_options() -> None:
    """A "feature_group" key inside 'context_options' is a misplaced scope and raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="subject_token", context_options={"feature_group": "Source582A"})

    message = str(exc_info.value)
    assert "feature_group" in message
    assert "top-level" in message


def test_loader_rejects_misplaced_feature_group_key_in_options() -> None:
    """Through the JSON surface, a scope written inside 'options' fails loudly instead of becoming an option."""
    config_str = json.dumps([{"name": "subject_token", "options": {"feature_group": "Source582A"}}])

    with pytest.raises(ValueError) as exc_info:
        load_features_from_config(config_str, format="json")

    message = str(exc_info.value)
    assert "feature_group" in message
    assert "top-level" in message


# ---------------------------------------------------------------------------
# Nested in_features: validation errors follow the CONFIG style
#
# The top-level field raises a config-style ValueError for a non-string scope.
# A nested in_features dict must not fall through to the Python-surface
# TypeError raised by Feature; the config surface is one surface.
# ---------------------------------------------------------------------------


def test_process_nested_features_rejects_non_string_scope() -> None:
    """A non-string "feature_group" in a nested in_features dict raises config-style ValueError."""
    options: dict[str, Any] = {
        "in_features": {"name": "subject_token", "feature_group": 123},
    }

    with pytest.raises(ValueError) as exc_info:
        process_nested_features(options)

    assert "feature_group" in str(exc_info.value)


def test_loader_rejects_non_string_scope_in_nested_in_features() -> None:
    """Through the loader, a non-string nested scope raises ValueError, not the Python-surface TypeError."""
    config_str = json.dumps(
        [
            {
                "name": "outer_feature",
                "options": {"in_features": {"name": "subject_token", "feature_group": 123}},
            }
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        load_features_from_config(config_str, format="json")

    assert "feature_group" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Characterization: one config array cannot request the same name twice with
# two different scopes. The scope is excluded from Feature identity, so the two
# entries collide in the request-level Features validation.
# ---------------------------------------------------------------------------


def test_config_same_name_two_scopes_raises_duplicate_feature_setup() -> None:
    """Characterization: the same name under two scopes in one config array is a duplicate.

    Mirrors the Python-side pin in
    tests/test_plugins/compute_framework/test_per_feature_fg_scope_integration.py.
    Scope is not part of Feature identity, so the two features compare equal and
    Features raises "Duplicate feature setup". Reading one column from two
    sources needs two separate requests, not two array entries.
    """
    config_str = json.dumps(
        [
            {"name": "subject_token", "feature_group": Source582A.get_class_name()},
            {"name": "subject_token", "feature_group": Source582B.get_class_name()},
        ]
    )

    features = load_features_from_config(config_str, format="json")

    with pytest.raises(ValueError, match="Duplicate feature setup"):
        mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=PluginCollector.enabled_feature_groups({Source582A, Source582B}),
        )
