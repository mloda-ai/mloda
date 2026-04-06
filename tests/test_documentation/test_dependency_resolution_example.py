"""Tests for the automatic dependency resolution example in README.md.

These tests validate the runnable code example that demonstrates mloda's
automatic dependency resolution: requesting only the final feature in a
chain and letting mloda resolve all intermediate steps.

Chain: api_data("text") -> NormalizedText -> WordCount
"""

from typing import Any

from mloda.provider import ApiInputDataFeature, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class NormalizedText(FeatureChainParserMixin, FeatureGroup):
    """Normalize text: lowercase and strip whitespace."""

    PREFIX_PATTERN = r".*__(normalized)$"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source = cls._extract_source_features(feature)[0]
            data[feature.name] = data[source].str.lower().str.strip()
        return data


class WordCount(FeatureChainParserMixin, FeatureGroup):
    """Count words in text."""

    PREFIX_PATTERN = r".*__(word_count)$"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source = cls._extract_source_features(feature)[0]
            data[feature.name] = data[source].str.split().str.len()
        return data


_PLUGIN_COLLECTOR = PluginCollector.enabled_feature_groups({ApiInputDataFeature, NormalizedText, WordCount})

_API_DATA: dict[str, dict[str, list[str]]] = {"TextData": {"text": ["  Hello World  ", "FOO BAR BAZ", "Test  "]}}


class TestDependencyResolutionExample:
    """Validate the automatic dependency resolution example from README.md."""

    def test_full_chain_text_normalized_word_count(self) -> None:
        """User requests only the final feature; mloda resolves the 3-step chain."""
        result = mloda.run_all(
            features=["text__normalized__word_count"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGIN_COLLECTOR,
            api_data=_API_DATA,
        )

        assert len(result) == 1
        df = result[0]
        assert list(df["text__normalized__word_count"]) == [2, 3, 1]

    def test_intermediate_step_normalized(self) -> None:
        """Requesting just the intermediate normalized step also works."""
        result = mloda.run_all(
            features=["text__normalized"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGIN_COLLECTOR,
            api_data=_API_DATA,
        )

        assert len(result) == 1
        df = result[0]
        assert list(df["text__normalized"]) == ["hello world", "foo bar baz", "test"]

    def test_single_step_word_count(self) -> None:
        """WordCount applied directly to raw data (single step, no chaining)."""
        result = mloda.run_all(
            features=["text__word_count"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGIN_COLLECTOR,
            api_data={"TextData": {"text": ["hello world", "foo bar baz", "test"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert list(df["text__word_count"]) == [2, 3, 1]
