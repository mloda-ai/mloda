"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any

from mloda_core.abstract_plugins.components.feature import Feature

from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from tests.test_plugins.integration_plugins.chainer.chainer_context_feature import ChainedContextFeatureGroupTest


class ChainedFeatureGroupTest(ChainedContextFeatureGroupTest):
    @classmethod
    def perform_operation(cls, data: Any, feature: Feature) -> Any:
        """
        PERFORM OPERATION HAS 3 STEPS:
        1. Check if the source feature exists in the data.
        2. Perform the operation (e.g., doubling the values).
        3. Add the result to the data with a new feature name.
        """

        has_prefix_configuration, source_feature = FeatureChainParser.parse_feature_name(
            feature.name, cls.PATTERN, cls.PREFIX_PATTERN
        )

        if has_prefix_configuration is None or source_feature is None:
            raise ValueError(f"Could not parse feature name: {feature.name}")

        if source_feature not in data.columns:
            raise ValueError(f"Source feature '{source_feature}' not found in data.")

        if has_prefix_configuration == "identifier1":
            val = 0
        elif has_prefix_configuration == "identifier2":
            val = 1
        else:
            raise ValueError("Invalid prefix configuration")

        data[f"{has_prefix_configuration}{cls.PATTERN}{source_feature}"] = data[source_feature] * 2 + val
        return data


class ChainedFeatureGroupTest_B(ChainedFeatureGroupTest):
    PATTERN = "_chainer_b__"
    PREFIX_PATTERN = [r"^([\w]+)_chainer_b__"]
