"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any

from mloda import Feature

from mloda.provider import FeatureChainParser
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

        has_suffix_configuration, source_feature = FeatureChainParser.parse_feature_name(
            feature.name, cls.SUFFIX_PATTERN
        )

        if has_suffix_configuration is None or source_feature is None:
            raise ValueError(f"Could not parse feature name: {feature.name}")

        if source_feature not in data.columns:
            raise ValueError(f"Source feature '{source_feature}' not found in data.")

        if has_suffix_configuration == "identifier1":
            val = 0
        elif has_suffix_configuration == "identifier2":
            val = 1
        else:
            raise ValueError("Invalid suffix configuration")

        data[f"{source_feature}__{cls.OPERATION_ID}{has_suffix_configuration}"] = data[source_feature] * 2 + val
        return data


class ChainedFeatureGroupTest_B(ChainedFeatureGroupTest):
    SUFFIX_PATTERN = [r".*__chainer_b_([\w]+)$"]
    OPERATION_ID = "chainer_b_"  # Used for constructing feature names
