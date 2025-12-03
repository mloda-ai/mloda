"""
Test Suite for Missing Links Validation

This test verifies that the framework provides a helpful error message when a user
creates a feature with multiple dependencies but forgets to provide explicit Links.

Expected Behavior:
When a feature declares multiple input_features() but no Links are provided,
the Engine should raise a clear ValueError during initialization (not at runtime)
with guidance on how to fix the issue.

This prevents confusing KeyError messages at runtime and educates users about
the requirement for explicit Links when merging multiple dependencies.
"""

from typing import Any, Optional, Set

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# Test Feature Groups
class RootFeatureA(AbstractFeatureGroup):
    """First root feature for testing"""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], cls.get_class_name(): [10, 20, 30]}


class RootFeatureB(AbstractFeatureGroup):
    """Second root feature for testing"""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], cls.get_class_name(): [100, 200, 300]}


class MultiDependencyFeature(AbstractFeatureGroup):
    """
    Feature with multiple dependencies - requires explicit Links to merge them.
    This will trigger the validation error when Links are not provided.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature.int32_of("RootFeatureA"),
            Feature.int32_of("RootFeatureB"),
        }

    @classmethod
    def compute_framework_rule(cls) -> bool:
        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # This would fail at runtime if Links are missing
        col_a = data.column("RootFeatureA")
        col_b = data.column("RootFeatureB")
        return {cls.get_class_name(): pc.add(col_a, col_b)}


class TestMissingLinksError:
    """Test suite for missing Links validation"""

    def test_missing_links_raises_helpful_error(self) -> None:
        """
        Test that a helpful error is raised when a feature has multiple dependencies
        but no Links are provided.

        Expected Error Location: During runtime
        Expected Error Type: Exception (wraps ValueError)
        Expected Error Content:
            - Mentions "Links" or "multiple dependencies"
            - Lists the feature name (MultiDependencyFeature)
            - Provides example code with Link.inner()
            - Shows how to use Index()
            - Lists available join types
        """
        with pytest.raises(Exception) as exc_info:
            mlodaAPI.run_all(
                features=[Feature.int32_of("MultiDependencyFeature")],
                links=set(),  # EMPTY - this should trigger the error
                compute_frameworks={PyArrowTable},
                plugin_collector=PlugInCollector.enabled_feature_groups(
                    {RootFeatureA, RootFeatureB, MultiDependencyFeature}
                ),
            )

        error_message = str(exc_info.value)

        # Verify error message contains helpful information
        assert "MultiDependencyFeature" in error_message, "Error should mention the feature with missing Links"

        # Check for guidance keywords
        assert any(keyword in error_message.lower() for keyword in ["link", "multiple", "dependencies"]), (
            "Error should mention Links or dependencies"
        )

        # Check for example code
        assert "Link.inner" in error_message, "Error should show Link.inner() example"
        assert "Index" in error_message, "Error should show Index usage"

        # Check for join type documentation
        assert any(join_type in error_message for join_type in ["inner", "left", "right", "outer"]), (
            "Error should list available join types"
        )
