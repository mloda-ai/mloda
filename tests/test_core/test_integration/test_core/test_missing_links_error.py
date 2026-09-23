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

from typing import Any

import pyarrow.compute as pc
import pytest

from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# Test Feature Groups
class RootFeatureA(FeatureGroup):
    """First root feature for testing"""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], cls.get_class_name(): [10, 20, 30]}


class RootFeatureB(FeatureGroup):
    """Second root feature for testing"""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], cls.get_class_name(): [100, 200, 300]}


class MultiDependencyFeature(FeatureGroup):
    """
    Feature with multiple dependencies - requires explicit Links to merge them.
    This will trigger the validation error when Links are not provided.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature.int32_of("RootFeatureA"),
            Feature.int32_of("RootFeatureB"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # This would fail at runtime if Links are missing
        col_a = data.column("RootFeatureA")
        col_b = data.column("RootFeatureB")
        return {cls.get_class_name(): pc.add(col_a, col_b)}


class SplitMetricSource(FeatureGroup):
    """Root feature group producing two columns from one DataCreator; requested with
    differing Options it plans as two separate option buckets."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"metric_a", "metric_b"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = {"metric_a": [10, 20, 30], "metric_b": [100, 200, 300]}
        return {"id": [1, 2, 3], **{name: values[name] for name in features.get_all_names() if name in values}}


class MetricConverter(FeatureGroup):
    """Requests metric_a with an extra group option, splitting SplitMetricSource into a
    second option bucket relative to a sibling requesting metric_b without it."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("metric_a", options={"unit": "x"})}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data.column("metric_a")}


class MetricCombiner(FeatureGroup):
    """Declares inputs spanning both option buckets of SplitMetricSource (via MetricConverter
    and directly via metric_b) with no Link, triggering the runtime KeyError."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature.int32_of("MetricConverter"),
            Feature.int32_of("metric_b"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        col_conv = data.column("MetricConverter")
        col_b = data.column("metric_b")
        return {cls.get_class_name(): pc.add(col_conv, col_b)}


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
            - Provides example code with Link.inner() and JoinSpec
            - Shows Link.inner_on() for feature groups without index_columns()
            - Lists available join types
        """
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MultiDependencyFeature")],
                links=set(),  # EMPTY - this should trigger the error
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
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
        assert "JoinSpec" in error_message, "Error should show JoinSpec usage"

        # Check for join type documentation
        assert any(join_type in error_message for join_type in ["inner", "left", "right", "outer"]), (
            "Error should list available join types"
        )

    def test_missing_links_error_has_no_option_split_hint_when_no_split_occurred(self) -> None:
        """Regression pin: without an option-split root, the error must not carry the new hint."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MultiDependencyFeature")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {RootFeatureA, RootFeatureB, MultiDependencyFeature}
                ),
            )

        error_message = str(exc_info.value)
        assert "differing option" not in error_message.lower(), (
            "Error should not mention a differing-option split when the root feature groups never split by options"
        )

    def test_missing_links_error_hints_option_split_when_root_splits_by_options(self) -> None:
        """When a downstream step's ancestors span two option buckets of one root feature group,
        the error must name that root and the differing option key."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MetricCombiner")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {SplitMetricSource, MetricConverter, MetricCombiner}
                ),
            )

        error_message = str(exc_info.value)

        assert "differing option" in error_message.lower(), "Error should mention the differing-option split hint"
        assert "SplitMetricSource" in error_message, (
            "Error should name the feature group whose option-based split caused the mismatch"
        )
        assert "unit" in error_message, "Error should name the differing option key that caused the split"
