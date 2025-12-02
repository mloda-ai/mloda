"""
Tests for the modernized AggregatedFeatureGroup with configuration-based approach.
"""

from typing import Any, List, Set

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class ConcreteAggregatedFeatureGroupForTest(AggregatedFeatureGroup):
    """Concrete subclass for testing base class methods."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> Set[str]:
        return set()

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        pass

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        return data

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, in_features: List[str]) -> Any:
        return None


class TestModernizedAggregatedFeatureGroup:
    """Tests for the modernized AggregatedFeatureGroup with proper group/context separation."""

    def test_configuration_based_feature_creation(self) -> None:
        """Test that configuration-based features work with proper parameter separation."""

        # Create a configuration-based feature with context parameters
        feature = Feature(
            name="placeholder",
            options=Options(
                group={
                    # No group parameters for this feature group
                },
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                    DefaultOptionKeys.in_features: "sales",
                },
            ),
        )

        # Test that match_feature_group_criteria works
        assert AggregatedFeatureGroup.match_feature_group_criteria(feature.name, feature.options)

        # Test parameter extraction
        aggregation_type, source_feature = AggregatedFeatureGroup._extract_aggr_and_source_feature(feature)
        assert aggregation_type == "sum"
        assert source_feature == "sales"

    def test_string_based_feature_still_works(self) -> None:
        """Test that string-based features still work (backward compatibility)."""

        feature = Feature("sales__sum_aggr")

        # Test that match_feature_group_criteria works
        assert AggregatedFeatureGroup.match_feature_group_criteria(feature.name, feature.options)

        # Test parameter extraction
        aggregation_type, source_feature = AggregatedFeatureGroup._extract_aggr_and_source_feature(feature)
        assert aggregation_type == "sum"
        assert source_feature == "sales"

    def test_context_parameter_validation(self) -> None:
        """Test that context parameters are properly validated."""

        # Valid aggregation type should work
        feature_valid = Feature(
            name="placeholder",
            options=Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "avg",
                    DefaultOptionKeys.in_features: "price",
                }
            ),
        )

        assert AggregatedFeatureGroup.match_feature_group_criteria(feature_valid.name, feature_valid.options)

        feature_invalid = Feature(
            name="placeholder",
            options=Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "invalid_type",
                    DefaultOptionKeys.in_features: "price",
                }
            ),
        )

        with pytest.raises(
            ValueError, match="Property value 'invalid_type' not found in mapping for 'aggregation_type'"
        ):
            AggregatedFeatureGroup.match_feature_group_criteria(feature_invalid.name, feature_invalid.options)

    def test_mixed_parameter_placement(self) -> None:
        """Test that parameters can be placed in different categories."""

        # Test with aggregation_type in context (default)
        feature_context = Feature(
            name="placeholder1",
            options=Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "max",
                    DefaultOptionKeys.in_features: "temperature",
                }
            ),
        )

        # Test with aggregation_type explicitly in group (user override)
        feature_group = Feature(
            name="placeholder2",
            options=Options(
                group={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "max",
                },
                context={
                    DefaultOptionKeys.in_features: "temperature",
                },
            ),
        )

        # Both should match
        assert AggregatedFeatureGroup.match_feature_group_criteria(feature_context.name, feature_context.options)
        assert AggregatedFeatureGroup.match_feature_group_criteria(feature_group.name, feature_group.options)

        # Both should extract parameters correctly
        agg_type1, source1 = AggregatedFeatureGroup._extract_aggr_and_source_feature(feature_context)
        agg_type2, source2 = AggregatedFeatureGroup._extract_aggr_and_source_feature(feature_group)

        assert agg_type1 == "max" and source1 == "temperature"
        assert agg_type2 == "max" and source2 == "temperature"

    def test_input_features_configuration_based(self) -> None:
        """Test input_features method with configuration-based approach."""

        # Create source feature
        source_feature = Feature("sales")

        options = Options(context={DefaultOptionKeys.in_features: frozenset([source_feature])})

        feature_group = ConcreteAggregatedFeatureGroupForTest()
        input_features = feature_group.input_features(options, FeatureName("placeholder"))

        assert input_features == {source_feature}

    def test_all_aggregation_types_supported(self) -> None:
        """Test that all defined aggregation types are supported in configuration."""

        for agg_type in AggregatedFeatureGroup.AGGREGATION_TYPES.keys():
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        AggregatedFeatureGroup.AGGREGATION_TYPE: agg_type,
                        DefaultOptionKeys.in_features: "test_feature",
                    }
                ),
            )

            # Should match
            assert AggregatedFeatureGroup.match_feature_group_criteria(feature.name, feature.options), (
                f"Aggregation type '{agg_type}' should be supported"
            )

            # Should extract correctly
            extracted_type, source = AggregatedFeatureGroup._extract_aggr_and_source_feature(feature)
            assert extracted_type == agg_type
            assert source == "test_feature"
