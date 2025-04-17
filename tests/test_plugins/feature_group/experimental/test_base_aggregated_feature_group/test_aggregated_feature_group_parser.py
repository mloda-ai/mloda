"""
Tests for the AggregatedFeatureGroup parser functionality.
"""

import pytest
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.options import Options


class TestAggregatedFeatureGroupParser:
    def test_parse_from_options_valid(self) -> None:
        """Test parsing with valid options."""
        options = {AggregatedFeatureGroup.AGGREGATION_TYPE: "sum", DefaultOptionKeys.mloda_source_feature: "Sales"}

        feature_name = AggregatedFeatureGroup.configurable_feature_chain_parser().create_feature_without_options(  # type: ignore
            Feature("x", Options(options))
        )

        assert feature_name.name == "sum_aggr__Sales"  # type: ignore

    def test_parse_from_options_all_aggregation_types(self) -> None:
        """Test parsing with all supported aggregation types."""
        source_feature = "Revenue"

        for agg_type in AggregatedFeatureGroup.AGGREGATION_TYPES.keys():
            options = Options(
                {
                    AggregatedFeatureGroup.AGGREGATION_TYPE: agg_type,
                    DefaultOptionKeys.mloda_source_feature: source_feature,
                }
            )

            feature_name = AggregatedFeatureGroup.configurable_feature_chain_parser().parse_from_options(options)  # type: ignore

            assert feature_name == f"{agg_type}_aggr__{source_feature}"

    def test_parse_from_options_missing_aggregation_type(self) -> None:
        """Test parsing with missing aggregation type."""
        options = Feature("x", Options({DefaultOptionKeys.mloda_source_feature: "Sales"}))

        result = AggregatedFeatureGroup.configurable_feature_chain_parser().create_feature_without_options(options)  # type: ignore
        assert result is None

    def test_parse_from_options_missing_source_feature(self) -> None:
        """Test parsing with missing source feature."""
        options = Options({AggregatedFeatureGroup.AGGREGATION_TYPE: "sum"})

        result = AggregatedFeatureGroup.configurable_feature_chain_parser().parse_from_options(options)  # type: ignore
        assert result is None

    def test_parse_from_options_invalid_aggregation_type(self) -> None:
        """Test parsing with invalid aggregation type."""
        options = Options(
            {
                AggregatedFeatureGroup.AGGREGATION_TYPE: "invalid_type",
                DefaultOptionKeys.mloda_source_feature: "Sales",
            }
        )

        with pytest.raises(ValueError, match="Unsupported aggregation type: invalid_type"):
            AggregatedFeatureGroup.configurable_feature_chain_parser().parse_from_options(options)  # type: ignore

    def test_create_feature_valid(self) -> None:
        """Test creating a feature with valid options."""
        options = Feature(
            "x",
            Options({AggregatedFeatureGroup.AGGREGATION_TYPE: "sum", DefaultOptionKeys.mloda_source_feature: "Sales"}),
        )

        feature = AggregatedFeatureGroup.configurable_feature_chain_parser().create_feature_without_options(options)  # type: ignore

        assert isinstance(feature, Feature)
        assert feature.name == "sum_aggr__Sales"
        # Check that the options values match
        assert feature.options.get(AggregatedFeatureGroup.AGGREGATION_TYPE) is None
        assert feature.options.get(DefaultOptionKeys.mloda_source_feature) is None

    def test_chained_feature_parsing(self) -> None:
        """Test parsing with a chained source feature."""
        options = Options(
            {
                AggregatedFeatureGroup.AGGREGATION_TYPE: "max",
                DefaultOptionKeys.mloda_source_feature: "mean_imputed__Sales",
            }
        )

        feature_name = AggregatedFeatureGroup.configurable_feature_chain_parser().parse_from_options(options)  # type: ignore

        assert feature_name == "max_aggr__mean_imputed__Sales"

        # Verify that the feature chain parser can extract the source feature correctly
        source_feature = FeatureChainParser.extract_source_feature(feature_name, AggregatedFeatureGroup.PREFIX_PATTERN)

        assert source_feature == "mean_imputed__Sales"
