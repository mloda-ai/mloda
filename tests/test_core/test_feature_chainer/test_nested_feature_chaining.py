"""
Test nested Feature object chaining functionality.
"""

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestNestedFeatureChaining:
    """Test the new nested Feature object chaining functionality."""

    def test_extract_source_feature_from_options_string(self) -> None:
        """Test extracting string source feature from options."""
        options = Options(context={DefaultOptionKeys.mloda_source_feature: "price"})

        source = FeatureChainParser.extract_source_feature_from_options(options)
        assert source == "price"

    def test_extract_source_feature_from_options_feature_object(self) -> None:
        """Test extracting Feature object source feature from options."""
        nested_feature = Feature(
            "sum_aggr", Options(context={"aggregation_type": "sum", DefaultOptionKeys.mloda_source_feature: "price"})
        )

        options = Options(context={DefaultOptionKeys.mloda_source_feature: nested_feature})

        source = FeatureChainParser.extract_source_feature_from_options(options)
        assert isinstance(source, Feature)
        assert source.name.name == "sum_aggr"

    def test_extract_source_feature_from_options_none(self) -> None:
        """Test extracting source feature when none exists."""
        options = Options()

        source = FeatureChainParser.extract_source_feature_from_options(options)
        assert source is None

    def test_build_feature_with_nested_source_string(self) -> None:
        """Test building feature with string source."""
        feature = FeatureChainParser.build_feature_with_nested_source("max_aggr", "price")

        assert feature.name.name == "max_aggr"
        assert feature.options.get(DefaultOptionKeys.mloda_source_feature) == "price"

    def test_build_feature_with_nested_source_feature_object(self) -> None:
        """Test building feature with Feature object source."""
        nested_feature = Feature(
            "sum_aggr", Options(context={"aggregation_type": "sum", DefaultOptionKeys.mloda_source_feature: "price"})
        )

        feature = FeatureChainParser.build_feature_with_nested_source("max_aggr", nested_feature)

        assert feature.name.name == "max_aggr"
        source = feature.options.get(DefaultOptionKeys.mloda_source_feature)
        assert isinstance(source, Feature)
        assert source.name.name == "sum_aggr"

    def test_build_feature_with_additional_options(self) -> None:
        """Test building feature with additional options."""
        additional_options = Options(context={"aggregation_type": "max"})

        feature = FeatureChainParser.build_feature_with_nested_source("max_aggr", "price", additional_options)

        assert feature.name.name == "max_aggr"
        assert feature.options.get(DefaultOptionKeys.mloda_source_feature) == "price"
        assert feature.options.get("aggregation_type") == "max"

    def test_supports_nested_chaining_false(self) -> None:
        """Test supports_nested_chaining returns False for string source."""
        options = Options(context={DefaultOptionKeys.mloda_source_feature: "price"})

        assert not FeatureChainParser.supports_nested_chaining(options)

    def test_supports_nested_chaining_true(self) -> None:
        """Test supports_nested_chaining returns True for Feature object source."""
        nested_feature = Feature("sum_aggr")
        options = Options(context={DefaultOptionKeys.mloda_source_feature: nested_feature})

        assert FeatureChainParser.supports_nested_chaining(options)

    def test_supports_nested_chaining_no_source(self) -> None:
        """Test supports_nested_chaining returns False when no source."""
        options = Options()

        assert not FeatureChainParser.supports_nested_chaining(options)

    def test_convert_nested_to_string_equivalent_base_feature(self) -> None:
        """Test converting base feature (no source) to string."""
        feature = Feature("price")

        result = FeatureChainParser.convert_nested_to_string_equivalent(feature)
        assert result == "price"

    def test_convert_nested_to_string_equivalent_string_source(self) -> None:
        """Test converting feature with string source to string."""
        feature = Feature("max_aggr", Options(context={DefaultOptionKeys.mloda_source_feature: "price"}))

        result = FeatureChainParser.convert_nested_to_string_equivalent(feature)
        assert result == "max_aggr__price"

    def test_convert_nested_to_string_equivalent_nested_feature(self) -> None:
        """Test converting nested Feature objects to string."""
        # Create nested structure: max_aggr -> sum_aggr -> price
        nested_feature = Feature("sum_aggr", Options(context={DefaultOptionKeys.mloda_source_feature: "price"}))

        feature = Feature("max_aggr", Options(context={DefaultOptionKeys.mloda_source_feature: nested_feature}))

        result = FeatureChainParser.convert_nested_to_string_equivalent(feature)
        assert result == "max_aggr__sum_aggr__price"

    def test_convert_nested_to_string_equivalent_deep_nesting(self) -> None:
        """Test converting deeply nested Feature objects to string (5 levels)."""
        # Create 5-level deep nesting: level5 -> level4 -> level3 -> level2 -> level1 -> base
        level1 = Feature("level1", Options(context={DefaultOptionKeys.mloda_source_feature: "base"}))

        level2 = Feature("level2", Options(context={DefaultOptionKeys.mloda_source_feature: level1}))

        level3 = Feature("level3", Options(context={DefaultOptionKeys.mloda_source_feature: level2}))

        level4 = Feature("level4", Options(context={DefaultOptionKeys.mloda_source_feature: level3}))

        level5 = Feature("level5", Options(context={DefaultOptionKeys.mloda_source_feature: level4}))

        result = FeatureChainParser.convert_nested_to_string_equivalent(level5)
        assert result == "level5__level4__level3__level2__level1__base"

    def test_realistic_aggregation_chain(self) -> None:
        """Test realistic aggregation feature chain."""
        # Create: max_aggr -> sum_7_day_window -> mean_imputed -> price
        base_feature = "price"

        imputed_feature = Feature(
            "mean_imputed",
            Options(context={"imputation_method": "mean", DefaultOptionKeys.mloda_source_feature: base_feature}),
        )

        windowed_feature = Feature(
            "sum_7_day_window",
            Options(
                context={
                    "window_function": "sum",
                    "window_size": 7,
                    "time_unit": "day",
                    DefaultOptionKeys.mloda_source_feature: imputed_feature,
                }
            ),
        )

        final_feature = Feature(
            "max_aggr",
            Options(context={"aggregation_type": "max", DefaultOptionKeys.mloda_source_feature: windowed_feature}),
        )

        # Test that we can extract the source feature
        source = FeatureChainParser.extract_source_feature_from_options(final_feature.options)
        assert isinstance(source, Feature)
        assert source.name.name == "sum_7_day_window"

        # Test that we can convert to string equivalent
        result = FeatureChainParser.convert_nested_to_string_equivalent(final_feature)
        assert result == "max_aggr__sum_7_day_window__mean_imputed__price"

        # Test nested chaining detection
        assert FeatureChainParser.supports_nested_chaining(final_feature.options)

    def test_mixed_string_and_nested_approach(self) -> None:
        """Test that string-based and nested approaches can coexist."""
        # String-based feature
        string_feature = Feature("max_aggr__price")

        # Nested feature
        nested_inner = Feature("sum_aggr", Options(context={DefaultOptionKeys.mloda_source_feature: "temperature"}))

        nested_feature = Feature("avg_aggr", Options(context={DefaultOptionKeys.mloda_source_feature: nested_inner}))

        # Both should work
        assert not FeatureChainParser.supports_nested_chaining(string_feature.options)
        assert FeatureChainParser.supports_nested_chaining(nested_feature.options)

        # Convert nested to string should work
        nested_string = FeatureChainParser.convert_nested_to_string_equivalent(nested_feature)
        assert nested_string == "avg_aggr__sum_aggr__temperature"

    def test_group_context_separation_in_nested_features(self) -> None:
        """Test that group/context separation works with nested features."""
        # Create nested feature with group and context parameters
        nested_feature = Feature(
            "sum_aggr",
            Options(
                group={"data_source": "production"},
                context={"aggregation_type": "sum", DefaultOptionKeys.mloda_source_feature: "price"},
            ),
        )

        final_feature = Feature(
            "max_aggr",
            Options(
                group={"environment": "prod"},
                context={"aggregation_type": "max", DefaultOptionKeys.mloda_source_feature: nested_feature},
            ),
        )

        # Test that source feature extraction works
        source = FeatureChainParser.extract_source_feature_from_options(final_feature.options)
        assert isinstance(source, Feature)
        assert source.name.name == "sum_aggr"

        # Test that group/context parameters are preserved
        assert final_feature.options.group["environment"] == "prod"
        assert final_feature.options.context["aggregation_type"] == "max"
        assert nested_feature.options.group["data_source"] == "production"
        assert nested_feature.options.context["aggregation_type"] == "sum"
