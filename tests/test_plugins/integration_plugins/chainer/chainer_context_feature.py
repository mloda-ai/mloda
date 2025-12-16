"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda import FeatureGroup
from mloda import Feature

from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda import Options
from mloda.provider import FeatureChainParser
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class ChainedContextFeatureGroupTest(FeatureGroup):
    SUFFIX_PATTERN = [r".*__chainer_([\w]+)$"]
    OPERATION_ID = "chainer_"  # Used for constructing feature names

    PROPERTY_MAPPING = {
        "ident": {
            "identifier1": "explanation",
            "identifier2": "explanation",
            DefaultOptionKeys.context: True,  # Mark as context parameter
            DefaultOptionKeys.strict_validation: True,  # Enable strict validation for ident
        },
        "property2": {
            "value1": "explanation",
            "value2": "explanation",
            "specific_val_3_test": "explanation",  # Special case for testing
            DefaultOptionKeys.strict_validation: True,  # Enable strict validation for property2
            DefaultOptionKeys.default: "value1",  # Default value
            # Not marked as context -> defaults to group parameter
        },
        "property3": {
            "opt_val1": "explanation",
            "opt_val2": "explanation",
            DefaultOptionKeys.default: "opt_val1",  # Default value
            DefaultOptionKeys.context: True,  # Mark as context parameter
            DefaultOptionKeys.strict_validation: False,  # Disable strict validation for property3
        },
        DefaultOptionKeys.in_features: {
            "explanation": "explanation",
            DefaultOptionKeys.context: True,  # Mark as context parameter
            # No strict validation for source feature -> defaults to flexible
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern and aggregation type."""

        if not FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name=feature_name,
            options=options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=cls.SUFFIX_PATTERN,  # Using suffix patterns with L→R syntax
        ):
            return False

        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        features = set()

        # String-based feature extraction
        config, string_based_feature = FeatureChainParser.parse_feature_name(feature_name, self.SUFFIX_PATTERN)
        if config is not None and string_based_feature is not None:
            feat = Feature(
                string_based_feature,
                options=Options(
                    group={
                        "property2": "value1",  # default group parameter
                        DefaultOptionKeys.feature_chainer_parser_key: frozenset(
                            ["ident", DefaultOptionKeys.in_features.value, "property2", "property3"]
                        ),
                    },
                    context={
                        "ident": config,  # context parameter parsed from the feature name
                        "property3": "opt_val1",  # optional parameter set
                    },
                ),
            )
            features.add(feat)
            return features

        # Configuration-based approach
        source_features = options.get_in_features()
        for source_feature in source_features:
            source_feature.options.add(
                DefaultOptionKeys.feature_chainer_parser_key,
                frozenset(["ident", DefaultOptionKeys.in_features.value, "property2", "property3"]),
            )
            features.add(source_feature)

        if features:
            return features
        raise ValueError

    @classmethod
    def perform_operation(cls, data: Any, feature: Feature) -> Any:
        """
        PERFORM OPERATION HAS 3 STEPS:
        1. Check if the source feature exists in the data.
        2. Perform the operation (e.g., doubling the values).
        3. Add the result to the data with a new feature name.
        """

        # Try config-based approach first
        try:
            source_features = feature.options.get_in_features()
            source_feature = next(iter(source_features))
            source_feature_name: str | None = source_feature.get_name()

            # Apply configuration from options
            has_prefix_configuration = feature.options.get("ident")
            if has_prefix_configuration == "identifier1":
                val = 0
            elif has_prefix_configuration == "identifier2":
                val = 1
            else:
                raise ValueError("does not match the expected property mapping")

            if "placeholder" in feature.get_name():
                if feature.options.get("property2") == "specific_val_3_test":
                    val = 4

                assert_propert2_is_set(feature)
                assert_property3_is_set(feature)

        except ValueError:
            # Fall back to string-based approach
            has_suffix_configuration, source_feature_name = FeatureChainParser.parse_feature_name(
                feature.name, cls.SUFFIX_PATTERN
            )

            if has_suffix_configuration is None or source_feature_name is None:
                raise ValueError(f"Could not parse feature name: {feature.name} and no source features in options")

            if source_feature_name not in data.columns:
                raise ValueError(f"Source feature '{source_feature_name}' not found in data.")

            if has_suffix_configuration == "identifier1":
                val = 0
            elif has_suffix_configuration == "identifier2":
                val = 1
            else:
                raise ValueError("does not match the expected property mapping")

            if source_feature_name is None:
                raise ValueError("Source feature name is None, cannot perform operation.")

        data[feature.get_name()] = data[source_feature_name] * 2 + val
        return data

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data = cls.perform_operation(data, feature)
        return data


def assert_propert2_is_set(feature: Feature) -> None:
    """
    Assert that the 'property2' is set in the feature options and tests
    that the correct value is propagated through the chained features.

    This is a utility function for testing chainer context features.

    Do not use it for concrete implementations.
    """

    val = feature.options.get("property2")
    if not val:
        raise ValueError("Property 'property2' is not somehow set in the test correctly.")

    expected = {
        "placeholder1": "value1",
        "placeholder2": "value2",
        "placeholder3": "value1",
        "placeholder4": "value1",
        "placeholder4_5": "specific_val_3_test",
        "placeholder4_6": "value1",
    }.get(feature.get_name())

    if expected is None:
        raise ValueError(f"Unknown feature name: {feature.get_name()}. Cannot assert 'property2' value.")

    if val != expected:
        raise ValueError(f"Property 'property2' for {feature.get_name()} should be '{expected}', but got '{val}'")


def assert_property3_is_set(feature: Feature) -> None:
    """
    Assert that the 'property3' is set correctly for features that should have it,
    and is absent for features that should not have it (demonstrating optional behavior).

    This is a utility function for testing chainer context features with optional properties.

    Do not use it for concrete implementations.
    """

    val = feature.options.get("property3")

    # Define which features should have property3 and their expected values
    expected_values = {
        "placeholder1": "opt_val1",  # Should have property3
        "placeholder2": None,  # Should NOT have property3 (optional)
        "placeholder3": "opt_val2",  # Should have property3
        "placeholder4": None,  # Should NOT have property3 (optional)
        "placeholder4_5": None,  # Should NOT have property3 (optional)
        "placeholder4_6": None,  # Should NOT have property3 (optional)
    }

    expected = expected_values.get(feature.get_name())

    if expected is None and feature.get_name() not in expected_values:
        raise ValueError(f"Unknown feature name: {feature.get_name()}. Cannot assert 'property3' value.")

    # For features that should NOT have property3
    if expected is None:
        if val is not None:
            raise ValueError(f"Property 'property3' for {feature.get_name()} should be None (absent), but got '{val}'")
    # For features that should have property3
    else:
        if val != expected:
            raise ValueError(f"Property 'property3' for {feature.get_name()} should be '{expected}', but got '{val}'")
