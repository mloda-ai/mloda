from typing import Any

import pytest

from mloda.user import Feature
from mloda.user import Options
from mloda.provider import FeatureChainParser
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec
from tests.test_plugins.integration_plugins.chainer.chainer_context_feature import (
    ChainedContextFeatureGroupTest,
)


class TestParameterResolutionUnit:
    """Unit tests for parameter resolution logic in chained features."""

    def test_parameter_cannot_exist_in_both_group_and_context_simultaneously(self) -> None:
        with pytest.raises(ValueError, match="Keys cannot exist in both group and context: {'ident'}"):
            Feature(
                name="conflict_test",
                options=Options(group={"ident": "identifier1"}, context={"ident": "identifier2"}),
            )

    def test_missing_required_parameters_validation(self) -> None:
        """Test proper error handling when required parameters are missing."""

        property_mapping = {
            "ident": PropertySpec(
                "Required identifier",
                allowed_values={"identifier1": "explanation", "identifier2": "explanation"},
                context=True,
            ),
            "property2": PropertySpec(
                "Required group property",
                allowed_values={
                    "value1": "explanation",
                    "value2": "explanation",
                    "specific_val_3_test": "explanation",
                },
                context=False,
            ),
            "property3": PropertySpec(
                "Optional property with a default",
                allowed_values={"opt_val1": "explanation", "opt_val2": "explanation"},
                default="opt_val1",
                context=True,
            ),
            DefaultOptionKeys.in_features: PropertySpec(
                "explanation",
                context=True,  # Mark as context parameter
            ),
        }

        # Test: Missing required context parameter 'ident'
        options_missing_ident = Options(group={"property2": "value1"}, context={DefaultOptionKeys.in_features: "Sales"})

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_missing_ident, property_mapping
        )
        assert result is False, "Should fail validation when required 'ident' parameter is missing"

        # Test: Optional parameter 'property3' can be missing without errors
        options_missing_optional = Options(
            group={"property2": "value1"},
            context={
                DefaultOptionKeys.in_features: "Sales",
                "ident": "identifier1",
                # property3 omitted - should still work since it's optional
            },
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_missing_optional, property_mapping
        )
        assert result is True, "Should pass validation when optional 'property3' parameter is missing"

    def test_invalid_parameter_values_handling(self) -> None:
        """Test handling of parameter values not in PROPERTY_MAPPING."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: Invalid 'ident' value (not identifier1 or identifier2)
        options_invalid_ident = Options(
            group={"property2": "value1"},
            context={DefaultOptionKeys.in_features: "Sales", "ident": "invalid_identifier"},
        )

        with pytest.raises(ValueError, match="Property value 'invalid_identifier' not found in mapping for 'ident'"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "test_feature", options_invalid_ident, property_mapping
            )

        # Test: property3 with unlisted value should pass (non-strict validation)
        options_property3_unlisted = Options(
            group={"property2": "value1"},
            context={
                DefaultOptionKeys.in_features: "Sales",
                "ident": "identifier1",
                "property3": "unlisted_optional_value",  # Should be allowed since property3 is non-strict
            },
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_property3_unlisted, property_mapping
        )
        assert result is True, "Should pass validation when property3 has unlisted value (non-strict validation)"

    def test_parameter_type_variations(self) -> None:
        """Test different parameter types and value formats."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: Feature object as source feature
        source_feature = Feature(name="source", options=Options())
        options_with_feature = Options(
            group={"property2": "value1"},
            context={DefaultOptionKeys.in_features: source_feature, "ident": "identifier1"},
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_with_feature, property_mapping
        )
        assert result is True, "Should handle Feature objects as source features"

        # Test: Frozenset of Features for multiple source features
        source_features = frozenset(
            [Feature(name="source1", options=Options()), Feature(name="source2", options=Options())]
        )
        options_with_frozenset = Options(
            group={"property2": "value1"},
            context={DefaultOptionKeys.in_features: source_features, "ident": "identifier1"},
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_with_frozenset, property_mapping
        )
        assert result is True, "Should handle frozenset of Features as source features"

        # Test: String values for various parameters
        options_with_strings = Options(
            group={"property2": "value2"},
            context={DefaultOptionKeys.in_features: "Sales", "ident": "identifier2", "property3": "opt_val1"},
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_with_strings, property_mapping
        )
        assert result is True, "Should handle string values for all parameters"

    def test_parameter_categorization_logic(self) -> None:
        """Test the _determine_parameter_category logic."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: Context parameter marked in mapping goes to context by default
        options_context_default = Options(
            context={"ident": "identifier1"}  # ident is marked as context in mapping
        )

        category = FeatureChainParser._determine_parameter_category(
            "ident", property_mapping["ident"], options_context_default
        )
        assert category == DefaultOptionKeys.context, "Context-marked parameter should go to context"

        # Test: Group parameter (not marked as context) goes to group by default
        options_group_default = Options(
            group={"property2": "value1"}  # property2 is not marked as context in mapping
        )

        category = FeatureChainParser._determine_parameter_category(
            "property2", property_mapping["property2"], options_group_default
        )
        assert category == DefaultOptionKeys.group, "Non-context parameter should go to group"

        # Test: User override - context parameter forced to group
        options_user_override = Options(
            group={"ident": "identifier1"}  # User forces context parameter to group
        )

        category = FeatureChainParser._determine_parameter_category(
            "ident", property_mapping["ident"], options_user_override
        )
        assert category == DefaultOptionKeys.group, "User override should take precedence"

    def test_optional_parameter_handling(self) -> None:
        """Test comprehensive optional parameter behavior."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: Optional parameter present
        options_optional_present = Options(
            group={"property2": "value1"},
            context={
                DefaultOptionKeys.in_features: "Sales",
                "ident": "identifier1",
                "property3": "opt_val1",  # Optional parameter present
            },
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_optional_present, property_mapping
        )
        assert result is True, "Should pass validation when optional parameter is present"

        # Test: Optional parameter absent
        options_optional_absent = Options(
            group={"property2": "value1"},
            context={
                DefaultOptionKeys.in_features: "Sales",
                "ident": "identifier1",
                # property3 omitted
            },
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_optional_absent, property_mapping
        )
        assert result is True, "Should pass validation when optional parameter is absent"

        # Test: Check if parameter is correctly identified as optional
        can_skip = FeatureChainParser._can_skip_required_check(property_mapping["property3"])
        assert can_skip is True, "property3 should be identified as optional (has default)"

        cannot_skip = FeatureChainParser._can_skip_required_check(property_mapping["ident"])
        assert cannot_skip is False, "ident should not be identified as optional"

    def test_context_parameter_identification(self) -> None:
        """Test the _is_context_parameter logic."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: Parameters marked as context
        is_context_ident = FeatureChainParser._is_context_parameter(property_mapping["ident"])
        assert is_context_ident is True, "ident should be identified as context parameter"

        is_context_property3 = FeatureChainParser._is_context_parameter(property_mapping["property3"])
        assert is_context_property3 is True, "property3 should be identified as context parameter"

        is_context_source = FeatureChainParser._is_context_parameter(property_mapping[DefaultOptionKeys.in_features])
        assert is_context_source is True, "in_features should be identified as context parameter"

        # Test: Parameters not marked as context (group parameters)
        is_context_property2 = FeatureChainParser._is_context_parameter(property_mapping["property2"])
        assert is_context_property2 is False, "property2 should not be identified as context parameter"

    def test_edge_case_parameter_values(self) -> None:
        """Test boundary values and special cases."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: Empty frozenset for source features
        options_empty_frozenset = Options(
            group={"property2": "value1"},
            context={DefaultOptionKeys.in_features: frozenset(), "ident": "identifier1"},
        )

        # A present-but-empty container has zero elements, so there is nothing to reject:
        # the property mapping passes vacuously. Arity ("at least one source feature") is
        # enforced by MIN_IN_FEATURES in FeatureChainParserMixin, not by the value check.
        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_empty_frozenset, property_mapping
        )
        assert result is True, "An empty frozenset is present with zero elements, hence vacuously valid"

        # Test: Special test value for property2
        options_special_value = Options(
            group={"property2": "specific_val_3_test"},  # Special test value
            context={DefaultOptionKeys.in_features: "Sales", "ident": "identifier1"},
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_special_value, property_mapping
        )
        assert result is True, "Should handle special test value for property2"

    def test_parameter_extraction_logic(self) -> None:
        """Test the _extract_property_values logic."""
        # Test: the declared value space is returned, the fields around it are not
        spec_with_default = PropertySpec(
            "Optional property with a default",
            allowed_values={"opt_val1": "explanation", "opt_val2": "explanation"},
            default="opt_val1",
            context=True,
        )

        extracted = FeatureChainParser._extract_property_values(spec_with_default)
        expected = {"opt_val1": "explanation", "opt_val2": "explanation"}
        assert extracted == expected, "Should return exactly the declared allowed_values"

        # Test: a spec that declares no allowed_values declares an EMPTY value space.
        # (The retired dict form used to absorb unrecognized keys as accepted values;
        # PropertySpec has no such fallback.)
        spec_without_allowed_values = PropertySpec("Spec without a declared value space")

        extracted = FeatureChainParser._extract_property_values(spec_without_allowed_values)
        assert extracted == {}, "Should declare an empty value space when allowed_values is absent"

    def test_validation_final_properties(self) -> None:
        """Test the _validate_final_properties logic."""
        property_mapping = ChainedContextFeatureGroupTest.PROPERTY_MAPPING

        # Test: All required properties present. The tracker holds the collected elements;
        # None means the option was absent, an empty list means present with zero elements.
        property_tracker_valid: dict[str, list[Any] | None] = {
            "ident": ["identifier1"],
            "property2": ["value1"],
            DefaultOptionKeys.in_features: ["Sales"],
            "property3": [],  # Optional, can be empty
        }

        result = FeatureChainParser._validate_final_properties(property_tracker_valid, property_mapping)
        assert result is True, "Should pass validation when all required properties are present"

        # Test: Required property missing
        property_tracker_missing_required: dict[str, list[Any] | None] = {
            "ident": None,  # Required property missing
            "property2": ["value1"],
            DefaultOptionKeys.in_features: ["Sales"],
            "property3": [],
        }

        result = FeatureChainParser._validate_final_properties(property_tracker_missing_required, property_mapping)
        assert result is False, "Should fail validation when required property is missing"

        # Test: Optional property missing (should still pass)
        property_tracker_missing_optional: dict[str, list[Any] | None] = {
            "ident": ["identifier1"],
            "property2": ["value1"],
            DefaultOptionKeys.in_features: ["Sales"],
            "property3": None,  # Optional property missing
        }

        result = FeatureChainParser._validate_final_properties(property_tracker_missing_optional, property_mapping)
        assert result is True, "Should pass validation when only optional property is missing"

    def test_strict_validation_functionality(self) -> None:
        """Test the new strict validation functionality with default non-strict behavior."""

        # Test property mapping with mixed strict/flexible validation
        property_mapping_mixed = {
            "strict_param": PropertySpec(
                "Strictly validated parameter",
                allowed_values={"allowed_value1": "explanation", "allowed_value2": "explanation"},
                context=True,
                strict_validation=True,
            ),
            "flexible_param": PropertySpec(
                "Explicitly flexible parameter",
                context=True,
                strict_validation=False,  # Explicit flexible validation
            ),
            "default_flexible_param": PropertySpec("Parameter without a validation flag", context=True),
            DefaultOptionKeys.in_features: PropertySpec("Source features", context=True),
        }

        # Test: All parameters with valid values should pass
        options_all_valid = Options(
            context={
                "strict_param": "allowed_value1",
                "flexible_param": "suggested_value1",
                "default_flexible_param": "value1",
                DefaultOptionKeys.in_features: "Sales",
            }
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_all_valid, property_mapping_mixed
        )
        assert result is True, "Should pass validation when all parameters have valid values"

        # Test: Strict parameter with invalid value should fail
        options_strict_invalid = Options(
            context={
                "strict_param": "invalid_value",  # Not in mapping, should fail
                "flexible_param": "suggested_value1",
                "default_flexible_param": "value1",
                DefaultOptionKeys.in_features: "Sales",
            }
        )

        with pytest.raises(ValueError, match="Property value 'invalid_value' not found in mapping for 'strict_param'"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "test_feature", options_strict_invalid, property_mapping_mixed
            )

        # Test: Flexible parameter with value not in mapping should pass
        options_flexible_unlisted = Options(
            context={
                "strict_param": "allowed_value1",
                "flexible_param": "unlisted_value",  # Not in mapping but should be allowed
                "default_flexible_param": "value1",
                DefaultOptionKeys.in_features: "Sales",
            }
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_flexible_unlisted, property_mapping_mixed
        )
        assert result is True, "Should pass validation when flexible parameter has unlisted value"

        # Test: Default flexible parameter (no flag) with unlisted value should pass
        options_default_flexible_unlisted = Options(
            context={
                "strict_param": "allowed_value1",
                "flexible_param": "suggested_value1",
                "default_flexible_param": "unlisted_default_value",  # Not in mapping but should be allowed
                DefaultOptionKeys.in_features: "Sales",
            }
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_default_flexible_unlisted, property_mapping_mixed
        )
        assert result is True, "Should pass validation when default flexible parameter has unlisted value"

        # Test: in_features with default flexible validation should allow any value
        options_flexible_source = Options(
            context={
                "strict_param": "allowed_value1",
                "flexible_param": "suggested_value1",
                "default_flexible_param": "value1",
                DefaultOptionKeys.in_features: "any_source_feature_name",  # Should be allowed
            }
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_flexible_source, property_mapping_mixed
        )
        assert result is True, "Should pass validation when in_features has default flexible validation"

    def test_strict_validation_helper_methods(self) -> None:
        """Test the _is_strict_validation helper method with default non-strict behavior."""

        # Test: Spec with explicit strict validation = True
        strict_spec = PropertySpec(
            "Strict spec",
            allowed_values={"value1": "explanation"},
            strict_validation=True,
        )
        assert FeatureChainParser._is_strict_validation(strict_spec) is True, "Should identify strict validation = True"

        # Test: Spec with explicit strict validation = False
        flexible_spec = PropertySpec(
            "Flexible spec",
            allowed_values={"value1": "explanation"},
            strict_validation=False,
        )
        assert FeatureChainParser._is_strict_validation(flexible_spec) is False, (
            "Should identify strict validation = False"
        )

        # Test: Spec without an explicit strict validation flag (defaults to False)
        default_spec = PropertySpec(
            "Spec without a validation flag",
            allowed_values={"value1": "explanation"},
            context=True,
        )
        assert FeatureChainParser._is_strict_validation(default_spec) is False, (
            "Should default to strict validation = False"
        )

        # Test: Minimal spec (explanation only) defaults to False. The non-spec inputs
        # the old helper tolerated (bare strings, empty dicts) are rejected at class
        # definition now, so PropertySpec instances are the only shapes left to test.
        minimal_spec = PropertySpec("Minimal spec")
        assert FeatureChainParser._is_strict_validation(minimal_spec) is False, (
            "Should default to strict validation = False for a minimal spec"
        )

    def test_mixed_validation_scenarios(self) -> None:
        """Test complex scenarios with mixed strict and flexible validation."""

        property_mapping_complex = {
            "algorithm_type": PropertySpec(
                "Flexible algorithm selector",
                allowed_values={"sum": "explanation", "avg": "explanation"},
                context=True,
            ),
            # The retired dict spec also carried a DefaultOptionKeys.group key here; it was
            # dead (core never read it, group is simply context=False) and PropertySpec has
            # no such field, so it is dropped.
            "data_source": PropertySpec(
                "Strictly validated group parameter",
                allowed_values={"production": "explanation", "staging": "explanation"},
                context=False,
                strict_validation=True,
            ),
            "debug_mode": PropertySpec(
                "Explicitly flexible debug flag",
                allowed_values={"true": "explanation", "false": "explanation"},
                context=True,
                strict_validation=False,
            ),
            DefaultOptionKeys.in_features: PropertySpec(
                "explanation",
                # No strict_validation flag -> defaults to False (flexible)
                context=True,
            ),
        }

        # Test: Valid combination should pass
        options_valid_mix = Options(
            group={"data_source": "production"},  # Strict, valid value
            context={
                "algorithm_type": "custom_algorithm",  # Default flexible, unlisted value
                "debug_mode": "verbose",  # Explicit flexible, unlisted value
                DefaultOptionKeys.in_features: "any_feature",  # Default flexible
            },
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_valid_mix, property_mapping_complex
        )
        assert result is True, "Should pass validation with mixed strict/flexible parameters"

        # Test: Invalid strict parameter should fail
        options_invalid_strict = Options(
            group={"data_source": "development"},  # Strict, invalid value
            context={
                "algorithm_type": "custom_algorithm",  # Default flexible, should be fine
                "debug_mode": "verbose",  # Explicit flexible, should be fine
                DefaultOptionKeys.in_features: "any_feature",  # Default flexible
            },
        )

        with pytest.raises(ValueError, match="Property value 'development' not found in mapping for 'data_source'"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "test_feature", options_invalid_strict, property_mapping_complex
            )

        # Test: All flexible parameters with unlisted values should pass
        options_all_flexible_unlisted = Options(
            group={"data_source": "staging"},  # Strict, valid value
            context={
                "algorithm_type": "neural_network",  # Default flexible, unlisted
                "debug_mode": "trace",  # Explicit flexible, unlisted
                DefaultOptionKeys.in_features: "custom_feature",  # Default flexible, unlisted
            },
        )

        result = FeatureChainParser.match_configuration_feature_chain_parser(
            "test_feature", options_all_flexible_unlisted, property_mapping_complex
        )
        assert result is True, "Should pass validation when all flexible parameters have unlisted values"
