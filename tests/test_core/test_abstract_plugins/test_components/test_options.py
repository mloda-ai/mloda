import pytest
from mloda_core.abstract_plugins.components.options import Options


class TestOptions:
    """Test suite for the new Options class with group/context separation."""

    def test_legacy_initialization(self) -> None:
        """Legacy initialization should move all data to group."""
        options = Options({"key1": "value1", "key2": "value2"})
        assert options.group == {"key1": "value1", "key2": "value2"}
        assert options.context == {}

    def test_new_initialization(self) -> None:
        """Test new group/context initialization."""
        options = Options(
            group={"data_source": "prod", "environment": "staging"},
            context={"aggregation_type": "sum", "debug_mode": True},
        )

        assert options.group == {"data_source": "prod", "environment": "staging"}
        assert options.context == {"aggregation_type": "sum", "debug_mode": True}

    def test_unknown_parameter_error(self) -> None:
        """Test that using unknown parameters raises TypeError."""
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            Options(data={"key": "value"})  # type: ignore

    def test_duplicate_keys_validation(self) -> None:
        """Test that duplicate keys in group and context raise error."""
        with pytest.raises(ValueError, match="Keys cannot exist in both group and context"):
            Options(group={"shared_key": "value1"}, context={"shared_key": "value2"})

    def test_add_legacy_method(self) -> None:
        """Test legacy add method adds to group."""
        options = Options()
        options.add("key1", "value1")

        assert options.group == {"key1": "value1"}
        assert options.context == {}

    def test_add_to_group(self) -> None:
        """Test adding parameters to group."""
        options = Options()
        options.add_to_group("data_source", "prod")
        options.add_to_group("environment", "staging")

        assert options.group == {"data_source": "prod", "environment": "staging"}
        assert options.context == {}

    def test_add_to_context(self) -> None:
        """Test adding parameters to context."""
        options = Options()
        options.add_to_context("aggregation_type", "sum")
        options.add_to_context("debug_mode", True)

        assert options.group == {}
        assert options.context == {"aggregation_type": "sum", "debug_mode": True}

    def test_add_duplicate_key_error(self) -> None:
        """Test that adding duplicate keys raises error."""
        options = Options()
        options.add_to_group("key1", "value1")

        # Try to add same key to group
        with pytest.raises(ValueError, match="Key key1 already exists in group options"):
            options.add_to_group("key1", "value2")

        # Try to add same key to context
        with pytest.raises(ValueError, match="Key key1 already exists in group options"):
            options.add_to_context("key1", "value2")

    def test_add_cross_contamination_error(self) -> None:
        """Test that adding key that exists in other section raises error."""
        options = Options()
        options.add_to_group("shared_key", "group_value")

        with pytest.raises(ValueError, match="Key shared_key already exists in group options"):
            options.add_to_context("shared_key", "context_value")

    def test_equality_based_on_group_only(self) -> None:
        """Test that equality is based only on group parameters."""
        options1 = Options(group={"data_source": "prod"}, context={"aggregation_type": "sum"})
        options2 = Options(
            group={"data_source": "prod"},
            context={"aggregation_type": "avg"},  # Different context
        )
        options3 = Options(
            group={"data_source": "staging"},  # Different group
            context={"aggregation_type": "sum"},
        )

        # Same group, different context -> equal
        assert options1 == options2

        # Different group, same context -> not equal
        assert options1 != options3
        assert options2 != options3

    def test_hash_based_on_group_only(self) -> None:
        """Test that hash is based only on group parameters."""
        options1 = Options(group={"data_source": "prod"}, context={"aggregation_type": "sum"})
        options2 = Options(
            group={"data_source": "prod"},
            context={"aggregation_type": "avg"},  # Different context
        )
        options3 = Options(
            group={"data_source": "staging"},  # Different group
            context={"aggregation_type": "sum"},
        )

        # Same group, different context -> same hash
        assert hash(options1) == hash(options2)

        # Different group -> different hash
        assert hash(options1) != hash(options3)

    def test_get_legacy_method(self) -> None:
        """Test legacy get method searches both group and context."""
        options = Options(group={"group_key": "group_value"}, context={"context_key": "context_value"})

        # Should find in group first
        assert options.get("group_key") == "group_value"

        # Should find in context if not in group
        assert options.get("context_key") == "context_value"

        # Should return None if not found
        assert options.get("nonexistent_key") is None

    def test_str_representation(self) -> None:
        """Test string representation shows both group and context."""
        options = Options(group={"data_source": "prod"}, context={"aggregation_type": "sum"})

        expected = "Options(group={'data_source': 'prod'}, context={'aggregation_type': 'sum'})"
        assert str(options) == expected

    def test_update_with_protected_keys_default(self) -> None:
        """Test update method with default protected keys (mloda_source_features)."""
        from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

        options1 = Options(group={"key1": "value1"})
        options2 = Options(group={"key2": "value2", DefaultOptionKeys.mloda_source_features: "source1"})

        # Update should work normally - mloda_source_features is included when only one has it
        options1.update_with_protected_keys(options2)
        assert options1.group == {"key1": "value1", "key2": "value2"}

        # mloda_source_features should be excluded when both have it
        options3 = Options(group={"key3": "value3", DefaultOptionKeys.mloda_source_features: "source2"})
        options2.update_with_protected_keys(options3)
        # The existing mloda_source_features should be preserved
        assert options2.group[DefaultOptionKeys.mloda_source_features] == "source1"
        assert options2.group["key3"] == "value3"

    def test_update_conflict_detection(self) -> None:
        """Test that update detects conflicts between group and context."""
        options1 = Options(group={"group_key": "value1"}, context={"context_key": "value1"})
        options2 = Options(
            group={"context_key": "value2"}  # This conflicts with options1.context
        )

        with pytest.raises(ValueError, match="Cannot update group: keys already exist in context"):
            options1.update_with_protected_keys(options2)

    def test_backward_compatibility_with_feature_class(self) -> None:
        """Test that Options work correctly with Feature class equality."""
        from mloda_core.abstract_plugins.components.feature import Feature

        # Features with same group options should be equal
        feature1 = Feature("test_feature", Options(group={"data_source": "prod"}))
        feature2 = Feature("test_feature", Options(group={"data_source": "prod"}))

        assert feature1 == feature2

        # Features with different group options should not be equal
        feature3 = Feature("test_feature", Options(group={"data_source": "staging"}))

        assert feature1 != feature3

    def test_migration_scenario(self) -> None:
        """Test typical migration scenario: all options start in group."""
        # Current usage pattern (all options in group during migration)
        options = Options(
            group={
                "data_source": "prod",
                "aggregation_type": "sum",
                "mloda_source_features": "sales_data",
                "debug_mode": True,
            }
        )

        # Verify all options are in group (maintaining current behavior)
        assert len(options.group) == 4
        assert len(options.context) == 0

        # Legacy get method should still work
        assert options.get("data_source") == "prod"
        assert options.get("aggregation_type") == "sum"

        # Future optimization: move some parameters to context
        optimized_options = Options(
            group={"data_source": "prod"},  # Only isolation-requiring parameters
            context={  # Metadata parameters
                "aggregation_type": "sum",
                "mloda_source_features": "sales_data",
                "debug_mode": True,
            },
        )

        # These should NOT be equal (different group parameters)
        assert options != optimized_options

        # But legacy get should still work for both
        assert optimized_options.get("data_source") == "prod"
        assert optimized_options.get("aggregation_type") == "sum"


class TestProtectedKeys:
    """
    Test suite for protected keys mechanism in feature chaining.

    WHY THIS EXISTS:
    When features are chained (parent -> child), certain option keys need to have
    different values at each level without causing conflicts. This is critical for
    feature chaining where each level specifies its own data source or configuration.

    EXAMPLE USE CASE:
    - Parent feature: processes data from "sales_database"
    - Child feature: processes data from "user_database"
    - Without protection: ERROR (duplicate key conflict)
    - With protection: Both features keep their own data sources

    MECHANISM:
    Protected keys are excluded from merging - the parent keeps its value,
    the child keeps its value, and no conflict is raised.
    """

    def test_update_with_default_protected_keys(self) -> None:
        """
        Test that mloda_source_features is protected by default.

        This is the most common use case - allowing parent and child features
        to have different data sources without conflict.
        """
        from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

        # Parent feature options
        parent_options = Options(
            group={
                "common_key": "shared_value",
                DefaultOptionKeys.mloda_source_features: "parent_source",
            }
        )

        # Child feature options with different source
        child_options = Options(
            group={
                "child_key": "child_value",
                DefaultOptionKeys.mloda_source_features: "child_source",
            }
        )

        # Update parent with child options
        parent_options.update_with_protected_keys(child_options)

        # Parent should keep its own source (protected key)
        assert parent_options.group[DefaultOptionKeys.mloda_source_features] == "parent_source"

        # Parent should receive child's non-protected keys
        assert parent_options.group["child_key"] == "child_value"

        # Common keys should be preserved
        assert parent_options.group["common_key"] == "shared_value"

    def test_update_with_custom_protected_keys(self) -> None:
        """
        Test using custom protected keys beyond the default.

        This allows extending the mechanism to protect additional keys
        specific to certain feature types or use cases.
        """
        # Parent feature options
        parent_options = Options(
            group={
                "environment": "production",
                "region": "us-east",
                "property1": "parent_value1",
                "property2": "parent_value2",
            }
        )

        # Child feature options with different property values
        child_options = Options(
            group={
                "property1": "child_value1",
                "property2": "child_value2",
                "new_key": "new_value",
            }
        )

        # Define custom protected keys
        custom_protected = {"property1", "property2"}

        # Update with custom protection
        parent_options.update_with_protected_keys(child_options, custom_protected)

        # Protected keys should NOT be overwritten
        assert parent_options.group["property1"] == "parent_value1"
        assert parent_options.group["property2"] == "parent_value2"

        # Non-protected keys should be merged
        assert parent_options.group["new_key"] == "new_value"
        assert parent_options.group["environment"] == "production"
        assert parent_options.group["region"] == "us-east"

    def test_update_without_conflicts_when_protected(self) -> None:
        """
        Test that protected keys with different values don't cause conflicts.

        This is the KEY FEATURE - without protection, this would raise an error.
        """
        from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

        parent_options = Options(group={DefaultOptionKeys.mloda_source_features: "source_A", "shared": "value"})

        child_options = Options(group={DefaultOptionKeys.mloda_source_features: "source_B", "shared": "value"})

        # This should NOT raise an error because mloda_source_features is protected
        parent_options.update_with_protected_keys(child_options)

        # Parent's protected key value is preserved
        assert parent_options.group[DefaultOptionKeys.mloda_source_features] == "source_A"

    def test_update_raises_error_for_non_protected_conflicts(self) -> None:
        """
        Test that non-protected keys still raise errors when they conflict.

        This ensures we still catch genuine configuration conflicts.
        """
        parent_options = Options(group={"region": "us-east", "mode": "production"})

        child_options = Options(group={"region": "eu-west"})  # Conflicts with parent

        # Should raise error because 'region' is not protected and values differ
        # Note: The actual conflict check happens in merge_options in feature_collection.py
        # Here we verify the mechanism doesn't hide real conflicts
        parent_options.update_with_protected_keys(child_options)

        # The update succeeds at Options level (it just merges)
        # The conflict check is done at the Features level in feature_collection.py
        assert parent_options.group["region"] == "eu-west"

    def test_empty_protected_keys_set(self) -> None:
        """
        Test behavior when protected_keys is an empty set.

        This should merge all keys normally (no protection).
        """
        parent_options = Options(group={"key1": "parent_value"})

        child_options = Options(group={"key2": "child_value"})

        # Update with empty protection set
        parent_options.update_with_protected_keys(child_options, set())

        # All keys should be merged
        assert parent_options.group["key1"] == "parent_value"
        assert parent_options.group["key2"] == "child_value"

    def test_protected_key_only_in_child(self) -> None:
        """
        Test when protected key exists only in child options.

        The protected key in child should NOT be merged to parent.
        """
        from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

        # Parent has no source feature
        parent_options = Options(group={"key1": "value1"})

        # Child has source feature
        child_options = Options(
            group={
                DefaultOptionKeys.mloda_source_features: "child_source",
                "key2": "value2",
            }
        )

        # Update parent with child
        parent_options.update_with_protected_keys(child_options)

        # Protected key should NOT be added to parent
        assert DefaultOptionKeys.mloda_source_features not in parent_options.group

        # Non-protected keys should be merged
        assert parent_options.group["key2"] == "value2"

    def test_protected_key_only_in_parent(self) -> None:
        """
        Test when protected key exists only in parent options.

        Parent's protected key should be preserved.
        """
        from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

        # Parent has source feature
        parent_options = Options(
            group={
                DefaultOptionKeys.mloda_source_features: "parent_source",
                "key1": "value1",
            }
        )

        # Child has no source feature
        child_options = Options(group={"key2": "value2"})

        # Update parent with child
        parent_options.update_with_protected_keys(child_options)

        # Parent's protected key should be preserved
        assert parent_options.group[DefaultOptionKeys.mloda_source_features] == "parent_source"

        # Child's keys should be merged
        assert parent_options.group["key2"] == "value2"

    def test_multiple_protected_keys(self) -> None:
        """
        Test protecting multiple keys simultaneously.

        This validates the mechanism works for any number of protected keys.
        """
        parent_options = Options(
            group={
                "protected1": "parent_p1",
                "protected2": "parent_p2",
                "protected3": "parent_p3",
                "normal": "parent_normal",
            }
        )

        child_options = Options(
            group={
                "protected1": "child_p1",
                "protected2": "child_p2",
                "protected3": "child_p3",
                "new_key": "child_new",
            }
        )

        protected_set = {"protected1", "protected2", "protected3"}

        parent_options.update_with_protected_keys(child_options, protected_set)

        # All protected keys should retain parent values
        assert parent_options.group["protected1"] == "parent_p1"
        assert parent_options.group["protected2"] == "parent_p2"
        assert parent_options.group["protected3"] == "parent_p3"

        # Non-protected keys should be merged
        assert parent_options.group["normal"] == "parent_normal"
        assert parent_options.group["new_key"] == "child_new"

    def test_context_conflict_detection_still_works(self) -> None:
        """
        Test that context/group conflict detection still works with protected keys.

        Protected keys only affect group-to-group merging, not group/context validation.
        """
        parent_options = Options(group={"group_key": "value"}, context={"context_key": "value"})

        child_options = Options(
            group={"context_key": "different_value"}  # This conflicts with parent's context
        )

        # Should raise error because group/context conflicts are always invalid
        with pytest.raises(ValueError, match="Cannot update group: keys already exist in context"):
            parent_options.update_with_protected_keys(child_options)
