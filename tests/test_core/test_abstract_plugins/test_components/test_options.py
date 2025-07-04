import pytest
from mloda_core.abstract_plugins.components.options import Options


class TestOptions:
    """Test suite for the new Options class with group/context separation."""

    def test_legacy_initialization(self) -> None:
        """Legacy initialization should move all data to group."""
        options = Options({"key1": "value1", "key2": "value2"})
        assert options.group == {"key1": "value1", "key2": "value2"} == options.data
        assert options.context == {}

    def test_new_initialization(self) -> None:
        """Test new group/context initialization."""
        options = Options(
            group={"data_source": "prod", "environment": "staging"},
            context={"aggregation_type": "sum", "debug_mode": True},
        )

        assert options.group == {"data_source": "prod", "environment": "staging"}
        assert options.context == {"aggregation_type": "sum", "debug_mode": True}

    def test_mixed_initialization_error(self) -> None:
        """Test that mixing legacy and new initialization raises error."""
        with pytest.raises(ValueError, match="Cannot specify both 'data' and 'group'/'context' parameters"):
            Options(data={"key": "value"}, group={"other": "value"})

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

    def test_update_considering_mloda_source(self) -> None:
        """Test update method with mloda_source exclusion."""
        from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

        options1 = Options(group={"key1": "value1"})
        options2 = Options(group={"key2": "value2", DefaultOptionKeys.mloda_source_feature: "source1"})

        # Update should work normally - mloda_source_feature is included when only one has it
        options1.update_considering_mloda_source(options2)
        assert options1.group == {"key1": "value1", "key2": "value2", DefaultOptionKeys.mloda_source_feature: "source1"}

        # mloda_source_feature should be excluded when both have it
        options3 = Options(group={"key3": "value3", DefaultOptionKeys.mloda_source_feature: "source2"})
        options1.update_considering_mloda_source(options3)
        # The existing mloda_source_feature should be preserved
        assert options1.group[DefaultOptionKeys.mloda_source_feature] == "source1"
        assert options1.group["key3"] == "value3"

    def test_update_conflict_detection(self) -> None:
        """Test that update detects conflicts between group and context."""
        options1 = Options(group={"group_key": "value1"}, context={"context_key": "value1"})
        options2 = Options(
            group={"context_key": "value2"}  # This conflicts with options1.context
        )

        with pytest.raises(ValueError, match="Cannot update group: keys already exist in context"):
            options1.update_considering_mloda_source(options2)

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
                "mloda_source_feature": "sales_data",
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
                "mloda_source_feature": "sales_data",
                "debug_mode": True,
            },
        )

        # These should NOT be equal (different group parameters)
        assert options != optimized_options

        # But legacy get should still work for both
        assert optimized_options.get("data_source") == "prod"
        assert optimized_options.get("aggregation_type") == "sum"
