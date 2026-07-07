from copy import deepcopy

import pytest
from mloda.user import Options


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

    def test_backward_compatibility_with_feature_class(self) -> None:
        """Test that Options work correctly with Feature class equality."""
        from mloda.user import Feature

        # Features with same group options should be equal
        feature1 = Feature("test_feature", Options(group={"data_source": "prod"}))
        feature2 = Feature("test_feature", Options(group={"data_source": "prod"}))

        assert feature1 == feature2

        # Features with different group options should not be equal
        feature3 = Feature("test_feature", Options(group={"data_source": "staging"}))

        assert feature1 != feature3

    def test_getitem_group_key(self) -> None:
        """options["key"] returns same value as options.get("key") for group key."""
        options = Options(group={"group_key": "group_value"}, context={"context_key": "context_value"})
        assert options["group_key"] == options.get("group_key")

    def test_getitem_context_key(self) -> None:
        """options["key"] returns same value as options.get("key") for context key."""
        options = Options(group={"group_key": "group_value"}, context={"context_key": "context_value"})
        assert options["context_key"] == options.get("context_key")

    def test_getitem_missing_key_returns_none(self) -> None:
        """options["missing"] returns None, consistent with get()."""
        options = Options(group={"key": "value"})
        assert options["missing"] is None

    def test_migration_scenario(self) -> None:
        """Test typical migration scenario: all options start in group."""
        # Current usage pattern (all options in group during migration)
        options = Options(
            group={
                "data_source": "prod",
                "aggregation_type": "sum",
                "in_features": "sales_data",
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
                "in_features": "sales_data",
                "debug_mode": True,
            },
        )

        # These should NOT be equal (different group parameters)
        assert options != optimized_options

        # But legacy get should still work for both
        assert optimized_options.get("data_source") == "prod"
        assert optimized_options.get("aggregation_type") == "sum"


class TestPropagateContextKeys:
    """
    Test suite for the propagate_context_keys attribute on Options.

    WHY THIS EXISTS:
    When features are chained (parent -> child), context keys are normally
    scoped to the feature that defined them. propagate_context_keys explicitly
    marks which context keys should flow through to downstream features in
    the chain. Only keys listed here will be propagated; all others stay local.
    """

    def test_propagate_context_keys_default_empty(self) -> None:
        """Default propagate_context_keys should be an empty frozenset."""
        options = Options()
        assert options.propagate_context_keys == frozenset()

    def test_propagate_context_keys_stored(self) -> None:
        """propagate_context_keys should store the provided frozenset."""
        options = Options(context={"k": "v"}, propagate_context_keys=frozenset({"k"}))
        assert options.propagate_context_keys == frozenset({"k"})

    def test_propagate_context_keys_validates_subset(self) -> None:
        """propagate_context_keys must be a subset of context keys, else ValueError."""
        with pytest.raises(ValueError, match="propagate_context_keys"):
            Options(context={"a": 1}, propagate_context_keys=frozenset({"b"}))

    def test_propagate_context_keys_not_in_hash(self) -> None:
        """propagate_context_keys should not affect hash (same group+context, different propagate)."""
        o1 = Options(context={"k": "v"}, propagate_context_keys=frozenset({"k"}))
        o2 = Options(context={"k": "v"})
        assert hash(o1) == hash(o2)

    def test_propagate_context_keys_not_in_eq(self) -> None:
        """propagate_context_keys should not affect equality (same group+context, different propagate)."""
        o1 = Options(context={"k": "v"}, propagate_context_keys=frozenset({"k"}))
        o2 = Options(context={"k": "v"})
        assert o1 == o2

    def test_propagate_context_keys_deepcopy(self) -> None:
        """deepcopy should preserve propagate_context_keys."""
        o = Options(context={"k": "v"}, propagate_context_keys=frozenset({"k"}))
        o2 = deepcopy(o)
        assert o2.propagate_context_keys == frozenset({"k"})


class TestContextPropagationIntegration:
    """Integration tests for context propagation through merge_options."""

    def test_merge_options_propagates_specified_context_keys(self) -> None:
        """Test that merge_options propagates only specified context keys."""
        from mloda.core.abstract_plugins.components.feature_collection import Features
        from mloda.user import Feature

        dependency_options = Options(
            context={"session_id": "abc", "algo": "sum"},
            propagate_context_keys=frozenset({"session_id"}),
        )

        feature = Feature("input_feature")

        fc = Features([])
        fc.merge_options(feature, dependency_options)

        assert feature.options.context["session_id"] == "abc"
        assert "algo" not in feature.options.context


class TestOptionsNestedStructureHashing:
    """
    Test suite for Options hashing with nested structures.

    WHY THIS EXISTS:
    Options.group may contain nested dicts, lists, or sets for complex configuration.
    The _make_hashable helper recursively converts these to hashable equivalents,
    enabling proper hashing and use in sets/dicts for feature group resolution.
    """

    def test_hash_with_simple_values_baseline(self) -> None:
        """
        Baseline test: Simple hashable values should work.

        This test should PASS with the current implementation.
        """
        options = Options(group={"key": "value", "number": 42})
        result = hash(options)
        assert isinstance(result, int)

    def test_hash_with_nested_dict_in_group(self) -> None:
        """
        Test hashing Options with nested dict in group.

        The current implementation fails with TypeError: unhashable type: 'dict'
        because frozenset cannot hash dict values.
        """
        options = Options(group={"nested": {"key": "value"}})
        result = hash(options)
        assert isinstance(result, int)

    def test_hash_with_list_in_group(self) -> None:
        """
        Test hashing Options with list in group.

        The current implementation fails with TypeError: unhashable type: 'list'
        because frozenset cannot hash list values.
        """
        options = Options(group={"list": [1, 2, 3]})
        result = hash(options)
        assert isinstance(result, int)

    def test_hash_with_set_in_group(self) -> None:
        """
        Test hashing Options with set in group.

        The current implementation fails with TypeError: unhashable type: 'set'
        because frozenset cannot hash set values.
        """
        options = Options(group={"set_value": {1, 2, 3}})
        result = hash(options)
        assert isinstance(result, int)

    def test_equal_options_with_nested_dict_produce_equal_hashes(self) -> None:
        """
        Test that equal Options with nested structures produce equal hashes.

        Two Options with identical nested dicts should have the same hash.
        """
        options1 = Options(group={"nested": {"key": "value", "other": 123}})
        options2 = Options(group={"nested": {"key": "value", "other": 123}})

        assert options1 == options2
        assert hash(options1) == hash(options2)

    def test_equal_options_with_nested_list_produce_equal_hashes(self) -> None:
        """
        Test that equal Options with nested lists produce equal hashes.

        Two Options with identical lists should have the same hash.
        """
        options1 = Options(group={"items": [1, 2, 3]})
        options2 = Options(group={"items": [1, 2, 3]})

        assert options1 == options2
        assert hash(options1) == hash(options2)

    def test_deeply_nested_structure_hashes_successfully(self) -> None:
        """
        Test hashing Options with deeply nested structures.

        Complex nested structures combining dicts and lists should be hashable.
        """
        options = Options(
            group={
                "config": {
                    "nested": {"deep": {"value": 42}},
                    "list_of_dicts": [{"a": 1}, {"b": 2}],
                }
            }
        )
        result = hash(options)
        assert isinstance(result, int)
