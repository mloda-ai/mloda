import pytest
from typing import Any

from mloda.provider import OptionsValidator, DefaultOptionKeys


class TestValidateNoDuplicateKeys:
    """Test the validate_no_duplicate_keys static method."""

    def test_no_duplicates_passes(self) -> None:
        """Should not raise when no duplicate keys exist between group and context."""
        group = {"key1": "value1", "key2": "value2"}
        context = {"key3": "value3", "key4": "value4"}

        # Act & Assert - should not raise
        OptionsValidator.validate_no_duplicate_keys(group=group, context=context)

    def test_duplicate_keys_raises_value_error(self) -> None:
        """Should raise ValueError when key exists in both group and context."""
        group = {"key1": "value1", "duplicate": "group_value"}
        context = {"key3": "value3", "duplicate": "context_value"}

        # Act & Assert
        with pytest.raises(ValueError):
            OptionsValidator.validate_no_duplicate_keys(group=group, context=context)

    def test_error_message_includes_duplicate_keys(self) -> None:
        """Error message should list the duplicate keys."""
        group = {"key1": "value1", "dup1": "val1", "dup2": "val2"}
        context = {"key3": "value3", "dup1": "other1", "dup2": "other2"}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_no_duplicate_keys(group=group, context=context)

        # Verify error message includes duplicate keys
        error_msg = str(exc_info.value)
        assert "dup1" in error_msg or "dup2" in error_msg

    def test_empty_dicts_pass(self) -> None:
        """Empty dictionaries should not raise any errors."""
        group: dict[str, Any] = {}
        context: dict[str, Any] = {}

        # Act & Assert - should not raise
        OptionsValidator.validate_no_duplicate_keys(group=group, context=context)


class TestValidateCanAddToGroup:
    """Test the validate_can_add_to_group static method."""

    def test_new_key_passes(self) -> None:
        """Should not raise when adding a completely new key."""
        key = "new_key"
        value = "new_value"
        group = {"existing": "value"}
        context = {"other": "data"}

        # Act & Assert - should not raise
        OptionsValidator.validate_can_add_to_group(key=key, value=value, group=group, context=context)

    def test_same_key_same_value_passes(self) -> None:
        """Should not raise when key exists in group with the same value."""
        key = "existing_key"
        value = "same_value"
        group = {"existing_key": "same_value"}
        context = {"other": "data"}

        # Act & Assert - should not raise
        OptionsValidator.validate_can_add_to_group(key=key, value=value, group=group, context=context)

    def test_same_key_different_value_raises(self) -> None:
        """Should raise ValueError when key exists in group with different value."""
        key = "existing_key"
        value = "new_value"
        group = {"existing_key": "old_value"}
        context = {"other": "data"}

        # Act & Assert
        with pytest.raises(ValueError):
            OptionsValidator.validate_can_add_to_group(key=key, value=value, group=group, context=context)

    def test_key_in_context_raises(self) -> None:
        """Should raise ValueError when key already exists in context."""
        key = "context_key"
        value = "some_value"
        group = {"other": "data"}
        context = {"context_key": "existing_value"}

        # Act & Assert
        with pytest.raises(ValueError):
            OptionsValidator.validate_can_add_to_group(key=key, value=value, group=group, context=context)

    def test_error_message_includes_key(self) -> None:
        """Error message should include the problematic key."""
        key = "problem_key"
        value = "new_value"
        group = {"problem_key": "old_value"}
        context: dict[str, Any] = {}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_can_add_to_group(key=key, value=value, group=group, context=context)

        # Verify error message includes the key
        assert key in str(exc_info.value)


class TestValidateCanAddToContext:
    """Test the validate_can_add_to_context static method."""

    def test_new_key_passes(self) -> None:
        """Should not raise when adding a completely new key."""
        key = "new_key"
        value = "new_value"
        group = {"existing": "value"}
        context = {"other": "data"}

        # Act & Assert - should not raise
        OptionsValidator.validate_can_add_to_context(key=key, value=value, group=group, context=context)

    def test_same_key_same_value_passes(self) -> None:
        """Should not raise when key exists in context with the same value."""
        key = "existing_key"
        value = "same_value"
        group = {"other": "data"}
        context = {"existing_key": "same_value"}

        # Act & Assert - should not raise
        OptionsValidator.validate_can_add_to_context(key=key, value=value, group=group, context=context)

    def test_same_key_different_value_raises(self) -> None:
        """Should raise ValueError when key exists in context with different value."""
        key = "existing_key"
        value = "new_value"
        group = {"other": "data"}
        context = {"existing_key": "old_value"}

        # Act & Assert
        with pytest.raises(ValueError):
            OptionsValidator.validate_can_add_to_context(key=key, value=value, group=group, context=context)

    def test_key_in_group_raises(self) -> None:
        """Should raise ValueError when key already exists in group."""
        key = "group_key"
        value = "some_value"
        group = {"group_key": "existing_value"}
        context = {"other": "data"}

        # Act & Assert
        with pytest.raises(ValueError):
            OptionsValidator.validate_can_add_to_context(key=key, value=value, group=group, context=context)

    def test_error_message_includes_key(self) -> None:
        """Error message should include the problematic key."""
        key = "problem_key"
        value = "new_value"
        group: dict[str, Any] = {}
        context = {"problem_key": "old_value"}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_can_add_to_context(key=key, value=value, group=group, context=context)

        # Verify error message includes the key
        assert key in str(exc_info.value)


class TestValidateNoGroupContextConflicts:
    """Test the validate_no_group_context_conflicts static method."""

    def test_no_conflicts_passes(self) -> None:
        """Should not raise when no conflicts exist between sets."""
        other_group_keys = {"key1", "key2", "key3"}
        self_context_keys = {"key4", "key5", "key6"}

        # Act & Assert - should not raise
        OptionsValidator.validate_no_group_context_conflicts(
            other_group_keys=other_group_keys, self_context_keys=self_context_keys
        )

    def test_conflicts_raises_value_error(self) -> None:
        """Should raise ValueError when conflicts exist between sets."""
        other_group_keys = {"key1", "key2", "conflict"}
        self_context_keys = {"key4", "key5", "conflict"}

        # Act & Assert
        with pytest.raises(ValueError):
            OptionsValidator.validate_no_group_context_conflicts(
                other_group_keys=other_group_keys, self_context_keys=self_context_keys
            )

    def test_error_message_includes_conflicting_keys(self) -> None:
        """Error message should list the conflicting keys."""
        other_group_keys = {"key1", "conflict1", "conflict2"}
        self_context_keys = {"key4", "conflict1", "conflict2"}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_no_group_context_conflicts(
                other_group_keys=other_group_keys, self_context_keys=self_context_keys
            )

        # Verify error message includes conflicting keys
        error_msg = str(exc_info.value)
        assert "conflict1" in error_msg or "conflict2" in error_msg

    def test_empty_sets_pass(self) -> None:
        """Empty sets should not raise any errors."""
        other_group_keys: set[str] = set()
        self_context_keys: set[str] = set()

        # Act & Assert - should not raise
        OptionsValidator.validate_no_group_context_conflicts(
            other_group_keys=other_group_keys, self_context_keys=self_context_keys
        )


class TestValidateContextKeys:
    """Test the validate_context_keys static method (context-key schema validation)."""

    def test_unknown_key_with_near_match_raises_with_suggestion(self) -> None:
        """Unknown context key close to a schema key raises ValueError with a 'did you mean' hint."""
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_context_keys(
                feature_name="my_feature",
                context={"partiton_by": "col"},
                schema={"partition_by": str},
            )

        error_msg = str(exc_info.value)
        assert "context key 'partiton_by'" in error_msg
        assert "did you mean 'partition_by'" in error_msg
        assert "my_feature" in error_msg

    def test_unknown_key_without_near_match_raises(self) -> None:
        """Unknown context key with no close match still raises ValueError mentioning the key."""
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_context_keys(
                feature_name="my_feature",
                context={"zzz_unrelated": 1},
                schema={"partition_by": str},
            )

        error_msg = str(exc_info.value)
        assert "context key 'zzz_unrelated'" in error_msg

    def test_all_valid_keys_pass(self) -> None:
        """All context keys present in schema and type-correct should not raise."""
        # Act & Assert - should not raise
        OptionsValidator.validate_context_keys(
            feature_name="my_feature",
            context={"partition_by": "col", "frame_size": 10},
            schema={"partition_by": str, "frame_size": int},
        )

    def test_framework_reserved_key_allowed(self) -> None:
        """A framework-reserved key (DefaultOptionKeys) is allowed even when absent from schema."""
        # Act & Assert - should not raise
        OptionsValidator.validate_context_keys(
            feature_name="my_feature",
            context={DefaultOptionKeys.in_features.value: "some_source"},
            schema={"partition_by": str},
        )

    def test_type_mismatch_raises(self) -> None:
        """A context value whose type does not match the schema type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_context_keys(
                feature_name="my_feature",
                context={"frame_size": "10"},
                schema={"frame_size": int},
            )

        error_msg = str(exc_info.value)
        assert "frame_size" in error_msg

    def test_type_match_passes(self) -> None:
        """A context value whose type matches the schema type does not raise."""
        # Act & Assert - should not raise
        OptionsValidator.validate_context_keys(
            feature_name="my_feature",
            context={"frame_size": 10},
            schema={"frame_size": int},
        )

    def test_none_expected_type_skips_type_check(self) -> None:
        """A schema entry mapping to None skips the type check for that key."""
        # Act & Assert - should not raise even though value is an int
        OptionsValidator.validate_context_keys(
            feature_name="my_feature",
            context={"order_by": 123},
            schema={"order_by": None},
        )

    def test_extra_allowed_keys_exempts_key(self) -> None:
        """A key listed in extra_allowed_keys is exempt from the unknown-key error."""
        # Act & Assert - should not raise
        OptionsValidator.validate_context_keys(
            feature_name="my_feature",
            context={"partition_by": "col", "propagated_x": "v"},
            schema={"partition_by": str},
            extra_allowed_keys={"propagated_x"},
        )

    def test_empty_schema_rejects_non_reserved_key(self) -> None:
        """A declared-but-empty schema accepts nothing custom: a non-reserved key raises."""
        with pytest.raises(ValueError) as exc_info:
            OptionsValidator.validate_context_keys(
                feature_name="my_feature",
                context={"anything": 1},
                schema={},
            )

        error_msg = str(exc_info.value)
        assert "context key 'anything'" in error_msg
