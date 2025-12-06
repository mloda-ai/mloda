from typing import Any, Dict, Set


class OptionsValidator:
    """Validates Options configuration consistency."""

    @staticmethod
    def validate_no_duplicate_keys(group: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Ensure no key exists in both group and context.

        Raises ValueError if any key exists in both, with duplicate keys in message.
        """
        duplicate_keys = set(group.keys()) & set(context.keys())
        if duplicate_keys:
            raise ValueError(f"Keys cannot exist in both group and context: {duplicate_keys}")

    @staticmethod
    def validate_can_add_to_group(key: str, value: Any, group: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Validate that a key can be added to group.

        Checks:
        1. If key exists in group with different value -> ValueError (include key in message)
        2. If key exists in context -> ValueError (include key in message)
        """
        if key in group:
            if value != group[key]:
                raise ValueError(f"Key {key} already exists in group options with a different value: {group[key]}")
        if key in context:
            raise ValueError(f"Key {key} already exists in context options. Cannot add to group.")

    @staticmethod
    def validate_can_add_to_context(key: str, value: Any, group: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Validate that a key can be added to context.

        Checks:
        1. If key exists in context with different value -> ValueError (include key in message)
        2. If key exists in group -> ValueError (include key in message)
        """
        if key in context:
            if value != context[key]:
                raise ValueError(f"Key {key} already exists in context options with a different value: {context[key]}")
        if key in group:
            raise ValueError(f"Key {key} already exists in group options. Cannot add to context.")

    @staticmethod
    def validate_no_group_context_conflicts(other_group_keys: Set[str], self_context_keys: Set[str]) -> None:
        """
        Validate no conflicts between other's group keys and self's context keys.

        Raises ValueError if any key exists in both, with conflicting keys in message.
        """
        conflicting_keys = other_group_keys & self_context_keys
        if conflicting_keys:
            raise ValueError(f"Cannot update group: keys already exist in context: {conflicting_keys}")
