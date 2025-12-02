from __future__ import annotations

from typing import Any, Dict, Optional, Set, TYPE_CHECKING, cast
from copy import deepcopy

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

if TYPE_CHECKING:
    from mloda_core.abstract_plugins.components.feature import Feature


class Options:
    """
    Configuration container for features with group/context separation.

    Architecture:
    - group: Parameters affecting Feature Group resolution/splitting (used in hashing/equality)
    - context: Metadata parameters that don't affect splitting (excluded from hashing)

    Initialization:
    - Options() - Empty options (both group and context are empty)
    - Options({...}) - Positional dict goes to group
    - Options(group={...}) - Explicit group parameters
    - Options(context={...}) - Explicit context parameters
    - Options(group={...}, context={...}) - Both specified

    Common Methods:
    - .get(key) - Read value (searches group, then context)
    - .set(key, value) - Write value (auto-placement)
    - .items() / .keys() - Iterate over all options
    - key in options - Check existence

    Direct Access (when category matters):
    - .group dict or .add_to_group(key, value)
    - .context dict or .add_to_context(key, value)

    Constraint: A key cannot exist in both group and context simultaneously.

    Examples:
        >>> # Basic usage with positional dict (goes to group)
        >>> opts = Options({"data_source": "prod"})
        >>> opts.group
        {'data_source': 'prod'}

        >>> # Explicit group/context separation
        >>> opts = Options(
        ...     group={"data_source": "prod"},
        ...     context={"debug_mode": True}
        ... )
        >>> opts.get("data_source")
        'prod'
        >>> opts.get("debug_mode")
        True

        >>> # Using helper methods
        >>> opts = Options()
        >>> opts.add_to_group("model_type", "classifier")
        >>> opts.add_to_context("log_level", "INFO")
        >>> "model_type" in opts
        True
    """

    def __init__(
        self,
        group: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self.group = group or {}
        self.context = context or {}
        self._validate_no_duplicate_keys_in_group_and_context()

    def _validate_no_duplicate_keys_in_group_and_context(self) -> None:
        """Ensure no key exists in both group and context."""
        duplicate_keys = set(self.group.keys()) & set(self.context.keys())
        if duplicate_keys:
            raise ValueError(f"Keys cannot exist in both group and context: {duplicate_keys}")

    def add(self, key: str, value: Any) -> None:
        """
        Legacy method for backward compatibility.
        Adds to group to maintain existing behavior during migration.

        Possibility that we keep this as default method for adding options in the future.
        """
        self.add_to_group(key, value)

    def add_to_group(self, key: str, value: Any) -> None:
        """Add parameter to group (affects Feature Group resolution/splitting)."""

        if key in self.group:
            if value != self.group[key]:
                raise ValueError(f"Key {key} already exists in group options with a different value: {self.group[key]}")

        if key in self.context:
            raise ValueError(f"Key {key} already exists in context options. Cannot add to group.")

        self.group[key] = value

    def add_to_context(self, key: str, value: Any) -> None:
        """Add parameter to context (metadata only, doesn't affect splitting)."""

        if key in self.context:
            if value != self.context[key]:
                raise ValueError(
                    f"Key {key} already exists in context options with a different value: {self.context[key]}"
                )

        if key in self.group:
            raise ValueError(f"Key {key} already exists in group options. Cannot add to context.")

        self.context[key] = value

    def __hash__(self) -> int:
        """
        Hash based only on group parameters.
        Context parameters don't affect Feature Group resolution/splitting.
        """
        return hash(frozenset(self.group.items()))

    def __eq__(self, other: object) -> bool:
        """
        Equality based only on group parameters.
        Context parameters don't affect Feature Group resolution/splitting.
        """
        if not isinstance(other, Options):
            return False
        return self.group == other.group

    def get(self, key: str) -> Any:
        """
        Get a value from options, searching group first, then context.

        This is the recommended way to access option values when you don't need
        to distinguish between group and context parameters.
        """
        if key in self.group:
            return self.group[key]
        return self.context.get(key, None)

    def items(self) -> list[tuple[str, Any]]:
        """
        Get all key-value pairs from both group and context.

        Returns a list of tuples containing all options.
        Group options are returned first, followed by context options.
        """
        return list(self.group.items()) + list(self.context.items())

    def keys(self) -> list[str]:
        """
        Get all keys from both group and context.

        Returns a list of all option keys.
        """
        return list(self.group.keys()) + list(self.context.keys())

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in either group or context.

        Supports the 'in' operator: 'key' in options
        """
        return key in self.group or key in self.context

    def set(self, key: str, value: Any) -> None:
        """
        Set a value, automatically placing it in group or context.

        If the key already exists, update it in its current location.
        If the key is new, add it to group by default.
        """
        if key in self.group:
            self.group[key] = value
        elif key in self.context:
            self.context[key] = value
        else:
            # New key, add to group by default
            self.group[key] = value

    def get_in_features(self) -> "frozenset[Feature]":
        val = self.get(DefaultOptionKeys.in_features)

        if not val:
            raise ValueError(
                f"Input features not found in options. Please ensure that the key '{DefaultOptionKeys.in_features}' is set."
            )

        def _convert_to_feature(item: Any) -> "Feature":
            """Convert item to Feature object if possible."""
            if hasattr(item, "get_name"):  # Already a Feature object
                return cast("Feature", item)
            elif isinstance(item, str):
                # Import Feature locally to avoid circular import
                from mloda_core.abstract_plugins.components.feature import Feature

                return Feature(item)
            else:
                raise TypeError(f"Cannot convert {type(item)} to Feature. Expected Feature object or str.")

        if isinstance(val, (list, set, frozenset)):
            return frozenset(_convert_to_feature(item) for item in val)
        elif isinstance(val, str):
            # Handle comma-separated strings
            if "," in val:
                feature_names = [name.strip() for name in val.split(",")]
                return frozenset(_convert_to_feature(name) for name in feature_names)
            else:
                return frozenset([_convert_to_feature(val)])
        elif hasattr(val, "get_name"):  # Handle Feature objects
            return frozenset([_convert_to_feature(val)])
        else:
            raise TypeError(
                f"Unsupported type for source feature: {type(val)}. Expected frozenset, str, list, set, or Feature object."
            )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Options":
        def safe_deepcopy_dict(d: dict[str, Any]) -> dict[str, Any]:
            """Safely deepcopy a dictionary, falling back to shallow copy for unpickleable objects."""
            result = {}
            for key, value in d.items():
                try:
                    result[key] = deepcopy(value, memo)
                except (TypeError, AttributeError, RecursionError):
                    # If the object cannot be pickled/deepcopied or causes recursion, use shallow copy
                    result[key] = value
            return result

        copied_group = safe_deepcopy_dict(self.group)
        copied_context = safe_deepcopy_dict(self.context)
        return Options(group=copied_group, context=copied_context)

    def __str__(self) -> str:
        return f"Options(group={self.group}, context={self.context})"

    def update_with_protected_keys(self, other: "Options", protected_keys: Set[str] | None = None) -> None:
        """
        Updates this Options object with data from another Options object, respecting protected keys.

        Protected keys allow parent and child features in a chain to maintain different values
        without raising conflicts. This is essential for feature chaining where each level
        needs its own configuration for certain parameters.

        Protected keys can be specified in two ways:
        1. Explicitly passed as the protected_keys parameter
        2. Dynamically read from self.get(mloda_feature_chainer_parser_key) for backward compatibility

        Mechanism:
        - Protected keys in 'other' are NOT merged into 'self'
        - This preserves the parent's (self) configuration for those keys
        - Child features can have different values for protected keys without conflict

        Example:
            Parent feature has: in_features="parent_source"
            Child feature has:  in_features="child_source"

            Without protection: ERROR (duplicate key conflict)
            With protection: Both keep their own values (no merge, no error)

        Args:
            other: The Options object to merge from (typically child options)
            protected_keys: Set of keys to protect from merging.
                          If None, uses in_features + any keys from mloda_feature_chainer_parser_key

        Raises:
            ValueError: If non-protected keys conflict between group and context
        """
        # Build protected keys set
        if protected_keys is None:
            # Default: always protect in_features
            protected_keys = {DefaultOptionKeys.in_features}

            # Dynamic: read additional protected keys from mloda_feature_chainer_parser_key
            # This allows feature groups to specify which keys should be protected
            if self.get(DefaultOptionKeys.mloda_feature_chainer_parser_key):
                for key in self.get(DefaultOptionKeys.mloda_feature_chainer_parser_key):
                    protected_keys.add(key)

        # Create a copy of other.group excluding protected keys
        # Protected keys are intentionally skipped to preserve parent's configuration
        other_group_copy = other.group.copy()
        for protected_key in protected_keys:
            if protected_key in other_group_copy:
                del other_group_copy[protected_key]

        # Check for conflicts before updating
        conflicting_keys = set(other_group_copy.keys()) & set(self.context.keys())
        if conflicting_keys:
            raise ValueError(f"Cannot update group: keys already exist in context: {conflicting_keys}")

        self.group.update(other_group_copy)
