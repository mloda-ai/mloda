from typing import Any, Optional

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class Options:
    """
    Options can be passed into the feature, allowing arbitrary variables to be used.
    This enables configuration:
    - at request time
    - when defining input features of a feature_group.

    At-request options are forwarded to child features. This allows configuring children features by:
    - request feature options
    - defining input features of the feature_group.

    New Architecture (group/context separation):
    - group: Parameters that require Feature Groups to have independent resolved feature objects
    - context: Contextual parameters that don't affect Feature Group resolution/splitting

    During migration: All existing options are moved to 'group' to maintain current behavior.
    Future optimization: Move appropriate parameters from 'group' to 'context' for better performance.

    Constraint: A key cannot exist in both group and context simultaneously.
    """

    def __init__(
        self,
        data: Optional[dict[str, Any]] = None,
        group: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        # Handle different initialization patterns
        if data is not None:
            # Legacy initialization: Options(dict) -> move all to group for backward compatibility
            if group is not None or context is not None:
                raise ValueError("Cannot specify both 'data' and 'group'/'context' parameters")
            self.group = data.copy()
            self.context = {}
        else:
            # New initialization: Options(group=dict, context=dict)
            self.group = group or {}
            self.context = context or {}

        self._validate_no_duplicate_keys_in_group_and_context()

    def _validate_no_duplicate_keys_in_group_and_context(self) -> None:
        """Ensure no key exists in both group and context."""
        duplicate_keys = set(self.group.keys()) & set(self.context.keys())
        if duplicate_keys:
            raise ValueError(f"Keys cannot exist in both group and context: {duplicate_keys}")

    @property
    def data(self) -> dict[str, Any]:
        """
        Legacy property for backward compatibility.
        Returns group data to maintain existing behavior during migration.

        Note: This will be deprecated once migration to group/context is complete.
        """
        return self.group

    def add(self, key: str, value: Any) -> None:
        """
        Legacy method for backward compatibility.
        Adds to group to maintain existing behavior during migration.

        Possibility that we keep this as default method for adding options in the future.
        """
        self.add_to_group(key, value)

    def add_to_group(self, key: str, value: Any) -> None:
        """Add parameter to group (affects Feature Group resolution/splitting)."""
        self.validate_that_key_does_not_exist(key)

        self.group[key] = value

    def add_to_context(self, key: str, value: Any) -> None:
        """Add parameter to context (metadata only, doesn't affect splitting)."""
        self.validate_that_key_does_not_exist(key)

        self.context[key] = value

    def validate_that_key_does_not_exist(self, key: str) -> None:
        if key in self.context:
            raise ValueError(f"Key {key} already exists in context options.")
        if key in self.group:
            raise ValueError(f"Key {key} already exists in group options. Cannot add to context.")

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
        Legacy method for backward compatibility.
        Searches group first, then context for the key.
        """
        if key in self.group:
            return self.group[key]
        return self.context.get(key, None)

    def __str__(self) -> str:
        return f"Options(group={self.group}, context={self.context})"

    def update_considering_mloda_source(self, other: "Options") -> None:
        """
        Updates the options object with data from another Options object, excluding the mloda_source_feature key.

        The mloda_source_feature key is excluded to preserve the parent feature source, as it is not relevant to the child feature.

        During migration: Updates group parameters to maintain existing behavior.
        """

        exclude_key = DefaultOptionKeys.mloda_source_feature

        # Update group parameters (maintaining existing behavior)
        other_group_copy = other.group.copy()
        if exclude_key in other_group_copy and exclude_key in self.group:
            del other_group_copy[exclude_key]

        # Check for conflicts before updating
        conflicting_keys = set(other_group_copy.keys()) & set(self.context.keys())
        if conflicting_keys:
            raise ValueError(f"Cannot update group: keys already exist in context: {conflicting_keys}")

        self.group.update(other_group_copy)
