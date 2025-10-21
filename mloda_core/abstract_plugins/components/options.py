from typing import Any, Dict, Optional, TYPE_CHECKING
from copy import deepcopy

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

if TYPE_CHECKING:
    from mloda_core.abstract_plugins.components.feature import Feature


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
        Legacy method for backward compatibility.
        Searches group first, then context for the key.
        """
        if key in self.group:
            return self.group[key]
        return self.context.get(key, None)

    def get_source_features(self) -> "frozenset[Feature]":
        val = self.get(DefaultOptionKeys.mloda_source_features)

        if not val:
            raise ValueError(
                f"Source feature not found in options. Please ensure that the key '{DefaultOptionKeys.mloda_source_features}' is set."
            )

        def _convert_to_feature(item: Any) -> "Feature":
            """Convert item to Feature object if possible."""
            if hasattr(item, "get_name"):  # Already a Feature object
                return item  # type: ignore
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

    def update_considering_mloda_source(self, other: "Options") -> None:
        """
        Updates the options object with data from another Options object, excluding the mloda_source_features key.

        The mloda_source_features key is excluded to preserve the parent feature source, as it is not relevant to the child feature.

        During migration: Updates group parameters to maintain existing behavior.
        """

        # Case mloda_source_features
        exclude_keys = set([DefaultOptionKeys.mloda_source_features])

        # Case mloda_feature_chainer_parser_key
        if self.get(DefaultOptionKeys.mloda_feature_chainer_parser_key):
            # If the feature chainer parser key is set, we should not update the group with the source feature.
            for _key in self.get(DefaultOptionKeys.mloda_feature_chainer_parser_key):
                exclude_keys.add(_key)

        # Update group parameters (maintaining existing behavior)
        # We drop here the keys, so that we do not overwrite the excluded keys.
        # That means that given configuration of child features do not overwrite the parent feature configuration.
        other_group_copy = other.group.copy()
        for exclude_key in exclude_keys:
            if exclude_key in other_group_copy:  # and exclude_key in self.group:       -> maybe a bug somewhere else
                # if exclude_key in other_group_copy and exclude_key in self.group:
                del other_group_copy[exclude_key]

        # Check for conflicts before updating
        conflicting_keys = set(other_group_copy.keys()) & set(self.context.keys())
        if conflicting_keys:
            raise ValueError(f"Cannot update group: keys already exist in context: {conflicting_keys}")

        self.group.update(other_group_copy)
