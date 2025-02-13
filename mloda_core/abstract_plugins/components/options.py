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
    """

    def __init__(self, data: Optional[dict[str, Any]] = None) -> None:
        self.data = data or {}

    def add(self, key: str, value: Any) -> None:
        if key in self.data:
            raise ValueError(f"Key {key} already exists in options.")

        self.data[key] = value

    def __hash__(self) -> int:
        return hash(frozenset(self.data.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Options):
            return False
        return self.data == other.data

    def get(self, key: str) -> Any:
        return self.data.get(key, None)

    def __str__(self) -> str:
        return str(self.data)

    def update_considering_mloda_source(self, other: "Options") -> None:
        """
        Updates the options object with data from another Options object, excluding the mloda_source_feature key.

        The mloda_source_feature key is excluded to preserve the parent feature source, as it is not relevant to the child feature.
        """

        exclude_key = DefaultOptionKeys.mloda_source_feature

        other_data_copy = other.data.copy()
        if exclude_key in other_data_copy and exclude_key in self.data:
            del other_data_copy[exclude_key]

        self.data.update(other_data_copy)
