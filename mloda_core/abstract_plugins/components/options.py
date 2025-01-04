from typing import Any, Optional


class Options:
    """HashableDict

    Documentation Options:

    Options can be passed into the feature, so that we can use arbitrary variables in the feature.
    This means, we can define options in
    - at request
    - at defining input features of feature_group.

    We forward at request options to the child features. This is done by the engine.
    This enables us to configure children features by essentially two mechanism:
    - at request by request feature options
    - at defining input features of feature_group.
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
