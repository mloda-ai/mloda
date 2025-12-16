from typing import Any


class HashableDict:
    def __init__(self, data: dict[Any, Any]) -> None:
        self.data = data

    def __hash__(self) -> int:
        return hash(frozenset(self.data.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableDict):
            return False
        return self.data == other.data

    def items(self) -> Any:
        return self.data.items()
