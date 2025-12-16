from __future__ import annotations
from typing import Any, Tuple


class Index:
    """
    Documentation Index:

    This class is used to define the indexes used to merge datasets based on different primary sources.

    The index is a tuple of strings, which are the column names of the primary source.
    The tuples represent the multi index. (e.g. ("a", "b") is a multi index with two columns)
    """

    def __init__(self, index: Tuple[str, ...]) -> None:
        self.index = index

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Index):
            raise Exception(f"Cannot compare Index with {type(other)}.")
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def is_a_part_of_(self, other: Index) -> bool:
        len_index = len(self.index)

        # index is larger
        if len_index > len(other.index):
            return False

        for cnt, part in enumerate(other.index):
            if cnt > len_index - 1:
                break
            if part != self.index[cnt]:
                return False

        return True

    def __str__(self) -> str:
        return f"{self.index}"

    def is_multi_index(self) -> bool:
        return len(self.index) > 1
