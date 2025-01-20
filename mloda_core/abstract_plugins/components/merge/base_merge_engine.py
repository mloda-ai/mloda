from abc import ABC
from typing import Any, final

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType


class BaseMergeEngine(ABC):
    """
    Abstract base class for merge operations.

    This class defines the structure for implementing various types of merge operations
    between two datasets, based on the specified join type. Subclasses are expected to
    implement the merge methods for specific join types as needed.
    """

    def check_import(self) -> None:
        """
        Convenience method to check if the necessary imports are available. This is important for ensuring that not
        installed modules dont break the framework.

        This gets called in the final merge.

        Example:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        """
        pass

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType inner are not yet implemented {self.__class__.__name__}")

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType left are not yet implemented {self.__class__.__name__}")

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType right are not yet implemented {self.__class__.__name__}")

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType full outer are not yet implemented {self.__class__.__name__}")

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType append are not yet implemented {self.__class__.__name__}")

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType union are not yet implemented {self.__class__.__name__}")

    @final
    def merge(self, left_data: Any, right_data: Any, jointype: JoinType, left_index: Index, right_index: Index) -> Any:
        self.check_import()

        if jointype == JoinType.INNER:
            return self.merge_inner(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.LEFT:
            return self.merge_left(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.RIGHT:
            return self.merge_right(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.OUTER:
            return self.merge_full_outer(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.APPEND:
            return self.merge_append(left_data, right_data, left_index, right_index)
        elif jointype == JoinType.UNION:
            return self.merge_union(left_data, right_data, left_index, right_index)
        else:
            raise ValueError(f"JoinType {jointype} is not yet implemented {self.__class__.__name__}")
