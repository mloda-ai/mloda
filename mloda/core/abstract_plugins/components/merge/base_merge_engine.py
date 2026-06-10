from abc import ABC
from typing import Any, Optional, final

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig, JoinType, Link


class BaseMergeEngine(ABC):
    """
    Abstract base class for merge operations.

    This class defines the structure for implementing various types of merge operations
    between two datasets, based on the specified join type. Subclasses are expected to
    implement the merge methods for specific join types as needed.
    """

    def __init__(self, framework_connection: Optional[Any] = None) -> None:
        """
        Initialize the merge engine.

        Args:
            framework_connection: Optional connection object from the compute framework.
                                Some frameworks (e.g., DuckDB, Spark) need to share their
                                connection with merge engines for data consistency.
        """
        self.framework_connection = framework_connection

    def check_import(self) -> None:
        """
        Convenience method to check if the necessary imports are available. This is important for ensuring that not
        installed modules don't break the framework.

        This gets called in the final merge.

        Example:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        """
        pass

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType inner is not yet implemented in {self.__class__.__name__}")

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType left is not yet implemented in {self.__class__.__name__}")

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType right is not yet implemented in {self.__class__.__name__}")

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType full outer is not yet implemented in {self.__class__.__name__}")

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType append is not yet implemented in {self.__class__.__name__}")

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        raise ValueError(f"JoinType union is not yet implemented in {self.__class__.__name__}")

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        raise ValueError(f"JoinType asof is not yet implemented in {self.__class__.__name__}")

    def validate_asof_time_columns(self, left_data: Any, right_data: Any, asof_config: AsOfJoinConfig) -> None:
        """Guard that as-of time columns are ordered (datetime / numeric / timedelta).

        Raises a clear ValueError naming the offending column instead of letting the
        backend surface a cryptic low-level dtype error. This is the seam that opt-in
        coercion (issue #513) will build on.
        """
        for data, column, side in (
            (left_data, asof_config.left_time_column, "left"),
            (right_data, asof_config.right_time_column, "right"),
        ):
            if not self._asof_time_column_is_ordered(data, column):
                raise ValueError(
                    f"As-of {side} time column '{column}' must be datetime, numeric, or timedelta, "
                    f"but is a non-ordered (e.g. string/object) dtype; cast it before joining."
                )

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        """Return True if `column` in `data` has an ordered (datetime/numeric/timedelta) dtype.

        Overridden per engine with framework-native dtype detection. Default raises so an
        asof-capable engine cannot silently skip the guard.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _asof_time_column_is_ordered for as-of joins."
        )

    @final
    def merge(self, left_data: Any, right_data: Any, link: Link) -> Any:
        self.check_import()

        jointype = link.jointype
        left_index = link.left_index
        right_index = link.right_index

        if jointype not in (JoinType.APPEND, JoinType.UNION):
            if len(left_index.index) != len(right_index.index):
                raise ValueError(
                    f"Left and right index lengths must match for join. "
                    f"Left has {len(left_index.index)} columns {left_index.index}, "
                    f"right has {len(right_index.index)} columns {right_index.index}."
                )

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
        elif jointype == JoinType.ASOF:
            if link.asof_config is None:
                raise ValueError("ASOF join requires a Link carrying an asof_config.")
            return self.merge_asof(left_data, right_data, left_index, right_index, link.asof_config)
        else:
            raise ValueError(f"JoinType {jointype} is not yet implemented {self.__class__.__name__}")
