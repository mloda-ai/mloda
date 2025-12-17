from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum
from uuid import uuid4
from typing import Any, Dict, Optional, Tuple, Type, Union


from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.validators.link_validator import LinkValidator


class JoinType(Enum):
    """
    Enum defining types of dataset merge operations.

    Attributes:
        INNER: Includes rows with matching keys from both datasets.
        LEFT: Includes all rows from the left dataset, with matches from the right.
        RIGHT: Includes all rows from the right dataset, with matches from the left.
        OUTER: Includes all rows from both datasets, filling unmatched values with nulls.
        APPEND: Stacks datasets vertically, preserving all rows from both.
        UNION: Combines datasets, removing duplicate rows.
    """

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    APPEND = "append"
    UNION = "union"


class JoinSpec:
    """Specification for one side of a join operation.

    Args:
        feature_group: The feature group class for this side of the join.
        index: Join column(s) - can be:
            - str: single column name, e.g., "id"
            - Tuple[str, ...]: multiple columns, e.g., ("col1", "col2")
            - Index: explicit Index object
    """

    feature_group: Type[Any]
    index: Index

    def __init__(self, feature_group: Type[Any], index: Union[Index, Tuple[str, ...], str]) -> None:
        """Create JoinSpec, converting index input to Index if needed."""
        if isinstance(index, str):
            LinkValidator.validate_index_not_empty(index, "Index column name")
            index = Index((index,))
        elif isinstance(index, tuple):
            LinkValidator.validate_index_not_empty(index, "Index tuple")
            index = Index(index)

        object.__setattr__(self, "feature_group", feature_group)
        object.__setattr__(self, "index", index)

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field '{name}'")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, JoinSpec):
            return False
        return self.feature_group == other.feature_group and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.feature_group, self.index))


def _get_index_from_feature_group(
    feature_group: Type[Any],
    index_position: int,
    side_name: str,
) -> Index:
    """Extract Index from feature group's index_columns().

    Args:
        feature_group: The feature group class
        index_position: Which index to use (0-based)
        side_name: "left" or "right" for error messages

    Returns:
        The Index at the specified position

    Raises:
        ValueError: If index_columns() returns None or empty list
        IndexError: If index_position is out of range
    """
    index_columns = feature_group.index_columns()

    if index_columns is None:
        raise ValueError(
            f"{side_name.capitalize()} feature group {feature_group.__name__} does not define index_columns()"
        )

    if not index_columns:
        raise ValueError(
            f"{side_name.capitalize()} feature group {feature_group.__name__}.index_columns() returned empty list"
        )

    if index_position < 0 or index_position >= len(index_columns):
        raise IndexError(
            f"{side_name}_index {index_position} out of range for "
            f"{feature_group.__name__} (has {len(index_columns)} indexes)"
        )

    result: Index = index_columns[index_position]
    return result


class Link:
    """
    Defines a join relationship between two feature groups.

    Args:
        jointype: Type of join operation (inner, left, right, outer, append, union).
        left: JoinSpec for the left side of the join.
        right: JoinSpec for the right side of the join.
        self_left_alias: Optional dict to distinguish left instance in self-joins.
            Must match key-value pairs in the left feature's options. Named after
            SQL table aliases used in self-joins (e.g., SELECT * FROM t1 AS a JOIN t1 AS b).
        self_right_alias: Optional dict to distinguish right instance in self-joins.
            Must match key-value pairs in the right feature's options.

    Factory Methods:
        There are two styles of factory methods available:

        **Standard methods** - require explicit JoinSpec objects:
            - Link.inner(left_joinspec, right_joinspec)
            - Link.left(left_joinspec, right_joinspec)
            - Link.right(left_joinspec, right_joinspec)
            - Link.outer(left_joinspec, right_joinspec)
            - Link.append(left_joinspec, right_joinspec)
            - Link.union(left_joinspec, right_joinspec)

        **Convenience _on methods** - accept feature groups directly and derive
        JoinSpecs automatically from index_columns():
            - Link.inner_on(LeftFG, RightFG, left_index=0, right_index=0)
            - Link.left_on(LeftFG, RightFG, left_index=0, right_index=0)
            - Link.right_on(LeftFG, RightFG, left_index=0, right_index=0)
            - Link.outer_on(LeftFG, RightFG, left_index=0, right_index=0)
            - Link.append_on(LeftFG, RightFG, left_index=0, right_index=0)
            - Link.union_on(LeftFG, RightFG, left_index=0, right_index=0)

        The _on methods require feature groups to have index_columns() defined.
        Use left_index/right_index to select which index when multiple are available.

    Example:
        >>> # Verbose: explicit JoinSpec with index
        >>> Link.inner(JoinSpec(UserFG, "user_id"), JoinSpec(OrderFG, "user_id"))
        >>>
        >>> # Convenient: derive index from feature group's index_columns()
        >>> Link.inner_on(UserFG, OrderFG)
        >>>
        >>> # Multi-index selection (use second index from left, first from right)
        >>> Link.inner_on(UserFG, OrderFG, left_index=1, right_index=0)
        >>>
        >>> # Multi-column join using tuple index
        >>> Link.inner(JoinSpec(UserFG, ("id", "date")), JoinSpec(OrderFG, ("user_id", "order_date")))
        >>>
        >>> # Self-join with aliases (like SQL table aliases)
        >>> Link.inner_on(UserFG, UserFG,
        ...               self_left_alias={"side": "manager"},
        ...               self_right_alias={"side": "employee"})

    Polymorphic Matching:
        Links support inheritance-based matching, allowing a link defined with base
        classes to automatically apply to subclasses. The matching follows these rules:

        1. **Exact match first**: If a link's feature groups exactly match the classes
           being joined, it takes priority over any polymorphic matches.

        2. **Balanced inheritance**: For polymorphic matches, both sides must have the
           same inheritance distance. This prevents sibling class mismatches.

           Example - Given hierarchy:
               BaseFeatureGroup
               ├── ChildA
               └── ChildB

           Link(BaseFeatureGroup, BaseFeatureGroup) will match:
           - (ChildA, ChildA) ✓  - both sides distance=1
           - (ChildB, ChildB) ✓  - both sides distance=1
           - (ChildA, ChildB) ✗  - rejected: siblings, not balanced inheritance

        3. **Most specific wins**: Among valid matches, the link closest in the
           inheritance hierarchy is selected.
    """

    def __init__(
        self,
        jointype: Union[JoinType, str],
        left: JoinSpec,
        right: JoinSpec,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.jointype = JoinType(jointype) if isinstance(jointype, str) else jointype
        self.left_feature_group = left.feature_group
        self.right_feature_group = right.feature_group
        self.left_index = left.index
        self.right_index = right.index
        self.self_left_alias = self_left_alias
        self.self_right_alias = self_right_alias

        self.uuid = uuid4()

    def __str__(self) -> str:
        return f"{self.jointype.value} {self.left_feature_group.get_class_name()} {self.left_index} {self.right_feature_group.get_class_name()} {self.right_index} {self.uuid}"

    @classmethod
    def inner(
        cls,
        left: JoinSpec,
        right: JoinSpec,
    ) -> Link:
        return cls(JoinType.INNER, left, right)

    @classmethod
    def left(
        cls,
        left: JoinSpec,
        right: JoinSpec,
    ) -> Link:
        return cls(JoinType.LEFT, left, right)

    @classmethod
    def right(
        cls,
        left: JoinSpec,
        right: JoinSpec,
    ) -> Link:
        return cls(JoinType.RIGHT, left, right)

    @classmethod
    def outer(
        cls,
        left: JoinSpec,
        right: JoinSpec,
    ) -> Link:
        return cls(JoinType.OUTER, left, right)

    @classmethod
    def append(
        cls,
        left: JoinSpec,
        right: JoinSpec,
    ) -> Link:
        return cls(JoinType.APPEND, left, right)

    @classmethod
    def union(
        cls,
        left: JoinSpec,
        right: JoinSpec,
    ) -> Link:
        return cls(JoinType.UNION, left, right)

    @classmethod
    def inner_on(
        cls,
        left: Type[Any],
        right: Type[Any],
        left_index: int = 0,
        right_index: int = 0,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> "Link":
        """Create INNER join using feature groups' index_columns()."""
        left_idx = _get_index_from_feature_group(left, left_index, "left")
        right_idx = _get_index_from_feature_group(right, right_index, "right")
        return cls(
            JoinType.INNER,
            JoinSpec(left, left_idx),
            JoinSpec(right, right_idx),
            self_left_alias,
            self_right_alias,
        )

    @classmethod
    def left_on(
        cls,
        left: Type[Any],
        right: Type[Any],
        left_index: int = 0,
        right_index: int = 0,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> "Link":
        """Create LEFT join using feature groups' index_columns()."""
        left_idx = _get_index_from_feature_group(left, left_index, "left")
        right_idx = _get_index_from_feature_group(right, right_index, "right")
        return cls(
            JoinType.LEFT,
            JoinSpec(left, left_idx),
            JoinSpec(right, right_idx),
            self_left_alias,
            self_right_alias,
        )

    @classmethod
    def right_on(
        cls,
        left: Type[Any],
        right: Type[Any],
        left_index: int = 0,
        right_index: int = 0,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> "Link":
        """Create RIGHT join using feature groups' index_columns()."""
        left_idx = _get_index_from_feature_group(left, left_index, "left")
        right_idx = _get_index_from_feature_group(right, right_index, "right")
        return cls(
            JoinType.RIGHT,
            JoinSpec(left, left_idx),
            JoinSpec(right, right_idx),
            self_left_alias,
            self_right_alias,
        )

    @classmethod
    def outer_on(
        cls,
        left: Type[Any],
        right: Type[Any],
        left_index: int = 0,
        right_index: int = 0,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> "Link":
        """Create OUTER join using feature groups' index_columns()."""
        left_idx = _get_index_from_feature_group(left, left_index, "left")
        right_idx = _get_index_from_feature_group(right, right_index, "right")
        return cls(
            JoinType.OUTER,
            JoinSpec(left, left_idx),
            JoinSpec(right, right_idx),
            self_left_alias,
            self_right_alias,
        )

    @classmethod
    def append_on(
        cls,
        left: Type[Any],
        right: Type[Any],
        left_index: int = 0,
        right_index: int = 0,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> "Link":
        """Create APPEND join using feature groups' index_columns()."""
        left_idx = _get_index_from_feature_group(left, left_index, "left")
        right_idx = _get_index_from_feature_group(right, right_index, "right")
        return cls(
            JoinType.APPEND,
            JoinSpec(left, left_idx),
            JoinSpec(right, right_idx),
            self_left_alias,
            self_right_alias,
        )

    @classmethod
    def union_on(
        cls,
        left: Type[Any],
        right: Type[Any],
        left_index: int = 0,
        right_index: int = 0,
        self_left_alias: Optional[Dict[str, Any]] = None,
        self_right_alias: Optional[Dict[str, Any]] = None,
    ) -> "Link":
        """Create UNION join using feature groups' index_columns()."""
        left_idx = _get_index_from_feature_group(left, left_index, "left")
        right_idx = _get_index_from_feature_group(right, right_index, "right")
        return cls(
            JoinType.UNION,
            JoinSpec(left, left_idx),
            JoinSpec(right, right_idx),
            self_left_alias,
            self_right_alias,
        )

    def matches_exact(
        self,
        other_left_feature_group: Type[Any],
        other_right_feature_group: Type[Any],
    ) -> bool:
        """Exact class name match only."""
        left_match: bool = self.left_feature_group.get_class_name() == other_left_feature_group.get_class_name()
        right_match: bool = self.right_feature_group.get_class_name() == other_right_feature_group.get_class_name()
        return left_match and right_match

    def matches_polymorphic(
        self,
        other_left_feature_group: Type[Any],
        other_right_feature_group: Type[Any],
    ) -> bool:
        """Subclass match (inheritance). Returns True if both sides are subclasses."""
        return issubclass(other_left_feature_group, self.left_feature_group) and issubclass(
            other_right_feature_group, self.right_feature_group
        )

    def matches(
        self,
        other_left_feature_group: Type[Any],
        other_right_feature_group: Type[Any],
    ) -> bool:
        """Combined match: exact OR polymorphic."""
        return self.matches_exact(other_left_feature_group, other_right_feature_group) or self.matches_polymorphic(
            other_left_feature_group, other_right_feature_group
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Link):
            return False
        return (
            self.jointype == other.jointype
            and self.left_feature_group.get_class_name() == other.left_feature_group.get_class_name()
            and self.right_feature_group.get_class_name() == other.right_feature_group.get_class_name()
            and self.left_index == other.left_index
            and self.right_index == other.right_index
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.jointype,
                self.left_feature_group.get_class_name(),
                self.right_feature_group.get_class_name(),
                self.left_index,
                self.right_index,
            )
        )
