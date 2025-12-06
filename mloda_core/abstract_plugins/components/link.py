from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum
from uuid import uuid4
from typing import Any, Dict, Optional, Set, Tuple, Type, Union


from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.validators.link_validator import LinkValidator


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


class Link:
    """
    Defines a join relationship between two feature groups.

    Args:
        jointype: Type of join operation (inner, left, right, outer, append, union).
        left: JoinSpec for the left side of the join.
        right: JoinSpec for the right side of the join.
        left_pointer: Optional dict to distinguish left instance in self-joins.
            Must match key-value pairs in the left feature's options.
        right_pointer: Optional dict to distinguish right instance in self-joins.
            Must match key-value pairs in the right feature's options.

    Example:
        >>> # Simple join using string index (single column)
        >>> Link.inner(JoinSpec(UserFG, "user_id"), JoinSpec(OrderFG, "user_id"))
        >>>
        >>> # Multi-column join using tuple index
        >>> Link.inner(JoinSpec(UserFG, ("id", "date")), JoinSpec(OrderFG, ("user_id", "order_date")))
        >>>
        >>> # Self-join with pointers
        >>> Link("inner", JoinSpec(UserFG, "user_id"), JoinSpec(UserFG, "user_id"),
        ...      left_pointer={"side": "manager"},
        ...      right_pointer={"side": "employee"})

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
        left_pointer: Optional[Dict[str, Any]] = None,
        right_pointer: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.jointype = JoinType(jointype) if isinstance(jointype, str) else jointype
        self.left_feature_group = left.feature_group
        self.right_feature_group = right.feature_group
        self.left_index = left.index
        self.right_index = right.index
        self.left_pointer = left_pointer
        self.right_pointer = right_pointer

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
