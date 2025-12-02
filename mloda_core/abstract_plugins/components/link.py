from __future__ import annotations

from enum import Enum
from uuid import uuid4
from typing import Any, Dict, Optional, Set, Tuple, Type, Union


from mloda_core.abstract_plugins.components.index.index import Index


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


class Link:
    """
    Defines a join relationship between two feature groups.

    Args:
        jointype: Type of join operation (inner, left, right, outer, append, union).
        left: Tuple of (FeatureGroup class, Index) for the left side.
        right: Tuple of (FeatureGroup class, Index) for the right side.
        left_pointer: Optional dict to distinguish left instance in self-joins.
            Must match key-value pairs in the left feature's options.
        right_pointer: Optional dict to distinguish right instance in self-joins.
            Must match key-value pairs in the right feature's options.

    Example:
        >>> # Normal join
        >>> Link("inner", (UserFG, user_idx), (OrderFG, order_idx))
        >>>
        >>> # Self-join with pointers
        >>> Link("inner", (UserFG, idx), (UserFG, idx),
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
        left: Tuple[Type[Any], Index],  # Any is AbstractFeatureGroup
        right: Tuple[Type[Any], Index],  # Any is AbstractFeatureGroup
        left_pointer: Optional[Dict[str, Any]] = None,
        right_pointer: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.jointype = JoinType(jointype) if isinstance(jointype, str) else jointype
        self.left_feature_group = left[0]
        self.right_feature_group = right[0]
        self.left_index = left[1]
        self.right_index = right[1]
        self.left_pointer = left_pointer
        self.right_pointer = right_pointer

        self.uuid = uuid4()

    def __str__(self) -> str:
        return f"{self.jointype.value} {self.left_feature_group.get_class_name()} {self.left_index} {self.right_feature_group.get_class_name()} {self.right_index} {self.uuid}"

    @classmethod
    def inner(
        cls,
        left: Tuple[Type[Any], Index],
        right: Tuple[Type[Any], Index],
    ) -> Link:
        return cls(JoinType.INNER, left, right)

    @classmethod
    def left(
        cls,
        left: Tuple[Type[Any], Index],
        right: Tuple[Type[Any], Index],
    ) -> Link:
        return cls(JoinType.LEFT, left, right)

    @classmethod
    def right(
        cls,
        left: Tuple[Type[Any], Index],
        right: Tuple[Type[Any], Index],
    ) -> Link:
        return cls(JoinType.RIGHT, left, right)

    @classmethod
    def outer(
        cls,
        left: Tuple[Type[Any], Index],
        right: Tuple[Type[Any], Index],
    ) -> Link:
        return cls(JoinType.OUTER, left, right)

    @classmethod
    def append(
        cls,
        left: Tuple[Type[Any], Index],
        right: Tuple[Type[Any], Index],
    ) -> Link:
        return cls(JoinType.APPEND, left, right)

    @classmethod
    def union(
        cls,
        left: Tuple[Type[Any], Index],
        right: Tuple[Type[Any], Index],
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

    @staticmethod
    def validate(links: Optional[Set[Link]] = None) -> None:
        if links is None:
            return

        for i_link in links:
            if i_link.jointype not in JoinType:
                raise ValueError(f"Join type {i_link.jointype} is not supported")

            for j_link in links:
                if i_link == j_link:
                    continue

                # case: A B and B A -> is not clear which join to use
                # We exclude here append and union, because they are not directional.
                if (
                    i_link.left_feature_group == j_link.right_feature_group
                    and i_link.right_feature_group == j_link.left_feature_group
                    and i_link.jointype not in [JoinType.APPEND, JoinType.UNION]
                ):
                    raise ValueError(
                        f"Link {i_link} and {j_link} have at least two different defined joins. Please remove one."
                    )

                # case: Multiple different join types between two feature groups
                if (
                    i_link.left_feature_group == j_link.left_feature_group
                    and i_link.right_feature_group == j_link.right_feature_group
                    and i_link.jointype != j_link.jointype
                ):
                    raise ValueError(
                        f"Link {i_link} and {j_link} have different join types for the same feature groups. Please remove one."
                    )

                # case: Multiple right joins
                # For now, only small right joins are supported. Lets see if any use case will need this in future.
                if i_link.jointype == JoinType.RIGHT:
                    if (
                        i_link.left_feature_group == j_link.left_feature_group
                        or i_link.left_feature_group == j_link.right_feature_group
                    ):
                        raise ValueError(
                            f"Link {i_link} and {j_link} have multiple right joins for the same feature group on the left side or switching from left to right side although using right join. Please reconsider your joinlogic and if possible, use left joins instead of rightjoins. This will currently break the planner or during execution."
                        )
