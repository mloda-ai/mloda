from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.link import Link


class LinkValidator:
    @staticmethod
    def validate_index_not_empty(index: Union[str, Tuple[str, ...]], context: str = "index") -> None:
        if not index:
            raise ValueError(f"{context} cannot be empty")

    @staticmethod
    def validate_join_type(jointype: Any) -> None:
        from mloda.core.abstract_plugins.components.link import JoinType

        if not isinstance(jointype, JoinType):
            raise ValueError(f"Join type {jointype} is not supported")

    @staticmethod
    def validate_no_double_joins(links: Set["Link"]) -> None:
        from mloda.core.abstract_plugins.components.link import JoinType

        for i_link in links:
            for j_link in links:
                if i_link == j_link:
                    continue
                if (
                    i_link.left_feature_group == j_link.right_feature_group
                    and i_link.right_feature_group == j_link.left_feature_group
                    and i_link.jointype not in [JoinType.APPEND, JoinType.UNION]
                ):
                    raise ValueError(
                        f"Link {i_link} and {j_link} have at least two different defined joins. Please remove one."
                    )

    @staticmethod
    def validate_no_conflicting_join_types(links: Set["Link"]) -> None:
        for i_link in links:
            for j_link in links:
                if i_link == j_link:
                    continue
                if (
                    i_link.left_feature_group == j_link.left_feature_group
                    and i_link.right_feature_group == j_link.right_feature_group
                    and i_link.jointype != j_link.jointype
                ):
                    raise ValueError(
                        f"Link {i_link} and {j_link} have different join types for the same feature groups. Please remove one."
                    )

    @staticmethod
    def validate_right_join_constraints(links: Set["Link"]) -> None:
        from mloda.core.abstract_plugins.components.link import JoinType

        for i_link in links:
            if i_link.jointype == JoinType.RIGHT:
                for j_link in links:
                    if i_link == j_link:
                        continue
                    if (
                        i_link.left_feature_group == j_link.left_feature_group
                        or i_link.left_feature_group == j_link.right_feature_group
                    ):
                        raise ValueError(
                            f"Link {i_link} and {j_link} have multiple right joins for the same feature group on the left side or switching from left to right side although using right join. Please reconsider your joinlogic and if possible, use left joins instead of rightjoins. This will currently break the planner or during execution."
                        )

    @classmethod
    def validate_links(cls, links: Optional[Set["Link"]]) -> None:
        if links is None:
            return

        for link in links:
            cls.validate_join_type(link.jointype)

        cls.validate_no_double_joins(links)
        cls.validate_no_conflicting_join_types(links)
        cls.validate_right_join_constraints(links)
