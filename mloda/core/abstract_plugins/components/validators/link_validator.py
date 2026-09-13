from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.link import Link


class LinkValidator:
    @staticmethod
    def validate_index_not_empty(index: str | tuple[str, ...], context: str = "index") -> None:
        if not index:
            raise ValueError(f"{context} cannot be empty")

    @staticmethod
    def _same_node(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
        """An undiscriminated side is a wildcard that may be any node of its class."""
        return a is None or b is None or a == b

    @staticmethod
    def validate_join_type(jointype: Any) -> None:
        from mloda.core.abstract_plugins.components.link import JoinType

        if not isinstance(jointype, JoinType):
            raise ValueError(f"Join type {jointype} is not supported")

    @staticmethod
    def validate_no_double_joins(links: set["Link"]) -> None:
        from mloda.core.abstract_plugins.components.link import JoinType

        for i_link in links:
            for j_link in links:
                if i_link == j_link:
                    continue
                if (
                    i_link.left_feature_group == j_link.right_feature_group
                    and i_link.right_feature_group == j_link.left_feature_group
                    and i_link.jointype not in [JoinType.APPEND, JoinType.UNION]
                    and LinkValidator._same_node(i_link.left_discriminator, j_link.right_discriminator)
                    and LinkValidator._same_node(i_link.right_discriminator, j_link.left_discriminator)
                ):
                    raise ValueError(
                        f"Link {i_link} and {j_link} have at least two different defined joins. Please remove one."
                    )

    @staticmethod
    def validate_no_conflicting_join_types(links: set["Link"]) -> None:
        for i_link in links:
            for j_link in links:
                if i_link == j_link:
                    continue
                if (
                    i_link.left_feature_group == j_link.left_feature_group
                    and i_link.right_feature_group == j_link.right_feature_group
                    and i_link.jointype != j_link.jointype
                    and LinkValidator._same_node(i_link.left_discriminator, j_link.left_discriminator)
                    and LinkValidator._same_node(i_link.right_discriminator, j_link.right_discriminator)
                ):
                    raise ValueError(
                        f"Link {i_link} and {j_link} have different join types for the same feature groups. Please remove one."
                    )

    @staticmethod
    def validate_right_join_constraints(links: set["Link"]) -> None:
        from mloda.core.abstract_plugins.components.link import JoinType

        for i_link in links:
            if i_link.jointype == JoinType.RIGHT:
                for j_link in links:
                    if i_link == j_link:
                        continue
                    if (
                        i_link.left_feature_group == j_link.left_feature_group
                        and LinkValidator._same_node(i_link.left_discriminator, j_link.left_discriminator)
                    ) or (
                        i_link.left_feature_group == j_link.right_feature_group
                        and LinkValidator._same_node(i_link.left_discriminator, j_link.right_discriminator)
                    ):
                        raise ValueError(
                            f"Link {i_link} and {j_link} have multiple right joins for the same feature group on the left side or switching from left to right side although using right join. Please reconsider your joinlogic and if possible, use left joins instead of rightjoins. This will currently break the planner or during execution."
                        )

    @staticmethod
    def validate_same_class_discriminator_pairs(links: set["Link"]) -> None:
        for link in links:
            if link.left_feature_group is link.right_feature_group:
                if (link.left_discriminator is None) != (link.right_discriminator is None):
                    raise ValueError(
                        f"Link {link} joins the same class on both sides and needs both left_discriminator "
                        "and right_discriminator or neither."
                    )

    @classmethod
    def validate_links(cls, links: set["Link"] | None) -> None:
        if links is None:
            return

        for link in links:
            cls.validate_join_type(link.jointype)

        cls.validate_same_class_discriminator_pairs(links)
        cls.validate_no_double_joins(links)
        cls.validate_no_conflicting_join_types(links)
        cls.validate_right_join_constraints(links)
