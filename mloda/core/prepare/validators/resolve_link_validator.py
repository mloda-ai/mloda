from collections import OrderedDict
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.components.validators.link_validator import LinkValidator


class ResolveLinkValidator:
    @staticmethod
    def validate_data_consistency(
        data: dict[Any, set[UUID]],
        data_ordered: "OrderedDict[Any, set[UUID]]",
    ) -> None:
        if len(data.items()) != len(data_ordered.items()):
            raise ValueError("Data and data_ordered have different lengths")

    @staticmethod
    def validate_no_conflicting_join_types(data: dict[Any, set[UUID]]) -> None:
        links = {key[0] for key in data.keys()}

        for i_link in links:
            for j_link in links:
                if i_link == j_link:
                    continue
                if (
                    i_link.left_feature_group == j_link.left_feature_group
                    and i_link.right_feature_group == j_link.right_feature_group
                    and LinkValidator._same_node(i_link.left_discriminator, j_link.left_discriminator)
                    and LinkValidator._same_node(i_link.right_discriminator, j_link.right_discriminator)
                    and i_link.jointype != j_link.jointype
                ):
                    raise Exception(
                        f"Conflicting join types for {i_link.left_feature_group.get_class_name()} "
                        f"and {i_link.right_feature_group.get_class_name()}"
                    )
