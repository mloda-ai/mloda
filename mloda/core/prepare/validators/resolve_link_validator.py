from collections import OrderedDict
from typing import Any, Dict, Set
from uuid import UUID


class ResolveLinkValidator:
    @staticmethod
    def validate_data_consistency(
        data: Dict[Any, Set[UUID]],
        data_ordered: "OrderedDict[Any, Set[UUID]]",
    ) -> None:
        if len(data.items()) != len(data_ordered.items()):
            raise ValueError("Data and data_ordered have different lengths")

    @staticmethod
    def validate_no_conflicting_join_types(data: Dict[Any, Set[UUID]]) -> None:
        seen_pairs: Dict[Any, Any] = {}
        for key in data.keys():
            link, _, _ = key
            left_fg = link.left_feature_group
            right_fg = link.right_feature_group
            jointype = link.jointype

            pair_key = (left_fg, right_fg)

            if pair_key in seen_pairs:
                if seen_pairs[pair_key] != jointype:
                    raise Exception(
                        f"Conflicting join types for {left_fg.get_class_name()} and {right_fg.get_class_name()}"
                    )
            else:
                seen_pairs[pair_key] = jointype
