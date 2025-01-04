from collections import defaultdict
from typing import Dict, Set, Type
from uuid import UUID
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.core.step.join_step import JoinStep


class JoinStepCollection:
    def __init__(self) -> None:
        self.collection: Dict[JoinStep, Set[UUID]] = defaultdict(set)

    def similar_dependent_joins_uuids(
        self, left_framework: Type[ComputeFrameWork], right_framework: Type[ComputeFrameWork]
    ) -> Set[UUID]:
        """
        This functionality makes sure that we do not write on the same datasets due to overlapping joins at once.
        This can be optimized, but I just added a hard solution.
        """
        required_uuids = set()
        for step in self.collection:
            if (
                step.left_framework == left_framework
                or step.right_framework == left_framework
                or step.left_framework == right_framework
                or step.right_framework == right_framework
            ):
                required_uuids.update(step.get_uuids())

        return required_uuids

    def add(self, join_step: JoinStep) -> None:
        required_join_uuids = self.similar_dependent_joins_uuids(join_step.left_framework, join_step.right_framework)
        self.collection[join_step] = required_join_uuids

    def get_required_join_uuids(self, join_step: JoinStep) -> Set[UUID]:
        return self.collection[join_step]
