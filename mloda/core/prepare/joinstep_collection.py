from collections import defaultdict
from uuid import UUID
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.step.join_step import JoinStep


class JoinStepCollection:
    def __init__(self) -> None:
        self.collection: dict[JoinStep, set[UUID]] = defaultdict(set)

    def similar_dependent_joins_uuids(
        self, destination_framework: type[ComputeFramework], source_framework: type[ComputeFramework]
    ) -> set[UUID]:
        """
        This functionality makes sure that we do not write on the same datasets due to overlapping joins at once.
        This can be optimized, but I just added a hard solution.
        """
        required_uuids = set()
        for step in self.collection:
            if (
                step.destination_framework == destination_framework
                or step.source_framework == destination_framework
                or step.destination_framework == source_framework
                or step.source_framework == source_framework
            ):
                required_uuids.update(step.get_uuids())

        return required_uuids

    def add(self, join_step: JoinStep) -> None:
        required_join_uuids = self.similar_dependent_joins_uuids(
            join_step.destination_framework, join_step.source_framework
        )
        self.collection[join_step] = required_join_uuids

    def get_required_join_uuids(self, join_step: JoinStep) -> set[UUID]:
        return self.collection[join_step]
