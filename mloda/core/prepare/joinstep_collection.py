from collections import defaultdict
from collections.abc import Mapping
from uuid import UUID
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.step.join_step import JoinStep


class JoinStepCollection:
    def __init__(self) -> None:
        self.collection: dict[JoinStep, set[UUID]] = defaultdict(set)
        self._uuids_by_framework: defaultdict[type[ComputeFramework], set[UUID]] = defaultdict(set)

    def similar_dependent_joins_uuids(
        self, destination_framework: type[ComputeFramework], source_framework: type[ComputeFramework]
    ) -> set[UUID]:
        """
        This functionality makes sure that we do not write on the same datasets due to overlapping joins at once.
        This can be optimized, but I just added a hard solution.
        """
        return self._uuids_by_framework[destination_framework] | self._uuids_by_framework[source_framework]

    def add(self, join_step: JoinStep) -> None:
        required_join_uuids = self.similar_dependent_joins_uuids(
            join_step.destination_framework, join_step.source_framework
        )
        self.collection[join_step] = required_join_uuids
        self._uuids_by_framework[join_step.destination_framework].update(join_step.get_uuids())
        self._uuids_by_framework[join_step.source_framework].update(join_step.get_uuids())

    def earlier_joins_sharing_destination(
        self, join_step: JoinStep, owning_step_of: Mapping[UUID, UUID] | None = None
    ) -> set[UUID]:
        """Uuids of earlier joins of any link whose destination shares a feature uuid, or owning step if given."""
        owner = owning_step_of or {}
        wanted = {owner.get(u, u) for u in join_step.destination_framework_uuids}
        uuids: set[UUID] = set()
        for step in self.collection:
            if step == join_step:
                break
            if wanted & {owner.get(u, u) for u in step.destination_framework_uuids}:
                uuids.update(step.get_uuids())
        return uuids

    def get_required_join_uuids(self, join_step: JoinStep) -> set[UUID]:
        return self.collection.get(join_step, set())
