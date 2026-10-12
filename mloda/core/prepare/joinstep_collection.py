from collections import defaultdict
from collections.abc import Mapping
from uuid import UUID
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.step.join_step import JoinStep


class JoinStepCollection:
    def __init__(self) -> None:
        self.collection: dict[JoinStep, set[UUID]] = defaultdict(set)
        self._open: defaultdict[type[ComputeFramework], set[UUID]] = defaultdict(set)

    def similar_dependent_joins_uuids(
        self, destination_framework: type[ComputeFramework], source_framework: type[ComputeFramework]
    ) -> set[UUID]:
        """The returned tokens plus what those joins wait on cover every earlier join sharing a framework."""
        return self._open[destination_framework] | self._open[source_framework]

    def add(self, join_step: JoinStep) -> None:
        destination, source = join_step.destination_framework, join_step.source_framework
        # Only a cross-framework join reads its tokens and replaces the open set; a same-framework
        # join stores none and only adds itself.
        if destination != source:
            self.collection[join_step] = self.similar_dependent_joins_uuids(destination, source)
            self._open[destination] = {join_step.uuid}
            self._open[source] = {join_step.uuid}
        else:
            self.collection[join_step] = set()
            self._open[destination].add(join_step.uuid)

    def joins_by_destination_owner(
        self, owning_step_of: Mapping[UUID, UUID] | None = None
    ) -> dict[UUID, list[JoinStep]]:
        """Joins per destination owner, in registration order."""
        owner = owning_step_of or {}
        index: dict[UUID, list[JoinStep]] = defaultdict(list)
        for step in self.collection:
            for key in {owner.get(u, u) for u in step.destination_framework_uuids}:
                index[key].append(step)
        return dict(index)

    def earlier_joins_sharing_destination(
        self,
        join_step: JoinStep,
        owning_step_of: Mapping[UUID, UUID] | None = None,
        index: Mapping[UUID, list[JoinStep]] | None = None,
    ) -> set[UUID]:
        """Uuids of earlier joins of any link whose destination shares a feature uuid, or owning step if given."""
        owner = owning_step_of or {}
        if index is None:
            index = self.joins_by_destination_owner(owning_step_of)
        uuids: set[UUID] = set()
        for key in {owner.get(u, u) for u in join_step.destination_framework_uuids}:
            for step in index.get(key, ()):
                if step == join_step:
                    break
                uuids.update(step.get_uuids())
        return uuids

    def get_required_join_uuids(self, join_step: JoinStep) -> set[UUID]:
        return self.collection.get(join_step, set())
