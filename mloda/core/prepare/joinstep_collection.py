from collections import defaultdict
from uuid import UUID
from mloda.core.core.step.join_step import JoinStep


class JoinStepCollection:
    def __init__(self) -> None:
        self.collection: dict[JoinStep, set[UUID]] = defaultdict(set)

    def add(self, join_step: JoinStep) -> None:
        self.collection[join_step] = set()

    def earlier_joins_sharing_destination(self, join_step: JoinStep) -> set[UUID]:
        uuids: set[UUID] = set()
        for step in self.collection:
            if step == join_step:
                break
            if step.destination_framework_uuids & join_step.destination_framework_uuids:
                uuids.update(step.get_uuids())
        return uuids

    def get_required_join_uuids(self, join_step: JoinStep) -> set[UUID]:
        return self.earlier_joins_sharing_destination(join_step)
