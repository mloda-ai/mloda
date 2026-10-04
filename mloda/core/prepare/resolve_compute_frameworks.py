from copy import deepcopy
from typing import Any
from collections import defaultdict
from uuid import UUID
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda.core.abstract_plugins.components.link import JoinType, Link


class ResolveComputeFrameworks:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.to_invert_trekker_collection: list[LinkFrameworkTrekker] = []
        self.declared_frameworks: dict[UUID, frozenset[type[ComputeFramework]]] = {}

    def get_declared_frameworks(self) -> dict[UUID, frozenset[type[ComputeFramework]]]:
        return self.declared_frameworks

    def links(self, planned_queue: Any, link_trekker: LinkTrekker) -> Any:
        groups = [p for p in planned_queue if isinstance(p, tuple) and not isinstance(p[0], Link)]

        for p in groups:
            for f in p[1]:
                self.declared_frameworks[f.uuid] = frozenset(f.compute_frameworks or ())

        trekker_members: dict[LinkFrameworkTrekker, list[Any]] = defaultdict(list)
        for p in groups:
            for f in p[1]:
                for trekker in self.access_link_by_child_uuid(f.uuid, link_trekker):
                    trekker_members[trekker].append(f)

        # Snapshot before resolving: invert_link mutates data_ordered while the loop runs.
        trekked_uuids = {trekker: set(uuids) for trekker, uuids in link_trekker.data_ordered.items()}

        for trekker, members in trekker_members.items():
            if trekker not in trekked_uuids:
                raise ValueError(f"Trekker bookkeeping is inconsistent: no uuids recorded for link {trekker[0]}.")
            resolved_cfw = self.resolve_trekker(trekker, members)
            for member in members:
                if resolved_cfw is not None and member.get_compute_framework() is not resolved_cfw:
                    raise ValueError(
                        f"Feature {member.name} runs on {member.get_compute_framework().get_class_name()}, "
                        f"but its link {trekker[0]} resolves to {resolved_cfw.get_class_name()}."
                    )
            # Invert every uuid of the trekker at once, so a link keeps a single orientation across all groups.
            self.trekker_right_left_adjuster(link_trekker, trekked_uuids[trekker])

        new_planned_queue = list(planned_queue)

        link_trekker.order_links_by_frameworks()

        new_planned_queue = self.order_queue_by_trekker_order(new_planned_queue, link_trekker)
        return new_planned_queue

    def order_queue_by_trekker_order(self, planned_queue: Any, link_trekker: LinkTrekker) -> Any:
        orders = link_trekker.order

        new_planned_queue = []
        link_already_added: set[UUID] = set()

        issue_collector: dict[UUID, set[tuple[Any]]] = defaultdict(set)

        for pos, p in enumerate(planned_queue):
            breaker = False

            if isinstance(p, tuple):
                if isinstance(p[0], Link):
                    # search for those, which are too early
                    uuid = p[0].uuid
                    for k, v in orders.items():
                        if uuid in v:
                            if k not in link_already_added:
                                issue_collector[k].add(p)
                                breaker = True
                                break
                    if breaker:
                        continue
                    link_already_added.add(uuid)
            new_planned_queue.append(p)

            # look for those, which were too early and check if they can be handeled after adding this link
            if isinstance(p, tuple):
                if isinstance(p[0], Link):
                    # loop over issues
                    for k, dependent_links in issue_collector.items():
                        if p[0].uuid == k:
                            # loop over dependent links of issues
                            for dep_link in dependent_links:
                                breaker = False
                                dep_uuid = dep_link[0].uuid

                                # loop over all orders and check if all dependencies are already added
                                for k, v in orders.items():
                                    if dep_uuid in v:
                                        # if not break
                                        if k not in link_already_added:
                                            breaker = True
                                            break

                                # if all dependencies are there, add the link
                                if not breaker:
                                    new_planned_queue.append(dep_link)
                                    link_already_added.add(dep_uuid)

        return new_planned_queue

    @classmethod
    def access_link_by_child_uuid(cls, child_uuid: UUID, link_trekker: LinkTrekker) -> list[LinkFrameworkTrekker]:
        link_framework_trekker = []
        for trekker, uuids in link_trekker.data_ordered.items():
            if child_uuid in uuids:
                link_framework_trekker.append(trekker)
        return link_framework_trekker

    def trekker_right_left_adjuster(self, link_trekker: LinkTrekker, feature_uuids: set[UUID]) -> None:
        if not self.to_invert_trekker_collection:
            return

        for link, left_cfw, right_cfw in self.to_invert_trekker_collection:
            for trekker, uuids in deepcopy(link_trekker.data_ordered).items():
                if trekker == (link, left_cfw, right_cfw):
                    for uuid in deepcopy(uuids):
                        if uuid in feature_uuids:
                            link_trekker.invert_link(link, left_cfw, right_cfw, uuid)

        self.to_invert_trekker_collection = []

    def resolve_trekker(self, trekker: LinkFrameworkTrekker, members: list[Any]) -> type[ComputeFramework] | None:
        """The framework the link joins in, or None when its members run on neither side (a chained hop)."""
        link, left_cfw, right_cfw = trekker
        running = {m.get_compute_framework() for m in members}

        if left_cfw is not right_cfw and {left_cfw, right_cfw} <= running:
            names = sorted(str(m.name) for m in members)
            raise ValueError(
                f"No compute framework agreement for {names}: they run on both sides of link {link}, "
                f"{left_cfw.get_class_name()} and {right_cfw.get_class_name()}."
            )

        if link.jointype == JoinType.RIGHT:
            return right_cfw if running == {right_cfw} else None

        if link.jointype in (JoinType.APPEND, JoinType.UNION):
            return left_cfw if running == {left_cfw} else None

        if link.jointype in JoinType:
            if running == {left_cfw}:
                return left_cfw
            if running == {right_cfw}:
                self.to_invert_trekker_collection.append(trekker)
                return right_cfw
            return None

        raise ValueError(
            f"This jointype is not implemented: {link.jointype}. Possible types are: {[member.value for member in JoinType]}"
        )
