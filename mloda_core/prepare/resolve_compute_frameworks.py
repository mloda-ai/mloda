from copy import deepcopy
from typing import Any, Dict, List, Set, Type
from collections import defaultdict
from uuid import UUID
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.prepare.graph.graph import Graph
from mloda_core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda_core.abstract_plugins.components.link import JoinType, Link


class ResolveComputeFrameworks:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.to_invert_trekker_collection: List[LinkFrameworkTrekker] = []

    def links(self, planned_queue: Any, link_trekker: LinkTrekker) -> Any:
        new_planned_queue = []
        for p in planned_queue:
            if isinstance(p, tuple):
                if not isinstance(p[0], Link):
                    feature = next(iter(p[1]))
                    trekked_links = self.access_link_by_child_uuid(feature.uuid, link_trekker)
                    if trekked_links:
                        new_compute_frameworks = self.resolve_trekked_links(trekked_links, feature.compute_frameworks)
                        for f in p[1]:
                            f.compute_frameworks = new_compute_frameworks

                        self.trekker_right_left_adjuster(link_trekker, {_f.uuid for _f in p[1]})

            new_planned_queue.append(p)

        link_trekker.order_links_by_frameworks()

        new_planned_queue = self.order_queue_by_trekker_order(new_planned_queue, link_trekker)
        return new_planned_queue

    def order_queue_by_trekker_order(self, planned_queue: Any, link_trekker: LinkTrekker) -> Any:
        orders = link_trekker.order

        new_planned_queue = []
        link_already_added: Set[UUID] = set()

        issue_collector: Dict[UUID, Set[tuple[Any]]] = defaultdict(set)

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
    def access_link_by_child_uuid(cls, child_uuid: UUID, link_trekker: LinkTrekker) -> List[LinkFrameworkTrekker]:
        link_framework_trekker = []
        for trekker, uuids in link_trekker.data_ordered.items():
            if child_uuid in uuids:
                link_framework_trekker.append(trekker)
        return link_framework_trekker

    def trekker_right_left_adjuster(self, link_trekker: LinkTrekker, feature_uuids: Set[UUID]) -> None:
        if not self.to_invert_trekker_collection:
            return

        for link, left_cfw, right_cfw in self.to_invert_trekker_collection:
            for trekker, uuids in deepcopy(link_trekker.data_ordered).items():
                if trekker == (link, left_cfw, right_cfw):
                    for uuid in deepcopy(uuids):
                        if uuid in feature_uuids:
                            link_trekker.invert_link(link, left_cfw, right_cfw, uuid)

        self.to_invert_trekker_collection = []

    def resolve_trekked_links(
        self, trekked_links: List[LinkFrameworkTrekker], compute_frameworks: Set[Type[ComputeFrameWork]]
    ) -> Set[Type[ComputeFrameWork]]:
        new_cfws = set()

        for link, left_cfw, right_cfw in trekked_links:
            if link.jointype == JoinType.RIGHT:
                if right_cfw in compute_frameworks:
                    new_cfws.add(right_cfw)
                elif left_cfw in compute_frameworks:
                    new_cfws.add(right_cfw)
                    self.to_invert_trekker_collection.append((link, left_cfw, right_cfw))
                continue

            if link.jointype in JoinType:
                if left_cfw in compute_frameworks and right_cfw in compute_frameworks:
                    # We keep the left framework if possible. This can be dropped maybe later with more tests.
                    new_cfws.add(left_cfw)
                elif left_cfw in compute_frameworks:
                    new_cfws.add(left_cfw)
                elif right_cfw in compute_frameworks:
                    new_cfws.add(right_cfw)
                    self.to_invert_trekker_collection.append((link, left_cfw, right_cfw))
                continue

            raise ValueError(
                f"This jointype is not implemented: {link.jointype}. Possible types are: {[member.value for member in JoinType]}"
            )

        if not new_cfws:
            raise ValueError("No new compute frameworks have been found.")
        return new_cfws
