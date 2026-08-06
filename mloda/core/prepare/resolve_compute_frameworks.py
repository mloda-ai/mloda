from copy import deepcopy
from typing import Any
from collections import Counter, defaultdict
from uuid import UUID
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda.core.abstract_plugins.components.link import JoinType, Link


class ResolveComputeFrameworks:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.to_invert_trekker_collection: list[LinkFrameworkTrekker] = []

    def links(self, planned_queue: Any, link_trekker: LinkTrekker) -> Any:
        groups = [p for p in planned_queue if isinstance(p, tuple) and not isinstance(p[0], Link)]

        trekker_members: dict[LinkFrameworkTrekker, list[Any]] = defaultdict(list)
        feature_trekkers: dict[UUID, list[LinkFrameworkTrekker]] = defaultdict(list)
        for p in groups:
            for f in p[1]:
                for trekker in self.access_link_by_child_uuid(f.uuid, link_trekker):
                    trekker_members[trekker].append(f)
                    feature_trekkers[f.uuid].append(trekker)

        # Snapshot before resolving: invert_link mutates data_ordered while the loop runs.
        trekked_uuids = {trekker: set(uuids) for trekker, uuids in link_trekker.data_ordered.items()}

        resolved: dict[LinkFrameworkTrekker, type[ComputeFramework]] = {}
        for trekker, members in trekker_members.items():
            if trekker not in trekked_uuids:
                raise ValueError(f"Trekker bookkeeping is inconsistent: no uuids recorded for link {trekker[0]}.")
            resolved_cfw = self.resolve_trekker(trekker, members)
            if resolved_cfw is not None:
                resolved[trekker] = resolved_cfw
            # Invert every uuid of the trekker at once, so a link keeps a single orientation across all groups.
            self.trekker_right_left_adjuster(link_trekker, trekked_uuids[trekker])

        for p in groups:
            self.rewrite_group_frameworks(p, resolved, feature_trekkers)

        self.reject_self_merging_links(groups, link_trekker)

        new_planned_queue = list(planned_queue)

        link_trekker.order_links_by_frameworks()

        new_planned_queue = self.order_queue_by_trekker_order(new_planned_queue, link_trekker)
        return new_planned_queue

    def reject_self_merging_links(self, groups: Any, link_trekker: LinkTrekker) -> None:
        """Reject a scheduled link that joins in one framework none of its children run in.

        Both join sides would then filter to the same uuids and the merge degenerates into a
        self merge, dropping one parent's columns.
        """
        features_by_uuid = {f.uuid: f for p in groups for f in p[1]}

        for (link, left_cfw, right_cfw), child_uuids in link_trekker.data.items():
            if left_cfw != right_cfw:
                continue

            # A self join takes case_link_equal_feature_groups, which suppresses the join step.
            if link.left_feature_group == link.right_feature_group:
                continue

            # APPEND and UNION pick one uuid per side by index and feature group, never by framework.
            if link.jointype in (JoinType.APPEND, JoinType.UNION):
                continue

            children = [features_by_uuid[uuid] for uuid in child_uuids if uuid in features_by_uuid]
            if not children:
                continue

            if any(child.compute_frameworks is None or left_cfw in child.compute_frameworks for child in children):
                continue

            cfw_name = left_cfw.__name__
            left_name = link.left_feature_group.__name__
            right_name = link.right_feature_group.__name__
            names = sorted(str(child.name) for child in children)
            raise ValueError(
                f"Link {link.jointype.value} {left_name} {right_name} joins in {cfw_name}, "
                f"which none of its children run in: {names}. "
                "Both join sides would resolve to the same input. "
                "Give the joined feature groups distinct compute frameworks, "
                f"or bring the children of this link onto {cfw_name}."
            )

    def rewrite_group_frameworks(
        self,
        group: Any,
        resolved: dict[LinkFrameworkTrekker, type[ComputeFramework]],
        feature_trekkers: dict[UUID, list[LinkFrameworkTrekker]],
    ) -> None:
        group_cfws = {
            resolved[trekker] for f in group[1] for trekker in feature_trekkers.get(f.uuid, []) if trekker in resolved
        }

        any_rewritten = False
        for f in group[1]:
            if f.uuid not in feature_trekkers:
                if group_cfws:
                    supported = set(f.compute_frameworks) if f.compute_frameworks is not None else group_cfws
                    unsupported = group_cfws - supported
                    if unsupported:
                        names = sorted(cfw.__name__ for cfw in unsupported)
                        raise ValueError(
                            f"Feature {f.name} does not support the compute framework(s) {names} chosen for its group."
                        )
                    f.compute_frameworks = set(group_cfws)
                    any_rewritten = True
                continue
            new_cfws = {resolved[trekker] for trekker in feature_trekkers[f.uuid] if trekker in resolved}
            if not new_cfws:
                mismatches = sorted(
                    f"{link}: neither {left.__name__} nor {right.__name__}"
                    for link, left, right in feature_trekkers[f.uuid]
                )
                raise ValueError(
                    f"No compute framework agreement for feature {f.name}. Unresolvable links: {mismatches}"
                )
            f.compute_frameworks = new_cfws
            any_rewritten = True

        if not any_rewritten:
            return

        # Rehash via list so hashes are recomputed (set(group[1]) reuses stale ones), keeping set and aliases valid.
        members = list(group[1])
        group[1].clear()
        group[1].update(members)
        if len(group[1]) != len(members):
            names = sorted(name for name, count in Counter(str(member.name) for member in members).items() if count > 1)
            raise ValueError(
                "Compute framework rewrite collapsed features that were previously distinct "
                f"only by compute_frameworks. Affected: {names}"
            )

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
        link, left_cfw, right_cfw = trekker

        left_in_all = all(left_cfw in m.compute_frameworks for m in members)
        right_in_all = all(right_cfw in m.compute_frameworks for m in members)

        if link.jointype == JoinType.RIGHT:
            if right_in_all:
                return right_cfw
            if left_in_all:
                self.to_invert_trekker_collection.append(trekker)
                return right_cfw
            return None

        if link.jointype in JoinType:
            if left_in_all:
                return left_cfw
            if right_in_all:
                self.to_invert_trekker_collection.append(trekker)
                return right_cfw
            return None

        raise ValueError(
            f"This jointype is not implemented: {link.jointype}. Possible types are: {[member.value for member in JoinType]}"
        )
