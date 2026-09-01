from collections.abc import Sequence
from copy import copy, deepcopy
from typing import Any, Generator, NamedTuple, Optional
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.error_utils import REPORT_URL, internal_invariant_error
from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.core.abstract_plugins.components.index.index import Index

from mloda.core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda.core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda.core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.filter.single_filter import SingleFilter
from mloda.core.prepare.declared_sides import split_by_declared_side
from mloda.core.prepare.joinstep_collection import JoinStepCollection
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_graph import PlannedQueue
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda.core.prepare.resolved_join import (
    DeclinedOrientation,
    JoinSide,
    JoinSignature,
    ResolvedJoin,
    ResolvedJoinPlan,
)
from mloda.core.prepare.resolved_join_builder import (
    DeclaredFrameworks,
    build_resolved_join_side,
    joinstep_signatures,
    raise_on_join_plan_divergence,
    wire_join_dependencies,
)
from mloda.core.prepare.validate_resolved_join import raise_on_orphaned_join_source
from mloda.core.core.step.abstract_step import Step
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.abstract_plugins.feature_group import FeatureGroup, format_feature_group_class
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.link import JoinType, Link
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


def _filter_options_sort_key(single_filter: SingleFilter) -> tuple[str, str]:
    """Order enrichment variants: stable for values with a value-based repr, repr-identity ordering
    otherwise; a raising repr degrades to the value's type name."""
    options = single_filter.filter_feature.options
    return (
        repr(sorted((str(k), safe_field(lambda: repr(v), type(v).__name__)) for k, v in options.group.items())),
        repr(sorted((str(k), safe_field(lambda: repr(v), type(v).__name__)) for k, v in options.context.items())),
    )


def _describe_step(step: Step) -> str:
    """Name a step of the plan for an error message."""
    if isinstance(step, JoinStep):
        return f"JoinStep(link={step.link})"
    if isinstance(step, FeatureGroupStep):
        return f"FeatureGroupStep({format_feature_group_class(step.feature_group)}, uuid={step.uuid})"
    if isinstance(step, TransformFrameworkStep):
        return (
            f"TransformFrameworkStep({step.from_framework.get_class_name()} -> "
            f"{step.to_framework.get_class_name()}, uuid={step.uuid})"
        )
    return f"{type(step).__name__}(uuid={step.uuid})"


class AppendOrUnionSides(NamedTuple):
    """The left/right feature uuids and frameworks an APPEND or UNION link resolved to."""

    destination_framework: type[ComputeFramework]
    source_framework: type[ComputeFramework]
    left_uuid: UUID
    right_uuid: UUID


class _JoinServedParent(NamedTuple):
    """Stand-in for a parent delivered by a JoinStep (no TransformFrameworkStep is built for it), so it
    can still be named in the missing-Links conflict error."""

    from_feature_group: type[FeatureGroup]


class ExecutionPlan:
    def __init__(
        self,
        global_filter: Optional[GlobalFilter] = None,
        api_input_data_collection: Optional[ApiInputDataCollection] = None,
        resolved_input_feature_names: dict[UUID, frozenset[str] | None] | None = None,
    ) -> None:
        # Maps a step to itself so a dedup hit can recover the already-inserted canonical member.
        self.tfs_collection: dict[TransformFrameworkStep, TransformFrameworkStep] = {}
        self.joinstep_collection = JoinStepCollection()
        self.global_filter = global_filter
        self.api_input_data_collection = api_input_data_collection
        self.resolved_input_feature_names = resolved_input_feature_names

        # Helper variable
        self.feature_set_collections: list[set[UUID]] = []

        # Report each divergence once, then at DEBUG.
        self.reported_unmatched: set[tuple[type[FeatureGroup], str, tuple[str, ...]]] = set()

        self.planned_records: list[ResolvedJoin] = []
        self.declined_orientations: list[LinkFrameworkTrekker] = []
        self.declared_frameworks: DeclaredFrameworks = {}
        self.resolved_join_plan = ResolvedJoinPlan((), ())
        self.join_signatures_at_build: frozenset[JoinSignature] = frozenset()

    def __iter__(self) -> Generator[TransformFrameworkStep | JoinStep | FeatureGroupStep, None, None]:
        yield from self.execution_plan

    def __len__(self) -> int:
        return len(self.execution_plan)

    def create_execution_plan(
        self,
        queue: PlannedQueue,
        graph: Graph,
        link_trekker: LinkTrekker,
        declared_frameworks: DeclaredFrameworks | None = None,
        validate: bool = True,
    ) -> None:
        self.planned_records = []
        self.declined_orientations = []
        self.tfs_collection = {}
        self.joinstep_collection = JoinStepCollection()
        self.feature_set_collections = []
        self.declared_frameworks = declared_frameworks if declared_frameworks is not None else {}

        child_links = self.invert_link_trekker(link_trekker)
        pre_execution_plan = self.add_feature_group_step(queue, graph.parent_to_children_mapping, child_links)
        fw_execution_plan = self.add_joinstep(pre_execution_plan, link_trekker, graph)

        # Built before add_tfs, whose write serialization edges are not part of the join decision.
        join_steps = [step for step in fw_execution_plan if isinstance(step, JoinStep)]
        resolved_records = wire_join_dependencies(self.planned_records, join_steps)
        declined = tuple(DeclinedOrientation(key[0].uuid, key[1], key[2]) for key in self.declined_orientations)
        self.resolved_join_plan = ResolvedJoinPlan(resolved_records, declined)
        self.join_signatures_at_build = joinstep_signatures(join_steps)
        raise_on_join_plan_divergence(self.resolved_join_plan, join_steps)
        if validate:
            raise_on_orphaned_join_source(self.resolved_join_plan)

        self.execution_plan = self.add_tfs(fw_execution_plan, graph)
        self.raise_on_step_cycle(self.execution_plan)

        # Only read during add_joinstep above; ExecutionPlan gets deepcopy'd on every Engine.compute()
        # call, so this (potentially O(#features), UUID-keyed) dict and its live reference into
        # ResolveComputeFrameworks's own dict must not linger past the plan build that needs it, and
        # neither must the engine's resolved_input_feature_names map that run_feature_group read.
        self.declared_frameworks = {}
        self.resolved_input_feature_names = None

    def add_feature_group_step(
        self,
        queue: PlannedQueue,
        parent_to_children_mapping: dict[UUID, set[UUID]],
        child_links: dict[UUID, set[LinkFrameworkTrekker]],
    ) -> list[LinkFrameworkTrekker | FeatureGroupStep]:
        pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep] = []

        for element in queue:
            if isinstance(element[0], Link):
                pre_execution_plan.append(element)
                continue

            elif issubclass(element[0], FeatureGroup):
                if not isinstance(element[1], set):
                    raise ValueError(f"Element {element} is not a valid element.")

                links_pre_calulated = self.retrieve_links_which_must_be_calculated_before(element[1], child_links)
                feature_group_steps = self.run_feature_group(element, parent_to_children_mapping, links_pre_calulated)
                for fg_step in feature_group_steps.values():
                    pre_execution_plan.append(fg_step)

            else:
                raise ValueError(f"Element {element} is not a valid element.")
        return pre_execution_plan

    def add_joinstep(
        self,
        pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep],
        link_trekker: LinkTrekker,
        graph: Graph,
    ) -> list[JoinStep | FeatureGroupStep]:
        fw_execution_plan: list[JoinStep | FeatureGroupStep] = []

        for pex in pre_execution_plan:
            if isinstance(pex, tuple):
                js = self.run_link(pex, link_trekker, graph, pre_execution_plan)
                if js is not None:
                    fw_execution_plan.append(js)
            else:
                fw_execution_plan.append(pex)

        fw_execution_plan = self.handle_append_or_union_joinstep(fw_execution_plan)

        self.expand_link_tokens(fw_execution_plan, link_trekker)

        return fw_execution_plan

    def expand_link_tokens(
        self, fw_execution_plan: list[JoinStep | FeatureGroupStep], link_trekker: LinkTrekker
    ) -> None:
        """Replace every waited-on link uuid with the uuids of the JoinSteps planned for that link."""
        links_by_uuid: dict[UUID, Link] = {trekker[0].uuid: trekker[0] for trekker in link_trekker.data}

        joinstep_uuids: dict[UUID, set[UUID]] = defaultdict(set)
        for step in fw_execution_plan:
            if isinstance(step, JoinStep):
                joinstep_uuids[step.link.uuid].add(step.uuid)

        # The planned steps are a source of link uuids of their own, and handle_append_or_union_joinstep waits on them.
        link_uuids = set(links_by_uuid) | set(link_trekker.order) | set(joinstep_uuids)

        # Collected first: a raise on a later step must not leave a half expanded plan behind.
        expansions: list[tuple[JoinStep | FeatureGroupStep, set[UUID], set[UUID]]] = []
        for step in fw_execution_plan:
            required_links = step.required_uuids & link_uuids
            if not required_links:
                continue

            expanded: set[UUID] = set()
            for link_uuid in required_links:
                produced = joinstep_uuids.get(link_uuid)
                if not produced:
                    raise ValueError(self._no_joinstep_for_link_error(links_by_uuid.get(link_uuid, link_uuid)))
                expanded.update(produced)

            # A step must never wait for a token it produces itself.
            expansions.append((step, required_links, expanded - step.get_uuids()))

        for step, required_links, expanded in expansions:
            step.required_uuids.difference_update(required_links)
            step.required_uuids.update(expanded)

    @staticmethod
    def _no_joinstep_for_link_error(link: Link | UUID) -> str:
        """A link a step waits for that planned no join step is a configuration problem, not a bug."""
        return (
            f"No join step was planned for a link that a step of the plan waits for: {link}\n"
            "Possible causes:\n"
            "  - The left_discriminator or right_discriminator values match none of the features' options.\n"
            "  - The left compute framework of the link is not the compute framework of the child feature.\n"
            "Resolution: align the discriminator values with the options you set on the features, and declare "
            "the link on the compute framework the child is computed in.\n"
            f"If neither applies, please report this issue at {REPORT_URL} with the full traceback."
        )

    @staticmethod
    def _parents_linked_by_join(uuid_a: UUID, uuid_b: UUID, join_steps: set[JoinStep]) -> bool:
        """Whether two parents are linked, directly or transitively, via JoinSteps' genuine sides
        (not ``required_uuids``, which unions all of a join's consumers' parents, not just its own two)."""
        if uuid_a == uuid_b:
            return True

        adjacency: dict[UUID, set[UUID]] = defaultdict(set)
        for js in join_steps:
            for dest_uuid in js.destination_framework_uuids:
                adjacency[dest_uuid].update(js.source_framework_uuids)
            for src_uuid in js.source_framework_uuids:
                adjacency[src_uuid].update(js.destination_framework_uuids)

        visited = {uuid_a}
        frontier = {uuid_a}
        while frontier:
            frontier = set().union(*(adjacency[node] for node in frontier)) - visited
            if uuid_b in frontier:
                return True
            visited |= frontier
        return False

    @staticmethod
    def _conflicting_transform_hops_error(
        ep: FeatureGroupStep,
        first_hop: TransformFrameworkStep | _JoinServedParent,
        second_hop: TransformFrameworkStep | _JoinServedParent,
    ) -> str:
        """A FeatureGroupStep can only bind one incoming source; two distinct, unlinked ones is a
        missing-Link configuration problem, not a bug."""
        feature_name = format_feature_group_class(ep.feature_group)
        first_name = format_feature_group_class(first_hop.from_feature_group)
        second_name = format_feature_group_class(second_hop.from_feature_group)
        first_class_name = first_hop.from_feature_group.get_class_name()
        second_class_name = second_hop.from_feature_group.get_class_name()

        return f"""
Feature group '{feature_name}' depends on parents from two different, unlinked source feature
groups: '{first_name}' and '{second_name}'.

When a feature depends on multiple input features from different sources, you must provide explicit
Links to specify how to merge them. Without Links, the framework cannot determine how to combine the
data, and only one of the two sources would ever be read.

Option 1: Explicit JoinSpec (works with any feature group):
    from mloda.user import Link, JoinSpec

    links = {{
        Link.inner(
            JoinSpec({first_class_name}, "shared_column"),
            JoinSpec({second_class_name}, "shared_column"),
        )
    }}

Option 2: Shorthand via index_columns() (requires feature groups to define index_columns()):
    from mloda.user import Link

    links = {{
        Link.inner_on({first_class_name}, {second_class_name})
    }}

Available join types:
- Link.inner(left, right)    - Keep only matching rows from both sides
- Link.left(left, right)     - Keep all rows from left, matching from right
- Link.right(left, right)    - Keep all rows from right, matching from left
- Link.outer(left, right)    - Keep all rows from both sides
- Link.inner_on(left, right) - Shorthand using index_columns() definitions
""".strip()

    def raise_on_step_cycle(self, steps: Sequence[Step]) -> None:
        """Required tokens order the steps of the finished plan against each other, and a cycle would never run."""
        producer_of: dict[UUID, UUID] = {}
        steps_by_uuid: dict[UUID, Step] = {}
        for step in steps:
            steps_by_uuid[step.uuid] = step
            for token in step.get_uuids():
                producer_of[token] = step.uuid

        # A token no step produces is not a cycle; the runtime reports it as a missing producer.
        pending = {
            step.uuid: {producer_of[token] for token in step.required_uuids if token in producer_of} for step in steps
        }

        while True:
            ready = {uuid for uuid, waits_for in pending.items() if not waits_for}
            if not ready:
                break
            for uuid in ready:
                del pending[uuid]
            for waits_for in pending.values():
                waits_for -= ready

        if pending:
            raise ValueError(
                internal_invariant_error(
                    "the steps of the plan form a cycle.",
                    f"steps={sorted(_describe_step(steps_by_uuid[uuid]) for uuid in pending)}",
                )
            )

    def handle_append_or_union_joinstep(
        self,
        fw_execution_plan: list[JoinStep | FeatureGroupStep],
    ) -> list[JoinStep | FeatureGroupStep]:
        """
        This part is for the case that we have a join step with append or union.

        Example:
        UUID1 - UUID2 : UUID2 - UUID3 -> UUID1 must wait for UUID2 completion
        -> we add this to the required_uuids of the join step of UUID1

        We use two loops to make sure that we have the correct order.
        1) We map the destination framework uuid to the link uuid
        2) We use the mapping to update the required_uuids of the join step
        """

        map_destination_framework_uuid_to_link_uuid: dict[UUID, set[UUID]] = defaultdict(set)

        # Map the destination framework uuid to the link uuid
        for fw in fw_execution_plan:
            if isinstance(fw, JoinStep) and fw.link.jointype in (JoinType.APPEND, JoinType.UNION):
                if len(fw.destination_framework_uuids) > 1:
                    raise ValueError(
                        internal_invariant_error(
                            "APPEND/UNION JoinStep should have exactly 1 destination_framework_uuid.",
                            f"destination_framework_uuids={fw.destination_framework_uuids}, link={fw.link}",
                        )
                    )
                map_destination_framework_uuid_to_link_uuid[next(iter(fw.destination_framework_uuids))].add(
                    fw.link.uuid
                )

        # Use the mapping to update the required_uuids of the join step
        for fw in fw_execution_plan:
            if isinstance(fw, JoinStep) and fw.link.jointype in (JoinType.APPEND, JoinType.UNION):
                if len(fw.source_framework_uuids) > 1:
                    raise ValueError(
                        internal_invariant_error(
                            "APPEND/UNION JoinStep should have exactly 1 source_framework_uuid.",
                            f"source_framework_uuids={fw.source_framework_uuids}, link={fw.link}",
                        )
                    )

                source_framework_uuid = next(iter(fw.source_framework_uuids))
                required = map_destination_framework_uuid_to_link_uuid.get(source_framework_uuid)
                if required is not None:
                    fw.required_uuids.update(required)

        return fw_execution_plan

    def fill_tfs_by_joinstep(self, ep: JoinStep) -> TransformFrameworkStep:
        """The hop moves the source side into the destination side; swap_merge_sides names which side that is."""
        if ep.swap_merge_sides:
            from_feature_group, to_feature_group = ep.link.left_feature_group, ep.link.right_feature_group
        else:
            from_feature_group, to_feature_group = ep.link.right_feature_group, ep.link.left_feature_group

        return TransformFrameworkStep(
            from_framework=ep.source_framework,
            to_framework=ep.destination_framework,
            required_uuids=deepcopy(ep.required_uuids),
            from_feature_group=from_feature_group,
            to_feature_group=to_feature_group,
            link_id=ep.link.uuid,
            source_framework_uuids=ep.source_framework_uuids,
        )

    def add_tfs(
        self, execution_plan: list[JoinStep | FeatureGroupStep], graph: Graph
    ) -> list[TransformFrameworkStep | JoinStep | FeatureGroupStep]:
        new_execution_plan: list[TransformFrameworkStep | JoinStep | FeatureGroupStep] = []

        left_join_frameworks: set[JoinStep] = {ep for ep in execution_plan if isinstance(ep, JoinStep)}
        need_to_upload_collector: set[UUID] = set()

        # Features produced together by one FeatureGroupStep live on the same physical source cfw
        # instance, so a hop should key on the owning step, not each member feature's own uuid.
        owning_step_of: dict[UUID, UUID] = {
            feature_uuid: ep.uuid
            for ep in execution_plan
            if isinstance(ep, FeatureGroupStep)
            for feature_uuid in ep.get_uuids()
        }

        for ep in execution_plan:
            if isinstance(ep, JoinStep):
                if ep.destination_framework != ep.source_framework:
                    new_tfs = self.fill_tfs_by_joinstep(ep)

                    # Safe to reuse the canonical hop here: link_id is part of its identity, so both joins
                    # of this link re-find the hopped framework by link.uuid.
                    canonical_tfs = self.tfs_collection.get(new_tfs)
                    if canonical_tfs is None:
                        self.tfs_collection[new_tfs] = new_tfs
                        new_execution_plan.append(new_tfs)
                        canonical_tfs = new_tfs
                    ep.required_uuids.add(canonical_tfs.uuid)

                    need_to_upload_collector.update(ep.source_framework_uuids)

                    # We are updating the required uuids after the tfs is added as this makes sure, that the TFS can run in parallel before the join.
                    ep.required_uuids.update(self.joinstep_collection.get_required_join_uuids(ep))
                else:
                    # We need to do two things:
                    # 1) source feature group of the join step needs to know of the link, so that the cfw can be used by the joinstep
                    # 2) The child feature using this join needs to know which cfw to use. We use the tfs vehicle for this.
                    store_val = None

                    for inner_ep in execution_plan:
                        if isinstance(inner_ep, FeatureGroupStep):
                            # 1) We do 1 here:
                            for uuid in inner_ep.get_uuids():
                                if uuid in ep.source_framework_uuids:
                                    # add the link uuid to the children_if_root of the source feature group
                                    inner_ep.add_value_to_children_if_root(ep.link.uuid)

                                    # add to upload as this source feature group gets accessed in mp by other process
                                    need_to_upload_collector.update(ep.source_framework_uuids)
                                    break

                                if uuid in ep.destination_framework_uuids:
                                    # add the link uuid to the children_if_root of the destination feature group

                                    store_val = uuid

                            if store_val is None:
                                continue

                            # Check if any element of ep.destination_framework_uuids is in inner_ep.required_uuids
                            # same for source framework
                            if any(elem in inner_ep.required_uuids for elem in ep.destination_framework_uuids) and any(
                                elem in inner_ep.required_uuids for elem in ep.source_framework_uuids
                            ):
                                if ep.link.jointype in (JoinType.APPEND, JoinType.UNION):
                                    self.set_store_value_to_left_most_index_and_update_feature_group(
                                        inner_ep, store_val
                                    )
                                else:
                                    # Same value as any_uuid below: redundant with it, not load-bearing, but
                                    # lets get_unique_cfw_uuid short-circuit before add_compute_framework's locked path.
                                    inner_ep.tfs_ids = {store_val}
                                    inner_ep.features.any_uuid = (
                                        store_val  # Resets the any_uuid to one of the left side
                                    )

            elif isinstance(ep, FeatureGroupStep):
                if ep.features.any_uuid is None:
                    raise ValueError(f"Feature group {format_feature_group_class(ep.feature_group)} has no uuid.")

                parents: set[UUID] = set()
                for member_uuid in ep.get_uuids():
                    member_parents = graph.parent_to_children_mapping.get(member_uuid, set())
                    parents |= member_parents - self.get_parent_parents(member_parents, graph)

                # Explicit hops and join-served parents (delivered pre-merged by a JoinStep, no hop built)
                # both compete for this step's one binding, so both get grouped by the same linkage test
                # below. Order-independent: collected here, grouped once after the loop.
                bound_entries: list[tuple[TransformFrameworkStep | _JoinServedParent, UUID]] = []
                join_served_entries: list[tuple[type[FeatureGroup], UUID]] = []
                seen_hop_uuids: set[UUID] = set()

                for parent in parents:
                    parent_node_property = graph.get_nodes()[parent]
                    matching_join_steps = [
                        js
                        for js in left_join_frameworks
                        if js.matched(ep.compute_framework, parent_node_property.feature.uuid)
                    ]
                    if matching_join_steps:
                        # Served by a join, no explicit hop needed.
                        join_served_entries.append((parent_node_property.feature_group_class, parent))
                        continue

                    if ep.compute_framework != parent_node_property.feature.get_compute_framework():
                        new_tfs = TransformFrameworkStep(
                            from_framework=parent_node_property.feature.get_compute_framework(),
                            to_framework=ep.compute_framework,
                            required_uuids={parent},
                            from_feature_group=parent_node_property.feature_group_class,
                            to_feature_group=ep.feature_group,
                            source_step_uuid=owning_step_of.get(parent, parent),
                        )
                        canonical_tfs = self.tfs_collection.get(new_tfs)
                        if canonical_tfs is None:
                            self.tfs_collection[new_tfs] = new_tfs
                            new_execution_plan.append(new_tfs)
                            canonical_tfs = new_tfs

                        if canonical_tfs.uuid not in seen_hop_uuids:
                            seen_hop_uuids.add(canonical_tfs.uuid)
                            bound_entries.append((canonical_tfs, parent))

                        # Records every parent the hop covers; they all share one owning step, so
                        # this doesn't change the scheduling gate.
                        canonical_tfs.required_uuids.add(parent)
                        ep.required_uuids.add(canonical_tfs.uuid)

                        # Record the surviving hop's uuid so the step resolves its compute framework from it.
                        ep.tfs_ids.add(canonical_tfs.uuid)

                        need_to_upload_collector.add(parent)

                # Group entries by transitive linkage: same feature-group class, one entry's own class a
                # subclass (or superclass) of the other's (catches a case-override hop whose parent lost the
                # JoinStep's own uuid to a same-role sibling, see `_case_override_beats_nearer_wrong_framework_left`,
                # without also bridging two entries that merely share an unrelated common ancestor via some
                # third join's declared side), or `_parents_linked_by_join`.
                def _entries_linked(
                    entry_a: tuple[TransformFrameworkStep | _JoinServedParent, UUID],
                    entry_b: tuple[TransformFrameworkStep | _JoinServedParent, UUID],
                ) -> bool:
                    hop_a, parent_a = entry_a
                    hop_b, parent_b = entry_b
                    if hop_a.from_feature_group is hop_b.from_feature_group:
                        return True
                    if issubclass(hop_a.from_feature_group, hop_b.from_feature_group) or issubclass(
                        hop_b.from_feature_group, hop_a.from_feature_group
                    ):
                        return True
                    return self._parents_linked_by_join(parent_a, parent_b, left_join_frameworks)

                def _add_to_groups(
                    groups: list[list[tuple[TransformFrameworkStep | _JoinServedParent, UUID]]],
                    entry: tuple[TransformFrameworkStep | _JoinServedParent, UUID],
                ) -> None:
                    linked_groups = [
                        group for group in groups if any(_entries_linked(entry, member) for member in group)
                    ]
                    if linked_groups:
                        target_group = linked_groups[0]
                        target_group.append(entry)
                        for other_group in linked_groups[1:]:
                            target_group.extend(other_group)
                            groups.remove(other_group)
                    else:
                        groups.append([entry])

                hop_groups: list[list[tuple[TransformFrameworkStep | _JoinServedParent, UUID]]] = []
                for entry in bound_entries:
                    _add_to_groups(hop_groups, entry)

                # Join-served parents compete for the same binding, so merge them into the same groups too.
                for served_feature_group, served_parent in join_served_entries:
                    _add_to_groups(hop_groups, (_JoinServedParent(served_feature_group), served_parent))

                if len(hop_groups) > 1:
                    raise ValueError(
                        self._conflicting_transform_hops_error(ep, hop_groups[0][0][0], hop_groups[1][0][0])
                    )

            else:
                raise ValueError(f"Element {ep} is not a valid element.")
            new_execution_plan.append(ep)

        # We define that every parent of a transform framework step needs to be uploaded.
        # This step is only relevant for multi processing.
        #
        # One pass over the finished plan, not one per appended step: the marking is
        # monotone (only ever set to True, and need_to_upload_collector only grows),
        # nothing inside add_tfs reads need_to_upload, and no step escapes the list
        # mid-build - so a step ends up marked iff it is non-disjoint from the FINAL
        # collector either way. That drops the pass from O(steps^2 * features) to
        # O(steps * features).
        for _ep in new_execution_plan:
            if isinstance(_ep, FeatureGroupStep):
                if not need_to_upload_collector.isdisjoint(_ep.get_uuids()):
                    _ep.need_to_upload = True

        return new_execution_plan

    def set_store_value_to_left_most_index_and_update_feature_group(
        self, inner_ep: FeatureGroupStep, store_val: UUID
    ) -> None:
        """
        Sets the `store_val` to the left-most index and updates the given feature group step.

        This is during runtime used to identify correct compute framework.

        Args:
            inner_ep (FeatureGroupStep): The step to update.
            store_val (UUID): The value to set as the latest UUID.
        """
        joinsteps = self.joinstep_collection.collection

        # Step 1: Identify all left-most and right-most indexes
        left_indexes: set[Index] = set()
        right_indexes: set[Index] = set()

        for js, _ in joinsteps.items():
            # Skip if the index does not belong to the FeatureGroupStep.
            if js.link.left_feature_group != inner_ep.feature_group:
                continue

            if not left_indexes:
                # Initialize with the first left and right indexes
                left_indexes.add(js.link.left_index)
                right_indexes.add(js.link.right_index)
                continue

            elif js.link.left_index in right_indexes:
                # If the left index is already in the right set, update both
                right_indexes.add(js.link.left_index)
                right_indexes.add(js.link.right_index)
                continue
            else:
                # Otherwise, add new left and right indexes
                left_indexes.add(js.link.left_index)
                right_indexes.add(js.link.right_index)

        # Step 2: Reduce to a single left-most index (Should be the only one left)
        for js, _ in joinsteps.items():
            _right = js.link.right_index
            # Use a copy of left_indexes to safely modify the set
            for left_index in list(left_indexes):
                if left_index == _right:
                    left_indexes.remove(left_index)

        if len(left_indexes) == 0:
            return

        if len(left_indexes) > 1:
            raise ValueError("Expected exactly one left-most index, but found multiple or none.")

        left_most_index = next(iter(left_indexes))  # Extract the single left-most index

        # Step 3: Update the relevant fields in `inner_ep` based on conditions
        right_memory_index: set[Index] = set()

        for js, _ in joinsteps.items():
            # Skip if the left index is already in the memory index
            if right_memory_index:
                if js.link.left_index in (right_memory_index):
                    continue

            # Initialize the memory index with the first right index
            if not right_memory_index:
                right_memory_index.add(js.link.right_index)

            # Only update when this is the left-most index and belongs to the join step's destination framework.
            if store_val == next(iter(js.destination_framework_uuids)) and left_most_index == js.link.left_index:
                inner_ep.tfs_ids = {store_val}
                inner_ep.features.any_uuid = store_val

    def get_parent_parents(self, parents: set[UUID], graph: Graph) -> set[UUID]:
        parent_parents = set()
        for parent in parents:
            parent_parent = graph.parent_to_children_mapping.get(parent, set())
            if len(parent_parent) > 0:
                parent_parents.update(parent_parent)
        return parent_parents

    @staticmethod
    def _validate_join_step_uuids(
        link: Link,
        destination_framework_uuids: set[UUID],
        source_framework_uuids: set[UUID],
    ) -> None:
        """Both JoinStep sides must name at least one parent; the runtime later reads
        them with next(iter(...))."""
        if not destination_framework_uuids or not source_framework_uuids:
            raise ValueError(
                internal_invariant_error(
                    "run_link resolved an empty destination_framework_uuids or source_framework_uuids.",
                    f"link={link}, destination_framework_uuids={destination_framework_uuids}, "
                    f"source_framework_uuids={source_framework_uuids}",
                )
            )

    def run_link(
        self,
        link_fw: LinkFrameworkTrekker,
        link_trekker: LinkTrekker,
        graph: Graph,
        pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep],
    ) -> JoinStep | None:
        link = link_fw[0]
        destination_framework = link_fw[1]
        source_framework = link_fw[2]

        if link.jointype == JoinType.RIGHT:
            destination_framework = link_fw[2]
            source_framework = link_fw[1]

        # This gets the id of the children which needs the link to be calculated.
        children_uuids: set[UUID] = set()
        attempted_key = link_fw

        children_uuids.update(link_trekker.data.get(link_fw, set()))

        swap_merge_sides = False

        # The queue snapshots trekker keys before ResolveComputeFrameworks.links runs; LinkTrekker.invert_link
        # can later re-key an inverted link, so link_fw may already be stale here. This branch re-resolves
        # the link under its flipped key, the normal path for an inverted-orientation join.
        if len(children_uuids) == 0:
            # No child needs the declared orientation, so destination and source are the other way around.
            destination_framework = link_fw[2]
            source_framework = link_fw[1]
            # The join then executes in the right feature group's framework, so the merge arguments are inverted.
            swap_merge_sides = True
            attempted_key = (link, destination_framework, source_framework)

            children_uuids.update(link_trekker.data.get(attempted_key, set()))

            if len(children_uuids) == 0:
                raise ValueError(f"Link {link} has no matching uuids.")

        children_uuids = self.reduce_children_to_one_level(children_uuids, graph)

        # This gets the parent ids of the joinstep, which needs to be calculated before the link.
        required_uuids: set[UUID] = set()
        for uuid in children_uuids:
            required_uuids.update(graph.parent_to_children_mapping[uuid])

        # Split before the order-edge link uuids join required_uuids.
        split = split_by_declared_side(link, required_uuids, graph)

        # This filters the required_uuids to only the one with the final compute framework.
        destination_framework_uuids: set[UUID] = set()
        source_framework_uuids: set[UUID] = set()

        for uuid in required_uuids:
            node_framework = graph.get_nodes()[uuid].feature.get_compute_framework()

            if node_framework == destination_framework:
                destination_framework_uuids.add(uuid)

            if node_framework == source_framework:
                source_framework_uuids.add(uuid)

        declared_left_frameworks = {graph.get_nodes()[u].feature.get_compute_framework() for u in split.left_uuids}
        declared_right_frameworks = {graph.get_nodes()[u].feature.get_compute_framework() for u in split.right_uuids}
        widened_right_frameworks = {
            graph.get_nodes()[u].feature.get_compute_framework()
            for u in split.right_uuids_any_distance - split.left_uuids
        }

        # The order shows which items should be added first.
        # Thus, we need to make sure that higher ordered links are calculated first.
        for k, v in link_trekker.order.items():
            if link.uuid in v:
                required_uuids.add(k)

        # Potential  -> This should be the feature uuid of the child of the joinstep. Can this be more than 1?
        # This part can be dropped if we have more tests.
        # if len(children_uuids) > 1:
        #    raise ValueError("This is not supported yet.")

        # Hoisted above the case-override loop: a case helper's result is in declared left/right
        # order, so orienting it needs swap_sides.
        swap_sides = self.swap_merge_sides_by_declared_side(
            destination_framework=destination_framework,
            source_framework=source_framework,
            trekker_left_framework=link_fw[1],
            declared_left_frameworks=declared_left_frameworks,
            declared_right_frameworks=declared_right_frameworks,
            widened_right_frameworks=widened_right_frameworks,
            fallback=swap_merge_sides,
            jointype=link.jointype,
        )

        # This part is for handling specific join cases. Currently, we only deal with equal feature groups.
        for children_uuid in children_uuids:
            children_fw = graph.get_nodes()[children_uuid].feature.get_compute_framework()

            # This runs with the assumption that children_uuids is exactly 1.
            # result = True
            result = self.is_valid_join_step(link_fw, children_fw, children_uuid, graph)
            if result is False:
                self.declined_orientations.append(attempted_key)
                return None
            elif result is True:
                pass
            else:
                # case_link_fw_is_equal_to_children_fw and case_link_equal_feature_groups both
                # guarantee, by construction, that result[0] runs on link_fw[1] and result[1] runs
                # on link_fw[2]; unlike the nearest-split-derived swap_sides, that framework
                # identity cannot disagree with which side the case helper actually bound. Only
                # fall back to swap_sides when the frameworks coincide and identity cannot decide.
                # The destination side follows the same identity, so the record and the merge
                # argument order agree with the parents actually bound.
                if destination_framework != source_framework:
                    if destination_framework == link_fw[1]:
                        destination_framework_uuids, source_framework_uuids = result
                        swap_sides = False
                    else:
                        source_framework_uuids, destination_framework_uuids = result
                        swap_sides = True
                elif swap_sides:
                    source_framework_uuids, destination_framework_uuids = result
                else:
                    destination_framework_uuids, source_framework_uuids = result

        join_step_required_uuids: set[UUID]
        if link.jointype in (JoinType.APPEND, JoinType.UNION):
            sides = self.resolve_append_or_union_sides(link, link_fw, required_uuids, graph, pre_execution_plan)
            destination_framework = sides.destination_framework
            source_framework = sides.source_framework
            side = JoinSide.LEFT
            destination_framework_uuids = {sides.left_uuid}
            source_framework_uuids = {sides.right_uuid}
            left_uuids = frozenset({sides.left_uuid})
            right_uuids = frozenset({sides.right_uuid})
            # Append/union gates only on its own two feature uuids, not on the general required_uuids.
            join_step_required_uuids = {sides.left_uuid, sides.right_uuid}
            join_uuids_left, join_uuids_right = left_uuids, right_uuids
        else:
            side = JoinSide.RIGHT if swap_sides else JoinSide.LEFT
            destination = frozenset(destination_framework_uuids)
            source = frozenset(source_framework_uuids)
            resolved_left, resolved_right = (destination, source) if side is JoinSide.LEFT else (source, destination)
            left_from_split = split.left_uuids & resolved_left
            right_from_split = split.right_uuids & resolved_right
            if (
                split.left_uuids <= resolved_left
                and split.right_uuids <= resolved_right
                and split.left_uuids != split.right_uuids
            ):
                left_uuids, right_uuids = split.left_uuids, split.right_uuids
            elif left_from_split and right_from_split and left_from_split != right_from_split:
                # The full containment check failed (the declared side spans more than one framework), but
                # intersecting the declared split with the framework-resolved buckets still recovers the
                # declared-side members that fall in this step's own framework bucket, and drops any
                # unrelated parent that only shares a framework with one side. A declared-side member
                # sitting in the *other* bucket is not recovered here; it belongs to a different join
                # step/framework hop and is dropped by design.
                left_uuids, right_uuids = left_from_split, right_from_split
            else:
                # The step's own sets; a same-framework self link lands here too. Framework-broad, so
                # join_uuids_left/right below narrow independently rather than reusing these.
                left_uuids, right_uuids = resolved_left, resolved_right
            join_step_required_uuids = required_uuids

            # destination_uuids/source_uuids must only ever name genuine declared-side members, regardless
            # of which branch above ran; any-distance widening keeps a nearer wrong-framework sibling from
            # hiding a farther, correct one.
            declared_side_uuids = split.left_uuids_any_distance | split.right_uuids_any_distance
            join_uuids_left = resolved_left & declared_side_uuids
            join_uuids_right = resolved_right & declared_side_uuids

        destination_uuids, source_uuids = (
            (join_uuids_right, join_uuids_left) if side is JoinSide.RIGHT else (join_uuids_left, join_uuids_right)
        )
        self._validate_join_step_uuids(link, set(destination_uuids), set(source_uuids))

        record = ResolvedJoin(
            link_uuid=link.uuid,
            jointype=link.jointype,
            left=build_resolved_join_side(
                link.left_feature_group, link.left_index, left_uuids, self.declared_frameworks
            ),
            right=build_resolved_join_side(
                link.right_feature_group, link.right_index, right_uuids, self.declared_frameworks
            ),
            destination_side=side,
            destination_uuids=frozenset(destination_uuids),
            source_uuids=frozenset(source_uuids),
            destination_framework=destination_framework,
            source_framework=source_framework,
            consumers=frozenset(children_uuids),
            depends_on=frozenset(),
            token=uuid4(),
        )
        js = JoinStep(
            link=link,
            destination_framework=record.destination_framework,
            source_framework=record.source_framework,
            required_uuids=join_step_required_uuids,
            destination_framework_uuids=set(record.destination_uuids),
            source_framework_uuids=set(record.source_uuids),
            swap_merge_sides=record.inverted,
            token=record.token,
        )
        self.planned_records.append(record)

        # This makes sure that we do not write on the same datasets due to overlapping joins at once.
        self.joinstep_collection.add(js)
        return js

    @staticmethod
    def swap_merge_sides_by_declared_side(
        destination_framework: type[ComputeFramework],
        source_framework: type[ComputeFramework],
        trekker_left_framework: type[ComputeFramework],
        declared_left_frameworks: set[type[ComputeFramework]],
        declared_right_frameworks: set[type[ComputeFramework]],
        widened_right_frameworks: set[type[ComputeFramework]],
        fallback: bool,
        jointype: JoinType,
    ) -> bool:
        """The declared left group's data must stay the merge engine's left argument, wherever the join runs.

        Declared-side membership decides first, whenever exactly one side names the destination framework.
        For a key in declared order, `run_link` sets destination_framework to link_fw[2] for JoinType.RIGHT,
        and link_fw[2] there is the declared right framework, so membership settles on right. For a key
        reversed upstream by `LinkTrekker.invert_link`, link_fw[2] instead holds the declared left framework,
        and membership settles on left just the same, before the jointype check below ever runs; that is why
        the jointype check is ordered after the membership checks, not before it. See
        `test_a_right_join_reached_through_a_reversed_key_keeps_the_declared_merge_sides` for that reversed
        case. Only when membership is genuinely ambiguous (both sides silent, or both claiming the
        destination framework, which happens when left and right share one framework) does a RIGHT join fall
        through to jointype, which then always resolves to the declared right side. For other jointypes the
        trekker key breaks the tie instead: destination is always one of the key's two framework positions.
        ``trekker_left_framework`` is that key's first position (``link_fw[1]``): it is the destination
        exactly when `run_link` kept the queued (non-flipped) orientation, and the source when it flipped.
        It does not reliably mean "declared left", for the same reversed-key reason above. Links keyed on one
        single framework make the tie-break tautological, so the trekker-flip fallback stays for that case:
        it is a common path in practice, not a rare or unreachable one, hit by ordinary same-framework
        INNER/LEFT/APPEND/UNION/ASOF joins and self-joins throughout the test suite. When destination and
        source frameworks differ, a single-sided membership answer is trusted only after checking the other
        side's full (any-distance) candidates for a competing claim on the destination framework."""
        holds_left = destination_framework in declared_left_frameworks
        holds_right = destination_framework in declared_right_frameworks

        if jointype == JoinType.RIGHT and destination_framework != source_framework:
            if holds_left and not holds_right and destination_framework in widened_right_frameworks:
                # A farther, non-canonical right parent also sits on the destination framework, so the
                # nearest-only split's silence on the right side is not real ambiguity-free evidence.
                holds_right = True

        if holds_left and not holds_right:
            return False
        if holds_right and not holds_left:
            return True
        if jointype == JoinType.RIGHT:
            return True
        if destination_framework != source_framework:
            return destination_framework != trekker_left_framework
        return fallback

    def find_fg_per_uuid(
        self, pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep], uuid: UUID
    ) -> type[FeatureGroup]:
        """
        This function finds the feature group per UUID in the pre_execution_plan.

        This can certainly be optimized, but for now, this is the easiest.
        """
        for element in pre_execution_plan:
            if isinstance(element, FeatureGroupStep):
                if uuid in element.get_uuids():
                    return element.feature_group
        raise ValueError(f"Feature group for UUID {uuid} not found.")

    @staticmethod
    def _append_or_union_orientation_error(
        link: Link,
        side: str,
        queued_framework: type[ComputeFramework],
        resolved_framework: type[ComputeFramework],
    ) -> str:
        return (
            f"{link.jointype.value} link {link} cannot run: the {side} side was queued on "
            f"{queued_framework.get_class_name()}, but the link's declared {side} index feature resolves to "
            f"{resolved_framework.get_class_name()}.\n"
            "One possible cause is that the link got scheduled in an inverted orientation; unlike INNER/LEFT/RIGHT "
            "links, APPEND and UNION links do not support inversion.\n"
            "Resolution: keep the link's declared left/right sides aligned with the compute frameworks its "
            "features resolve to."
        )

    def resolve_append_or_union_sides(
        self,
        link: Link,
        link_fw: LinkFrameworkTrekker,
        required_uuids: set[UUID],
        graph: Graph,
        pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep],
    ) -> AppendOrUnionSides:
        """Resolve the left/right feature uuids and frameworks for an APPEND or UNION link; neither
        inverts, so the resolved sides stay in declared order."""

        # Unpack link-related data
        left_index, right_index = link.left_index, link.right_index
        left_feature_group, right_feature_group = link.left_feature_group, link.right_feature_group

        # Initialize variables for feature UUIDs and frameworks
        left_feature_uuid = None
        right_feature_uuid = None
        destination_framework, source_framework = link_fw[1], link_fw[2]

        # Identify the left and right feature UUIDs
        for uuid in required_uuids:
            # Skip non-feature UUIDs
            if uuid not in graph.get_nodes():
                continue

            # Get the feature, its index and feature groups
            feature = graph.get_nodes()[uuid].feature
            feature_feature_group = self.find_fg_per_uuid(pre_execution_plan, uuid)
            feature_index = feature.index
            if feature_index is None:
                continue

            # Match the left index and feature group
            if left_index == feature_index and feature_feature_group == left_feature_group:
                if left_feature_uuid is not None:
                    raise ValueError(f"Are the indexes for append or union set double? {left_index}")
                destination_framework = feature.get_compute_framework()
                left_feature_uuid = uuid

            # Match the right index and feature group
            if right_index == feature_index and feature_feature_group == right_feature_group:
                if right_feature_uuid is not None:
                    raise ValueError(f"Are the indexes for append or union set double? {right_index}")
                right_feature_uuid = uuid
                source_framework = feature.get_compute_framework()

        # Validate that both feature UUIDs are identified
        if left_feature_uuid is None or right_feature_uuid is None:
            raise ValueError(
                f"Are the indexes for the append or union set correctly? {left_index.index, right_index.index}"
            )

        if link_fw[1] != destination_framework:
            raise ValueError(self._append_or_union_orientation_error(link, "left", link_fw[1], destination_framework))

        if link_fw[2] != source_framework:
            raise ValueError(self._append_or_union_orientation_error(link, "right", link_fw[2], source_framework))

        return AppendOrUnionSides(destination_framework, source_framework, left_feature_uuid, right_feature_uuid)

    def reduce_children_to_one_level(self, children_uuids: set[UUID], graph: Graph) -> set[UUID]:
        """
        We reduce the children to one level. This is needed for the joinstep creation.
        """

        new_children_uuids: set[UUID] = copy(children_uuids)
        for child in children_uuids:
            child_of_child = graph.adjacency_list[child]

            for c_o_c in child_of_child:
                if c_o_c in children_uuids:
                    new_children_uuids.remove(c_o_c)

        return new_children_uuids

    def is_valid_join_step(
        self,
        link_fw: LinkFrameworkTrekker,
        children_fw: type[ComputeFramework],
        children_uuid: UUID,
        graph: Graph,
    ) -> bool | tuple[set[UUID], set[UUID]]:
        """Identify if the join is valid. If not, this marks it as invalid and returns False."""

        # Check that we handle links with equal feature groups specifically!
        if link_fw[0].left_feature_group == link_fw[0].right_feature_group:
            result = self.case_link_equal_feature_groups(link_fw, children_fw, children_uuid, graph)
            if result is False:
                return False
            return result

        # Check that we handle links where left cfw == children cfw
        if link_fw[1] == children_fw:
            result = self.case_link_fw_is_equal_to_children_fw(link_fw, children_uuid, graph)
            if result is False:
                return False
            return result
        return True

    def case_link_fw_is_equal_to_children_fw(
        self, link_fw: LinkFrameworkTrekker, children_uuid: UUID, graph: Graph
    ) -> bool | tuple[set[UUID], set[UUID]]:
        # get feature which could be left
        parents = graph.parent_to_children_mapping[children_uuid]
        local_feature_set_collection = deepcopy(self.feature_set_collections)
        feature_set_collection_per_uuid = self.find_feature_uuids(parents, local_feature_set_collection)

        if len(feature_set_collection_per_uuid) == 0:
            raise ValueError(
                internal_invariant_error(
                    "feature_set_collection_per_uuid is empty in case_link_fw_is_equal_to_children_fw.",
                    f"parents={parents}, link={link_fw[0]}, children_uuid={children_uuid}",
                    "The feature set collections do not contain any of the parent UUIDs.",
                )
            )

        valid_pairs: list[tuple[set[UUID], set[UUID]]] = []

        for uuid, uuid_complete in feature_set_collection_per_uuid.items():
            # get the feature set collection, where feature cfw = left link cfw
            if link_fw[1] != graph.nodes[uuid].feature.get_compute_framework():
                continue

            # Use polymorphic matching: concrete class should be subclass of link's base class
            if not issubclass(graph.nodes[uuid].feature_group_class, link_fw[0].left_feature_group):
                continue

            if link_fw[0].left_discriminator is not None:
                if not self._matches_discriminator(link_fw[0].left_discriminator, graph, uuid):
                    continue

            # loop over all other feature set collections
            for _uuid, _uuid_complete in feature_set_collection_per_uuid.items():
                if uuid == _uuid:
                    continue

                # get the feature set collection, where feature cfw = right link cfw
                if link_fw[2] != graph.nodes[_uuid].feature.get_compute_framework():
                    continue

                # Use polymorphic matching: concrete class should be subclass of link's base class
                if not issubclass(graph.nodes[_uuid].feature_group_class, link_fw[0].right_feature_group):
                    continue

                if link_fw[0].right_discriminator is not None:
                    if not self._matches_discriminator(link_fw[0].right_discriminator, graph, _uuid):
                        continue

                # Deduplicate using set equality
                if not any(left == uuid_complete and r == _uuid_complete for left, r in valid_pairs):
                    valid_pairs.append((uuid_complete, _uuid_complete))

        if len(valid_pairs) == 1:
            return valid_pairs[0]
        elif len(valid_pairs) == 0:
            return False

        # Secondary disambiguation: use right_index to pick the correct right batch
        right_index = link_fw[0].right_index
        if right_index is not None:
            # First pass: match by feature.index == link.right_index
            filtered = [
                (left, r)
                for left, r in valid_pairs
                if any(
                    graph.nodes[u].feature.index is not None and graph.nodes[u].feature.index == right_index
                    for u in r
                    if u in graph.nodes
                )
            ]
            if len(filtered) == 1:
                return filtered[0]

            # Second pass: match by feature name appearing in right_index columns
            if not filtered:
                filtered = [
                    (left, r)
                    for left, r in valid_pairs
                    if any(graph.nodes[u].feature.name in right_index.index for u in r if u in graph.nodes)
                ]
                if len(filtered) == 1:
                    return filtered[0]

        # check that we only support non-right joins for equal/polymorphic feature groups
        if link_fw[0].jointype == JoinType.RIGHT:
            raise Exception(
                f"Right joins are not supported for equal or polymorphic feature groups. link: {link_fw[0]}"
            )

        raise ValueError(
            "There are more than one solution for the join. "
            "If you encounter this, check your links and feature group configuration, "
            "or contact the mloda developers."
        )

    def case_link_equal_feature_groups(
        self,
        link_fw: LinkFrameworkTrekker,
        children_fw: type[ComputeFramework],
        children_uuid: UUID,
        graph: Graph,
    ) -> bool | tuple[set[UUID], set[UUID]]:
        """
        If we have equal feature groups in the link object, this creates an interesting scenario.

        The algorithm does not know in which order it should join these features.
        We handle this case with some assumptions:

        1) We only support non-right joins for equal feature groups.
        2) Left join cfw should be the child cfw and the left feature cfw.
        3) We only support one solution for the join.

        I have for now not thought if this is algorithmically enough for all cases.
        If that is the case, we might need to adjust the graph algorithm part.

        To date, my first concern is that people use this framework.
        If you find a use case needing different support here, please contact mloda developers.
        """

        # check that we only support non-right joins for equal/polymorphic feature groups
        if link_fw[0].jointype == JoinType.RIGHT:
            raise Exception(
                f"Right joins are not supported for equal or polymorphic feature groups. link: {link_fw[0]}"
            )

        # check that the compute framework of the child_fw is similar to the left cfw as this is the target cfw
        if link_fw[1] != children_fw:
            return False

        # get feature which could be left
        parents = graph.parent_to_children_mapping[children_uuid]
        local_feature_set_collection = deepcopy(self.feature_set_collections)
        feature_set_collection_per_uuid = self.find_feature_uuids(parents, local_feature_set_collection)

        if len(feature_set_collection_per_uuid) == 0:
            raise ValueError(
                internal_invariant_error(
                    "feature_set_collection_per_uuid is empty in case_link_equal_feature_groups.",
                    f"parents={parents}, link={link_fw[0]}, children_uuid={children_uuid}",
                    "The feature set collections do not contain any of the parent UUIDs.",
                )
            )

        unique_solution_counter = 0
        left_uuids = None
        right_uuids = None

        for uuid, uuid_complete in feature_set_collection_per_uuid.items():
            # get the feature set collection, where feature cfw = left link cfw
            if link_fw[1] != graph.nodes[uuid].feature.get_compute_framework():
                continue

            if link_fw[0].left_discriminator is not None:
                if not self.check_pointer(link_fw[0].left_discriminator, link_fw, graph, uuid):
                    continue

            # loop over all other feature set collections
            for _uuid, _uuid_complete in feature_set_collection_per_uuid.items():
                if uuid == _uuid:
                    continue

                # get the feature set collection, where feature cfw = right link cfw
                if link_fw[2] != graph.nodes[_uuid].feature.get_compute_framework():
                    continue

                if link_fw[0].right_discriminator is not None:
                    if not self.check_pointer(
                        link_fw[0].right_discriminator,
                        link_fw,
                        graph,
                        _uuid,
                    ):
                        continue
                # This should be the only solution
                left_uuids = uuid_complete
                right_uuids = _uuid_complete
                unique_solution_counter += 1

        # handle append, union
        if link_fw[0].jointype in (JoinType.APPEND, JoinType.UNION):
            if left_uuids is None or right_uuids is None:
                raise ValueError(
                    f"Could not resolve left/right UUIDs for APPEND/UNION join.\n"
                    f"link={link_fw[0]}, left_uuids={left_uuids}, right_uuids={right_uuids}\n"
                    "Possible causes:\n"
                    "  - The index was not set for the append or union Link.\n"
                    "  - The features are not unique (Link hash alone does not distinguish them).\n"
                    "Resolution: Set distinct options on each feature to make them unique, "
                    "or ensure each side of the Link has an explicit index.\n"
                    f"Please report this issue at https://github.com/mloda-ai/mloda/issues "
                    f"if the problem persists."
                )
            if unique_solution_counter > 0:
                return (left_uuids, right_uuids)
            else:
                return False

        if unique_solution_counter == 1:
            if left_uuids is None or right_uuids is None:
                raise ValueError(
                    internal_invariant_error(
                        "unique_solution_counter is 1 but left_uuids or right_uuids is None.",
                        f"left_uuids={left_uuids}, right_uuids={right_uuids}, link={link_fw[0]}",
                    )
                )
            return (left_uuids, right_uuids)
        elif unique_solution_counter == 0:
            return False
        else:
            raise ValueError(
                "Multiple same-class FeatureGroup nodes found with no discriminator set. "
                "When linking two nodes of the same FeatureGroup class (e.g. the same ReadFileFeature "
                "loading different files), use left_discriminator and right_discriminator on your Link "
                "to identify which node is left and which is right. "
                "Example: Link.inner(JoinSpec(MyFG, 'id'), JoinSpec(MyFG, 'id'), "
                "left_discriminator={'CsvReader': 'file_a.csv'}, "
                "right_discriminator={'CsvReader': 'file_b.csv'}). "
                "The discriminator values must match the corresponding feature's options."
            )

    def _matches_discriminator(self, discriminator: dict[str, Any], graph: Graph, uuid: UUID) -> bool:
        """Check that every discriminator key-value pair is present in a node's feature options.

        A discriminator identifies one node among several same-class FeatureGroup instances, so a
        partial overlap is not enough: two nodes that differ on the deciding key but share another
        one would both match.
        """
        options = graph.nodes[uuid].feature.options
        for dk, dv in discriminator.items():
            if dk not in options or options.get(dk) != dv:
                return False
        return True

    def check_pointer(
        self, pointer_dict: dict[str, Any], link_fw: LinkFrameworkTrekker, graph: Graph, uuid: UUID
    ) -> bool:
        if link_fw[0].right_discriminator is None:
            raise ValueError(
                internal_invariant_error(
                    "right_discriminator is None while left_discriminator is set in check_pointer.",
                    f"left_discriminator={link_fw[0].left_discriminator}, "
                    f"right_discriminator={link_fw[0].right_discriminator}",
                    "When using discriminators for same-class FeatureGroup links, both "
                    "left_discriminator and right_discriminator must be provided.",
                )
            )

        if link_fw[0].left_discriminator is None:
            raise ValueError(
                internal_invariant_error(
                    "left_discriminator is None while right_discriminator is set in check_pointer.",
                    f"left_discriminator={link_fw[0].left_discriminator}, "
                    f"right_discriminator={link_fw[0].right_discriminator}",
                    "When using discriminators for same-class FeatureGroup links, both "
                    "left_discriminator and right_discriminator must be provided.",
                )
            )

        return self._matches_discriminator(pointer_dict, graph, uuid)

    def find_feature_uuids(
        self, parents: set[UUID], local_feature_set_collection: list[set[UUID]]
    ) -> dict[UUID, set[UUID]]:
        """
        We group the feature_uuids by the feature_set_collection, which represent features of one concrete feature group (step).
        """
        feature_set_collection_per_uuid = defaultdict(set)
        already_used_parents = set()
        for parent in parents:
            if parent in already_used_parents:
                continue
            for feature_uuids in local_feature_set_collection:
                if parent in feature_uuids:
                    feature_set_collection_per_uuid[parent].update(feature_uuids)
                    already_used_parents.update(feature_uuids)
        return feature_set_collection_per_uuid

    def _split_features_by_dependency_levels(
        self, features: set[Feature], parent_to_children_mapping: dict[UUID, set[UUID]]
    ) -> list[set[Feature]]:
        feature_uuids = {f.uuid for f in features}
        uuid_to_feature = {f.uuid: f for f in features}

        intra_deps: dict[UUID, set[UUID]] = {}
        for feature in features:
            ancestors = parent_to_children_mapping.get(feature.uuid, set())
            intra_deps[feature.uuid] = ancestors & feature_uuids

        if not any(deps for deps in intra_deps.values()):
            return [features]

        levels: list[set[Feature]] = []
        remaining = set(feature_uuids)
        placed: set[UUID] = set()

        while remaining:
            ready = {uuid for uuid in remaining if intra_deps[uuid].issubset(placed)}
            if not ready:
                ready = remaining

            levels.append({uuid_to_feature[uuid] for uuid in ready})
            placed.update(ready)
            remaining -= ready

        return levels

    def run_feature_group(
        self,
        feature_group_features: tuple[type[FeatureGroup], set[Feature]],
        parent_to_children_mapping: dict[UUID, set[UUID]],
        pre_required_uuids: set[UUID],
    ) -> dict[Any, FeatureGroupStep]:
        feature_group, features = feature_group_features[0], feature_group_features[1]
        features_grouped_by_framework_and_options = self.group_features_by_compute_framework_and_options(features)

        fg_steps: dict[Any, FeatureGroupStep] = {}

        root_parent_children_mapping = self.get_parent_children_mapping(parent_to_children_mapping)

        for f_hash, features in features_grouped_by_framework_and_options.items():
            sub_groups = self._split_features_by_dependency_levels(features, parent_to_children_mapping)

            for level_idx, sub_features in enumerate(sub_groups):
                pre_calculated = self.retrieve_nodes_which_must_be_calculated_before(
                    sub_features, parent_to_children_mapping
                )
                pre_calculated.update(copy(pre_required_uuids))

                cf = next(iter(sub_features)).get_compute_framework()

                children_if_root = set()
                for feature in sub_features:
                    if feature.uuid in root_parent_children_mapping:
                        children_if_root.update(root_parent_children_mapping[feature.uuid])

                feature_set = FeatureSet()
                for feature in sub_features:
                    feature_set.add(feature)
                    feature.name

                self.feature_set_collections.append(feature_set.get_all_feature_ids())

                if self.resolved_input_feature_names is not None:
                    # An injected filter or index feature is batched with its host and takes the host's
                    # inputs, so the union over the resolved members is what the engine wired for the step.
                    union: set[str] = set()
                    for feature in sub_features:
                        union.update(self.resolved_input_feature_names.get(feature.uuid) or frozenset())
                    feature_set.declared_input_feature_names = frozenset(union) or None
                    feature_set.declared_input_features_resolved = True

                self.add_artifact_to_feature_set(feature_group, feature_set)
                self.add_single_filters_to_feature_set(feature_group, feature_set)

                feature_group_step = FeatureGroupStep(
                    feature_group,
                    feature_set,
                    pre_calculated,
                    cf,
                    children_if_root,
                    self.prepare_api_input_data(feature_group, feature_set),
                )

                fg_steps[(f_hash, level_idx)] = feature_group_step
        return fg_steps

    def prepare_api_input_data(self, feature_group: type[FeatureGroup], feature_set: FeatureSet) -> bool | BaseApiData:
        if not isinstance(feature_group.input_data(), ApiInputData):
            return False

        if self.api_input_data_collection is None:
            raise ValueError(
                f"Feature group {feature_group} has an api input data class, but no api_input_data_collection was given."
            )

        if feature_set.get_name_of_one_feature() is None:
            raise ValueError(f"Feature group {format_feature_group_class(feature_group)} has no feature set name.")

        api_input_name, matching_cls = self.api_input_data_collection.get_name_cls_by_matching_column_name(
            feature_set.get_name_of_one_feature()
        )

        if matching_cls is None:
            raise ValueError(
                f"Feature group {format_feature_group_class(feature_group)} has no matching api data class for feature."
            )

        matching_cls_initialized = matching_cls(
            api_input_name, feature_set.get_name_of_one_feature(), feature_set.options
        )

        return matching_cls_initialized

    def add_artifact_to_feature_set(self, feature_group: type[FeatureGroup], feature_set: FeatureSet) -> None:
        if feature_group.artifact() is None:
            return

        feature_set.add_artifact_name()

    def add_single_filters_to_feature_set(self, feature_group: type[FeatureGroup], feature_set: FeatureSet) -> None:
        if self.global_filter is None:
            return

        if len(self.global_filter.collection.keys()) == 0:
            return

        feature_names = {feature.name for feature in feature_set.features}
        probed_union = self._probed_filters_for_set(feature_group, feature_set)

        # One representative per declared filter: enrichment variants of one declaration share
        # its uuid; the resolved column name stays in the key because renames change the predicate.
        representatives: dict[tuple[UUID, str], tuple[tuple[int, str, str], SingleFilter]] = {}
        for (
            filtered_feature_group,
            filtered_feature_name,
        ), single_filters in self.global_filter.collection.items():
            if filtered_feature_group != feature_group or filtered_feature_name not in feature_names:
                continue
            for single_filter in single_filters:
                key = (single_filter.uuid, str(single_filter.filter_feature.name))
                # A variant this run's features probed outranks stale ones a reused GlobalFilter kept.
                rank = (0 if single_filter in probed_union else 1, *_filter_options_sort_key(single_filter))
                current = representatives.get(key)
                if current is None or rank < current[0]:
                    representatives[key] = (rank, single_filter)

        # Fresh set; the elements remain the collection's live objects.
        relevant_filters = {single_filter for _, single_filter in representatives.values()}

        self._warn_on_unmatched_features(feature_group, feature_set, relevant_filters)
        feature_set.add_filters(relevant_filters)

    def _probed_filters_for_set(self, feature_group: type[FeatureGroup], feature_set: FeatureSet) -> set[SingleFilter]:
        """Union of the filters this set's features probed; unprobed features contribute nothing."""
        probed_union: set[SingleFilter] = set()
        if self.global_filter is None:
            return probed_union
        for feature in feature_set.features:
            probed = self.global_filter.probes.get((feature_group, feature.name, feature.uuid))
            if probed is not None:
                probed_union |= probed
        return probed_union

    def _warn_on_unmatched_features(
        self, feature_group: type[FeatureGroup], feature_set: FeatureSet, relevant_filters: set[SingleFilter]
    ) -> None:
        """Warn about features that declined a filter their feature set gets anyway."""
        if self.global_filter is None or not relevant_filters:
            return

        for feature in feature_set.features:
            probed = self.global_filter.probes.get((feature_group, feature.name, feature.uuid))
            # Filter and index features enter the collection without being probed.
            if probed is None:
                continue
            # Diff by declared-filter identity so a match under another enrichment still counts.
            probed_keys = {(f.uuid, str(f.filter_feature.name)) for f in probed}
            unmatched = sorted(
                {
                    str(f.filter_feature.name)
                    for f in relevant_filters
                    if (f.uuid, str(f.filter_feature.name)) not in probed_keys
                }
            )
            if not unmatched:
                continue
            key = (feature_group, str(feature.name), tuple(unmatched))
            first = key not in self.reported_unmatched
            self.reported_unmatched.add(key)
            logger.log(
                logging.WARNING if first else logging.DEBUG,
                "The filter feature(s) %s were not matched for feature '%s' of %s, but the filter still applies "
                "because the filter scope is the FeatureSet.",
                ", ".join(f"'{name}'" for name in unmatched),
                feature.name,
                format_feature_group_class(feature_group),
            )

    def get_parent_children_mapping(self, parent_to_children_mapping: dict[UUID, set[UUID]]) -> dict[UUID, set[UUID]]:
        inverted_dict: dict[UUID, set[UUID]] = {}
        for key, values in parent_to_children_mapping.items():
            for value in values:
                if value not in inverted_dict:
                    inverted_dict[value] = set()
                inverted_dict[value].add(key)

        return inverted_dict

    def invert_link_trekker(self, link_trekker: LinkTrekker) -> dict[UUID, set[LinkFrameworkTrekker]]:
        new_dict: dict[UUID, set[LinkFrameworkTrekker]] = defaultdict(set)

        for link, uuids in link_trekker.data.items():
            for uuid in uuids:
                new_dict[uuid].add(link)

        return new_dict

    def retrieve_links_which_must_be_calculated_before(
        self, features: set[Feature], child_links: dict[UUID, set[LinkFrameworkTrekker]]
    ) -> set[UUID]:
        new_set: set[UUID] = set()

        for feature in features:
            if feature.uuid in child_links:
                new_set.update({link[0].uuid for link in child_links[feature.uuid]})
        return new_set

    def retrieve_nodes_which_must_be_calculated_before(
        self, features: set[Feature], parent_to_children_mapping: dict[UUID, set[UUID]]
    ) -> set[UUID]:
        new_set: set[UUID] = set()
        for feature in features:
            if feature.uuid in parent_to_children_mapping:
                new_set.update(parent_to_children_mapping[feature.uuid])
        return new_set

    def group_features_by_compute_framework_and_options(self, features: set[Feature]) -> dict[int, set[Feature]]:
        """Group features by compute framework, options, and data type.

        Features with data_type=None are "lenient" - they join existing groups
        with matching base properties (options + compute_frameworks).
        This allows index columns (which have no explicit type) to stay grouped
        with typed features from the same FeatureGroup.
        """
        hash_collector: dict[int, set[Feature]] = defaultdict(set)
        none_typed_features: list[Feature] = []

        # Any key inherited by some feature in scope splits all features by value, so equal
        # effective config groups together and differing values stay isolated regardless of
        # provenance (consistent with the provenance-blind Feature dedup).
        split_keys: frozenset[str] = frozenset(
            key for feature in features for key in feature.options.inherited_context_keys
        )

        # First pass: group features with explicit data_type
        for feature in features:
            if feature.data_type is None:
                none_typed_features.append(feature)
            else:
                f_hash = feature.similarity_hash(split_keys)
                hash_collector[f_hash].add(feature)

        # Second pass: assign None-typed features to existing groups with matching base hash.
        # Precompute each group representative's base hash once (O(groups)) and look up in O(1),
        # preserving the first-match-in-insertion-order semantics of the original scan.
        base_hash_to_group: dict[int, int] = {}
        for existing_hash, group in hash_collector.items():
            representative_base_hash = next(iter(group)).base_similarity_hash(split_keys)
            base_hash_to_group.setdefault(representative_base_hash, existing_hash)

        for feature in none_typed_features:
            base_hash = feature.base_similarity_hash(split_keys)
            matched_group = base_hash_to_group.get(base_hash)
            if matched_group is not None:
                hash_collector[matched_group].add(feature)
            else:
                # No matching typed group found; create a new group for this None-typed feature
                # and register it so later None-typed features with the same base hash reuse it.
                hash_collector[base_hash].add(feature)
                base_hash_to_group[base_hash] = base_hash

        return hash_collector
