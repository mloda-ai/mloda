from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Generator, NamedTuple
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
from mloda.core.prepare.declared_sides import DeclaredSideSplit, split_by_declared_side
from mloda.core.prepare.joinstep_collection import JoinStepCollection
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
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
from mloda.core.abstract_plugins.components.feature_set import (
    FeatureSet,
    merge_input_feature_edges,
    option_split_paragraph,
)
from mloda.core.abstract_plugins.components.hashable_dict import _deep_hashable
from mloda.core.abstract_plugins.components.link import JoinType, Link
from collections import defaultdict
import logging

if TYPE_CHECKING:
    from mloda.core.prepare.resolve_graph import PlannedQueue


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


class _LineageMemo:
    """Per-add_tfs memo of same-framework lineages and the graph's reverse edges, the latter built on first use."""

    def __init__(self) -> None:
        self.lineages: dict[UUID, set[UUID]] = {}
        self._edge_parents: dict[UUID, set[UUID]] | None = None

    def edge_parents(self, graph: Graph) -> dict[UUID, set[UUID]]:
        if self._edge_parents is None:
            parents: dict[UUID, set[UUID]] = defaultdict(set)
            for parent, children in graph.adjacency_list.items():
                for child in children:
                    parents[child].add(parent)
            self._edge_parents = parents
        return self._edge_parents


class _SameFrameworkParent(NamedTuple):
    """Stand-in for a parent already in the step's framework, so unlinked ones are still detected."""

    from_feature_group: type[FeatureGroup]


class ExecutionPlan:
    def __init__(
        self,
        global_filter: GlobalFilter | None = None,
        api_input_data_collection: ApiInputDataCollection | None = None,
        resolved_input_feature_names: dict[UUID, frozenset[str] | None] | None = None,
        specialized_from: dict[UUID, tuple[str, ...]] | None = None,
    ) -> None:
        # Maps a step to itself so a dedup hit can recover the already-inserted canonical member.
        self.tfs_collection: dict[TransformFrameworkStep, TransformFrameworkStep] = {}
        self.joinstep_collection = JoinStepCollection()
        self.global_filter = global_filter
        self.api_input_data_collection = api_input_data_collection
        self.resolved_input_feature_names = resolved_input_feature_names
        self.specialized_from = specialized_from

        # Helper variable
        self.feature_set_collections: list[set[UUID]] = []

        # Report each divergence once, then at DEBUG.
        self.reported_unmatched: set[tuple[type[FeatureGroup], str, tuple[str, ...]]] = set()

        self.planned_records: list[ResolvedJoin] = []
        self.declined_orientations: list[LinkFrameworkTrekker] = []
        self.declared_frameworks: DeclaredFrameworks = {}
        self.resolved_join_plan = ResolvedJoinPlan((), ())
        self.join_signatures_at_build: frozenset[JoinSignature] = frozenset()

        # Per root feature_group class, the distinct option/context buckets run_feature_group split
        # it into: f_hash -> (a representative feature, the union of feature uuids in that bucket).
        # Non-root feature groups (their features have upstream ancestors) are never recorded here,
        # since a split with no ancestors can never be the actual cause of a missing-Links error.
        # Feeds the missing-Links error's option-split hint.
        self._option_split_buckets: dict[type[FeatureGroup], dict[Any, tuple[Feature, set[UUID]]]] = {}
        # Per root feature_group class, the union of inherited_context_keys used to hash its
        # buckets (the actual split_keys `group_features_by_compute_framework_and_options` used).
        self._option_split_keys: dict[type[FeatureGroup], frozenset[Any]] = {}

    def __iter__(self) -> Generator[TransformFrameworkStep | JoinStep | FeatureGroupStep, None, None]:
        yield from self.execution_plan

    def __len__(self) -> int:
        return len(self.execution_plan)

    def create_execution_plan(
        self,
        queue: "PlannedQueue",
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
        self._option_split_buckets = {}
        self._option_split_keys = {}

        child_links = self.invert_link_trekker(link_trekker)
        pre_execution_plan = self.add_feature_group_step(
            queue, graph.parent_to_children_mapping, child_links, graph.get_nodes()
        )
        fw_execution_plan = self.add_joinstep(pre_execution_plan, link_trekker, graph)

        # Run after add_joinstep, not inside add_feature_group_step: self.planned_records (which
        # records already-resolved Links) is only populated by add_joinstep's run_link calls, and a
        # split already bridged by a Link must not be blamed (see _stamp_option_split_hints).
        self._stamp_option_split_hints(fw_execution_plan, graph.parent_to_children_mapping)
        self._stamp_link_index_columns(fw_execution_plan, graph.parent_to_children_mapping)

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
        self._stamp_direct_sibling_readers()

        # Only read during add_joinstep above; ExecutionPlan gets deepcopy'd on every Engine.compute()
        # call, so this (potentially O(#features), UUID-keyed) dict and its live reference into
        # ResolveComputeFrameworks's own dict must not linger past the plan build that needs it, and
        # neither must the engine's resolved_input_feature_names map that run_feature_group read.
        self.declared_frameworks = {}
        self.resolved_input_feature_names = None
        self.specialized_from = None

    def _stamp_direct_sibling_readers(self) -> None:
        """Protect only isolated root/reader stars; joins and hops can redirect shared frames."""
        steps = {step.uuid: step for step in self.execution_plan}
        by_token: dict[UUID, set[UUID]] = {}
        for step in steps.values():
            for token in step.get_uuids() | step.required_uuids | {step.uuid}:
                by_token.setdefault(token, set()).add(step.uuid)
        remaining = set(steps)
        while remaining:
            pending = {next(iter(remaining))}
            component: set[UUID] = set()
            while pending:
                step_uuid = pending.pop()
                if step_uuid in component:
                    continue
                component.add(step_uuid)
                step = steps[step_uuid]
                for token in step.get_uuids() | step.required_uuids | {step.uuid}:
                    pending.update(by_token[token] - component)
            remaining.difference_update(component)
            members = [steps[uuid] for uuid in component]
            groups = [step for step in members if isinstance(step, FeatureGroupStep) and not step.tfs_ids]
            roots = [step for step in groups if not step.required_uuids]
            if len(groups) != len(members) or len(roots) != 1:
                continue
            root = roots[0]
            readers = [step for step in groups if step is not root]
            if any(
                step.compute_framework is not root.compute_framework
                or not step.required_uuids.issubset(root.get_uuids())
                for step in readers
            ):
                continue
            for step in readers:
                step.direct_sibling_readers = tuple(
                    (reader.uuid, reader.features.any_uuid, reader.features.declared_input_feature_names)
                    for reader in readers
                    if reader is not step
                    and reader.features.any_uuid is not None
                    and reader.features.declared_input_feature_names is not None
                )

    def add_feature_group_step(
        self,
        queue: "PlannedQueue",
        parent_to_children_mapping: dict[UUID, set[UUID]],
        child_links: dict[UUID, set[LinkFrameworkTrekker]],
        nodes: dict[UUID, NodeProperties] | None = None,
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
                feature_group_steps = self.run_feature_group(
                    element, parent_to_children_mapping, links_pre_calulated, nodes
                )
                for fg_step in feature_group_steps.values():
                    pre_execution_plan.append(fg_step)

            else:
                raise ValueError(f"Element {element} is not a valid element.")

        return pre_execution_plan

    def _stamp_option_split_hints(
        self,
        plan: list[JoinStep | FeatureGroupStep],
        parent_to_children_mapping: dict[UUID, set[UUID]],
    ) -> None:
        """Stamp an option-split hint on steps whose ancestors span two option/context buckets of a
        root feature group, computing the differing keys only from the buckets that consumer's own
        ancestors actually intersect, and only when those buckets are not already bridged by a
        resolved Link."""
        candidate_feature_groups = {
            feature_group: buckets for feature_group, buckets in self._option_split_buckets.items() if len(buckets) >= 2
        }
        if not candidate_feature_groups:
            self._option_split_buckets = {}
            self._option_split_keys = {}
            return

        ordered_feature_groups = sorted(candidate_feature_groups, key=lambda fg: fg.get_class_name())

        for step in plan:
            if not isinstance(step, FeatureGroupStep):
                continue

            ancestor_union: set[UUID] = set()
            for uuid in step.features.get_all_feature_ids():
                ancestor_union.update(parent_to_children_mapping.get(uuid, set()))

            for feature_group in ordered_feature_groups:
                buckets = candidate_feature_groups[feature_group]
                intersected_hashes = [f_hash for f_hash, (_, uuids) in buckets.items() if ancestor_union & uuids]
                if len(intersected_hashes) < 2:
                    continue

                unresolved_hashes = self._exclude_link_resolved_buckets(intersected_hashes, buckets)
                if len(unresolved_hashes) < 2:
                    continue

                representatives = [buckets[f_hash][0] for f_hash in unresolved_hashes]
                split_keys = self._option_split_keys.get(feature_group, frozenset())
                differing_keys = self._differing_option_keys(representatives, split_keys)
                if not differing_keys:
                    continue

                step.features.option_split_hint = (feature_group.get_class_name(), differing_keys)
                break

        self._option_split_buckets = {}
        self._option_split_keys = {}

    def _stamp_link_index_columns(
        self,
        plan: list[JoinStep | FeatureGroupStep],
        parent_to_children_mapping: dict[UUID, set[UUID]],
    ) -> None:
        """Stamp link-read columns on the join members and every step upstream of them, which may pass them through."""
        for join_step in plan:
            if not isinstance(join_step, JoinStep):
                continue
            members = join_step.destination_framework_uuids | join_step.source_framework_uuids
            index_columns = frozenset(join_step.link.left_index.index) | frozenset(join_step.link.right_index.index)
            asof_config = join_step.link.asof_config
            if asof_config is not None:
                index_columns = index_columns | frozenset({asof_config.left_time_column, asof_config.right_time_column})

            reached: set[UUID] = set(members)
            stack: list[UUID] = list(members)
            while stack:
                for parent in parent_to_children_mapping.get(stack.pop(), set()) - reached:
                    reached.add(parent)
                    stack.append(parent)

            for step in plan:
                if isinstance(step, FeatureGroupStep) and step.get_uuids() & reached:
                    step.features.link_index_columns = step.features.link_index_columns | index_columns

    def _exclude_link_resolved_buckets(
        self,
        hashes: list[Any],
        buckets: dict[Any, tuple[Feature, set[UUID]]],
    ) -> list[Any]:
        """Collapse the given buckets into connected components bridged by an already-resolved Link
        (``self.planned_records``); one representative hash survives per component. All buckets
        already resolved into a single component means the split caused no actual problem."""
        parent: dict[Any, Any] = {f_hash: f_hash for f_hash in hashes}

        def find(x: Any) -> Any:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: Any, b: Any) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        for record in self.planned_records:
            left_hashes = [f_hash for f_hash in hashes if buckets[f_hash][1] & record.left.uuids]
            right_hashes = [f_hash for f_hash in hashes if buckets[f_hash][1] & record.right.uuids]
            for left_hash in left_hashes:
                for right_hash in right_hashes:
                    union(left_hash, right_hash)

        components: dict[Any, Any] = {}
        for f_hash in hashes:
            components.setdefault(find(f_hash), f_hash)
        return list(components.values())

    def _differing_option_keys(self, representatives: list[Feature], split_keys: frozenset[Any]) -> frozenset[Any]:
        """Option/forwarded-context keys whose value differs across the given representative features, counting a key present in one and absent in another as differing."""
        if len(representatives) < 2:
            return frozenset()

        candidate_keys: set[Any] = set()
        for feature in representatives:
            candidate_keys.update(feature.options.group.keys())
            candidate_keys.update(key for key in split_keys if key in feature.options.context)

        _ABSENT = object()
        differing_keys: set[Any] = set()
        for key in candidate_keys:
            observed: set[Any] = set()
            for feature in representatives:
                if key in feature.options.group:
                    observed.add(_deep_hashable(feature.options.group[key]))
                elif key in split_keys and key in feature.options.context:
                    observed.add(_deep_hashable(feature.options.context[key]))
                else:
                    observed.add(_ABSENT)
            if len(observed) > 1:
                differing_keys.add(key)
        return frozenset(differing_keys)

    def add_joinstep(
        self,
        pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep],
        link_trekker: LinkTrekker,
        graph: Graph,
    ) -> list[JoinStep | FeatureGroupStep]:
        fw_execution_plan: list[JoinStep | FeatureGroupStep] = []

        for pex in pre_execution_plan:
            if isinstance(pex, tuple):
                fw_execution_plan.extend(self.run_link(pex, link_trekker, graph, pre_execution_plan))
            else:
                fw_execution_plan.append(pex)

        fw_execution_plan = self.handle_append_or_union_joinstep(fw_execution_plan)

        self.expand_link_tokens(fw_execution_plan, link_trekker)

        return fw_execution_plan

    def expand_link_tokens(
        self, fw_execution_plan: list[JoinStep | FeatureGroupStep], link_trekker: LinkTrekker
    ) -> None:
        """Replace each waited-on link uuid with its JoinSteps; a consumer step keeps only the joins recording it."""
        consumers_by_token: dict[UUID, frozenset[UUID]] = {r.token: r.consumers for r in self.planned_records}
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
                own = (
                    {t for t in produced if consumers_by_token.get(t, frozenset()) & step.get_uuids()}
                    if isinstance(step, FeatureGroupStep)
                    else set()
                )
                expanded.update(own or produced)

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
    def _shares_graph_ancestor(uuid_a: UUID, uuid_b: UUID, graph: Graph) -> bool:
        """Whether two parents' full transitive ancestor closures (already computed by
        Graph.set_all_parents_for_each_child) intersect."""
        closure_a = {uuid_a} | graph.parent_to_children_mapping.get(uuid_a, set())
        closure_b = {uuid_b} | graph.parent_to_children_mapping.get(uuid_b, set())
        return bool(closure_a & closure_b)

    @staticmethod
    def _parents_linked_by_join(
        uuid_a: UUID,
        uuid_b: UUID,
        join_steps: set[JoinStep],
        graph: Graph,
        memo: _LineageMemo | None = None,
    ) -> bool:
        """Whether two parents are linked, directly or transitively, via JoinSteps' genuine sides, in either order.
        Sides widen only through same-framework ancestors."""
        if uuid_a == uuid_b:
            return True

        adjacency: dict[UUID, set[UUID]] = defaultdict(set)
        for js in join_steps:
            for dest_uuid in js.destination_framework_uuids:
                adjacency[dest_uuid].update(js.source_framework_uuids)
            for src_uuid in js.source_framework_uuids:
                adjacency[src_uuid].update(js.destination_framework_uuids)
            for carrier in js.carriers:
                adjacency[carrier].update(js.destination_framework_uuids | js.source_framework_uuids)
                for side_uuid in js.destination_framework_uuids | js.source_framework_uuids:
                    adjacency[side_uuid].add(carrier)

        starts = ExecutionPlan._same_framework_lineage(uuid_a, graph, memo)
        targets = ExecutionPlan._same_framework_lineage(uuid_b, graph, memo)

        # Test join neighbours before dropping visited ones, so a target that is also a start still counts.
        visited = set(starts)
        frontier = set(starts)
        while frontier:
            reached = set().union(*(adjacency[node] for node in frontier))
            if reached & targets:
                return True
            frontier = reached - visited
            visited |= frontier
        return False

    @staticmethod
    def _same_framework_lineage(
        uuid: UUID,
        graph: Graph,
        memo: _LineageMemo | None = None,
    ) -> set[UUID]:
        """`uuid` plus its ancestors reachable without crossing a compute-framework change."""
        if memo is not None and uuid in memo.lineages:
            return memo.lineages[uuid]
        nodes = graph.get_nodes()

        def framework(node: UUID) -> Any:
            props = nodes.get(node)
            return props.feature.get_compute_framework() if props is not None else None

        ancestors = graph.parent_to_children_mapping
        edge_parents = (memo or _LineageMemo()).edge_parents(graph)

        def direct_parents(node: UUID) -> set[UUID]:
            above = ancestors.get(node, set())
            return edge_parents.get(node, set()) | (above - set().union(*(ancestors.get(a, set()) for a in above)))

        own_framework = framework(uuid)
        lineage = {uuid}
        stack = [uuid]
        while stack:
            for parent in direct_parents(stack.pop()) - lineage:
                if framework(parent) == own_framework:
                    lineage.add(parent)
                    stack.append(parent)
        if memo is not None:
            memo.lineages[uuid] = lineage
        return lineage

    def _variant_conflict(
        self, feature_a: Feature, feature_b: Feature
    ) -> tuple[Feature, Feature, frozenset[Any]] | None:
        """Two same-name features differing in data type or options, else None."""
        if feature_a.name != feature_b.name:
            return None
        split_keys = feature_a.options.inherited_context_keys | feature_b.options.inherited_context_keys
        option_keys = self._differing_option_keys([feature_a, feature_b], split_keys)
        if feature_a.data_type != feature_b.data_type or option_keys:
            return feature_a, feature_b, option_keys
        return None

    def _split_by_variant_conflicts(
        self,
        features: set[Feature],
        parent_to_children_mapping: dict[UUID, set[UUID]],
        nodes: dict[UUID, NodeProperties],
    ) -> list[set[Feature]]:
        """Bucket members so none holds two that read one name from differing variants of one source class."""
        producer_of: dict[UUID, int] = {}
        for index, uuids in enumerate(self.feature_set_collections):
            for uuid in uuids:
                producer_of[uuid] = index

        member_uuids = {f.uuid: f for f in features}

        def direct_parents(feature: Feature) -> list[UUID]:
            ancestors = parent_to_children_mapping.get(feature.uuid, set())
            direct = ancestors - {p for a in ancestors for p in parent_to_children_mapping.get(a, set())}
            return sorted(uuid for uuid in direct if uuid in producer_of)

        def same_group_ancestors(feature: Feature) -> list[Feature]:
            ancestors = parent_to_children_mapping.get(feature.uuid, set())
            return [member_uuids[u] for u in sorted(ancestors) if u in member_uuids]

        def effective_parents(feature: Feature) -> list[UUID]:
            found = set(direct_parents(feature))
            for ancestor in same_group_ancestors(feature):
                found.update(direct_parents(ancestor))
            return sorted(found)

        read_names: set[str] = set()

        def conflicts(parents_a: list[UUID], parents_b: list[UUID]) -> bool:
            found = False
            for a in parents_a:
                for b in parents_b:
                    if (
                        producer_of[a] == producer_of[b]
                        or nodes[a].feature_group_class is not nodes[b].feature_group_class
                    ):
                        continue
                    if self._variant_conflict(nodes[a].feature, nodes[b].feature) is not None:
                        read_names.add(str(nodes[a].feature.name))
                        found = True
            return found

        buckets: list[list[tuple[Feature, list[UUID]]]] = []
        bucket_of: dict[UUID, int] = {}
        ordered = sorted(features, key=lambda f: (str(f.name), str(f.data_type), str(f.options), str(f.uuid)))
        for feature in ordered:
            parents = effective_parents(feature)
            for index, bucket in enumerate(buckets):
                if not any(conflicts(parents, other) for _, other in bucket):
                    bucket.append((feature, parents))
                    bucket_of[feature.uuid] = index
                    break
            else:
                bucket_of[feature.uuid] = len(buckets)
                buckets.append([(feature, parents)])
        if len(buckets) > 1:
            blocked = sorted(
                str(f.name)
                for f in features
                if not parent_to_children_mapping.get(f.uuid)
                or any(bucket_of[a.uuid] != bucket_of[f.uuid] for a in same_group_ancestors(f))
            )
            if blocked:
                group = format_feature_group_class(nodes[ordered[0].uuid].feature_group_class)
                raise ValueError(
                    f"'{group}' reads input {sorted(read_names)} in differing variants (data type or options), "
                    f"so its step cannot be split: members {blocked} serve every feature. "
                    "Align the options of the differing requests or request the features in separate runs."
                )
        return [{feature for feature, _ in bucket} for bucket in buckets]

    @staticmethod
    def _conflicting_variants_error(
        ep: FeatureGroupStep,
        source: type[FeatureGroup],
        feature_a: Feature,
        feature_b: Feature,
        option_keys: frozenset[Any],
    ) -> str:
        """One consumer reads one name from two unbound variants of the same source feature group."""
        differences = []
        if option_keys:
            differences.append(f"options {sorted(str(key) for key in option_keys)}")
        if feature_a.data_type != feature_b.data_type:
            differences.append(f"data types {sorted([str(feature_a.data_type), str(feature_b.data_type)])}")
        return (
            f"'{format_feature_group_class(ep.feature_group)}' reads feature '{feature_a.name}' from two steps of "
            f"'{format_feature_group_class(source)}' that differ in {' and '.join(differences)}. "
            "Only one of the two steps can be bound, so one variant's values would be read for both. "
            "Request the input once with the same options and data type."
        )

    @staticmethod
    def _conflicting_transform_hops_error(
        ep: FeatureGroupStep,
        first_hop: TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent,
        second_hop: TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent,
        further_hops: list[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent],
    ) -> str:
        """A FeatureGroupStep can only bind one incoming source; two distinct, unlinked ones is a
        missing-Link configuration problem, not a bug."""
        split_text = option_split_paragraph(ep.features.option_split_hint)
        feature_name = format_feature_group_class(ep.feature_group)

        hops = [first_hop, second_hop, *further_hops]
        step_counts: dict[type[FeatureGroup], int] = {}
        for hop in hops:
            step_counts[hop.from_feature_group] = step_counts.get(hop.from_feature_group, 0) + 1
        source_texts = [
            f"'{format_feature_group_class(group)}'" + (f" ({count} separate steps)" if count > 1 else "")
            for group, count in step_counts.items()
        ]
        sources_text = " and ".join(source_texts)

        hint = ep.features.option_split_hint
        hinted_class_name = hint[0] if hint is not None else None
        repeated_paragraph = ""
        for group, count in step_counts.items():
            if count > 1 and group.get_class_name() != hinted_class_name:
                repeated_paragraph += (
                    f"\n'{format_feature_group_class(group)}' ran as separate steps because its requests differ in "
                    "data type, compute framework, or source, or read differing variants of one input. "
                    "Align those requests so one step serves them all.\n"
                )

        example_groups = list(step_counts)
        first_class_name = example_groups[0].get_class_name()
        second_class_name = example_groups[1].get_class_name() if len(example_groups) > 1 else first_class_name
        link_options = ""
        if first_class_name != second_class_name:
            link_options = f"""
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
"""
        elif first_class_name == hinted_class_name:
            link_options = f"""
Option 1: Explicit JoinSpec (works with any feature group):
    from mloda.user import Link, JoinSpec

    links = {{
        Link.inner(
            JoinSpec({first_class_name}, "shared_column"),
            JoinSpec({second_class_name}, "shared_column"),
            left_discriminator={{"option_key": "left_value"}},
            right_discriminator={{"option_key": "right_value"}},
        )
    }}
"""

        return f"""
Feature group '{feature_name}' depends on parents from {len(hops)} unlinked sources (missing Links): {sources_text}.
{split_text}{repeated_paragraph}
When a feature depends on multiple input features from different sources, you must provide explicit
Links to specify how to merge them. Without Links, the framework cannot determine how to combine the
data, and only one of the sources would ever be read. A parent counts as reaching a join side only
through ancestors on its own compute framework, not through a compute-framework hop.
{link_options}
Available join types:
- Link.inner(left, right)    - Keep only matching rows from both sides
- Link.left(left, right)     - Keep all rows from left, matching from right
- Link.right(left, right)    - Keep all rows from right, matching from left
- Link.outer(left, right)    - Keep all rows from both sides
- Link.inner_on(left, right) - Shorthand using index_columns() definitions
""".strip()

    @staticmethod
    def _step_waits_for(
        start: UUID, goal: UUID, steps_by_uuid: Mapping[UUID, Step], producer_of: Mapping[UUID, UUID]
    ) -> bool:
        stack, visited = [start], {start}
        while stack:
            step = steps_by_uuid[stack.pop()]
            if step.uuid == goal:
                return True
            for token in step.get_wait_uuids():
                owner = producer_of.get(token)
                if owner is not None and owner not in visited:
                    visited.add(owner)
                    stack.append(owner)
        return False

    def _order_hop_after_same_framework_parent(
        self,
        consumer: FeatureGroupStep,
        hop: TransformFrameworkStep,
        parent: UUID,
        plan: Sequence[Step],
        steps_by_uuid: Mapping[UUID, Step],
        owning_step_of: dict[UUID, UUID],
        graph: Graph,
        memo: _LineageMemo,
        join_steps: set[JoinStep],
    ) -> str | tuple[TransformFrameworkStep, str] | None:
        """Order the upstream hop copying `parent`'s frame after it; an error text if none can, or a mirror hop to redirect to."""
        nodes = graph.get_nodes()
        consumer_name = format_feature_group_class(consumer.feature_group)
        parent_name = format_feature_group_class(nodes[parent].feature_group_class)
        hop_name = format_feature_group_class(hop.from_feature_group)

        def error(cause: str) -> str:
            return (
                f"'{consumer_name}' reads '{parent_name}' on {consumer.compute_framework.get_class_name()} and "
                f"'{hop_name}' through a hop from {hop.from_framework.get_class_name()}, but {cause} "
                "Compute the inputs on one compute framework."
            )

        parent_owners = {owning_step_of.get(u, u) for u in self._same_framework_lineage(parent, graph, memo)}
        targets: list[TransformFrameworkStep] = []
        seen: set[UUID] = set()
        frontier = [hop]
        while frontier:
            current = frontier.pop()
            source = steps_by_uuid.get(current.source_step_uuid) if current.source_step_uuid else None
            if not isinstance(source, FeatureGroupStep):
                continue
            for hop_uuid in source.tfs_ids:
                upstream = steps_by_uuid.get(hop_uuid)
                if not isinstance(upstream, TransformFrameworkStep) or upstream.link_id is not None:
                    continue
                if upstream.uuid in seen:
                    continue
                seen.add(upstream.uuid)
                lineage_owners = {
                    owning_step_of.get(u, u)
                    for req in upstream.required_uuids
                    for u in self._same_framework_lineage(req, graph, memo)
                }
                if upstream.from_framework == consumer.compute_framework and lineage_owners & parent_owners:
                    targets.append(upstream)
                else:
                    frontier.append(upstream)

        joined = any(js.reads_join_frame(consumer.required_uuids) for js in join_steps) or any(
            self._parents_linked_by_join(parent, source, join_steps, graph, memo) for source in hop.required_uuids
        )
        if not targets and joined:
            return None
        producer_of = {token: step.uuid for step in plan for token in step.get_uuids()}

        if not targets:
            hop_owners = {
                owning_step_of.get(u, u)
                for req in hop.required_uuids
                for u in self._same_framework_lineage(req, graph, memo)
            }
            mirrors: dict[UUID, TransformFrameworkStep] = {}
            for owner in parent_owners:
                owner_step = steps_by_uuid.get(owner)
                if not isinstance(owner_step, FeatureGroupStep):
                    continue
                for hop_uuid in owner_step.tfs_ids:
                    cand = steps_by_uuid.get(hop_uuid)
                    if (
                        isinstance(cand, TransformFrameworkStep)
                        and cand is not hop
                        and cand.link_id is None
                        and cand.from_framework == hop.from_framework
                        and cand.to_framework == consumer.compute_framework
                        and hop_owners
                        & {
                            owning_step_of.get(u, u)
                            for req in cand.required_uuids
                            for u in self._same_framework_lineage(req, graph, memo)
                        }
                    ):
                        mirrors[cand.uuid] = cand
            if len(mirrors) != 1:
                return error(
                    f"'{parent_name}' and the hopped frame live in two separate "
                    f"{consumer.compute_framework.get_class_name()} frames that nothing merges."
                )
            mirror = next(iter(mirrors.values()))
            dropped_starts = {owning_step_of.get(u, u) for u in hop.required_uuids}
            if hop.source_step_uuid is not None:
                dropped_starts.add(hop.source_step_uuid)
            if any(self._step_waits_for(start, mirror.uuid, steps_by_uuid, producer_of) for start in dropped_starts):
                return error(f"'{parent_name}' itself depends on that hop, so it cannot be ordered before it.")
            return mirror, error(
                f"'{parent_name}' and the hopped frame live in two separate "
                f"{consumer.compute_framework.get_class_name()} frames that nothing merges."
            )

        def waits_for(start: UUID, goal: UUID) -> bool:
            return self._step_waits_for(start, goal, steps_by_uuid, producer_of)

        parent_step = owning_step_of.get(parent, parent)
        for target in targets:
            if waits_for(parent_step, target.uuid):
                return error(f"'{parent_name}' itself depends on that hop, so it cannot be ordered before it.")
        for target in targets:
            target.order_after_uuids.add(parent)
        return None

    @staticmethod
    def _read_through_own_input_error(
        ep: FeatureGroupStep, declared: type[FeatureGroup], via: type[FeatureGroup]
    ) -> str:
        """A consumer reads a feature that its same-framework input already consumed through a hop."""
        consumer = format_feature_group_class(ep.feature_group)
        return (
            f"'{consumer}' reads '{format_feature_group_class(declared)}' and also '{format_feature_group_class(via)}', "
            f"which already consumed it through a compute-framework hop. The second read would need a frame that "
            "nothing merges. Read only the feature derived from it, or compute both on one compute framework."
        )

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
            step.uuid: {producer_of[token] for token in step.get_wait_uuids() if token in producer_of} for step in steps
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
            link_id=ep.uuid,
            source_framework_uuids=ep.source_framework_uuids,
        )

    @staticmethod
    def _destination_hop(
        ep: JoinStep, graph: Graph, owning_step_of: Mapping[UUID, UUID], private_copy: bool
    ) -> TransformFrameworkStep:
        """The hop that copies the carriers' (or the shared destination side's) frame for the join to merge into."""
        nodes = graph.get_nodes()
        members = ep.carriers or ep.destination_framework_uuids
        carrier = min(members)
        if len({owning_step_of.get(c, c) for c in members}) > 1:
            names = sorted(format_feature_group_class(nodes[c].feature_group_class) for c in members)
            raise ValueError(
                f"The consumers of {ep.link} read one link side through several steps on another compute framework "
                f"({', '.join(names)}), which the join cannot read through one hop. "
                "Read the side through one feature or compute them on one compute framework."
            )
        destination_group = ep.link.right_feature_group if ep.swap_merge_sides else ep.link.left_feature_group
        return TransformFrameworkStep(
            from_framework=nodes[carrier].feature.get_compute_framework(),
            to_framework=ep.destination_framework,
            required_uuids=set(members),
            from_feature_group=nodes[carrier].feature_group_class,
            to_feature_group=destination_group,
            source_step_uuid=owning_step_of.get(carrier, carrier),
            private_copy=private_copy,
        )

    @staticmethod
    def _redirect_consumers_to_hop(
        ep: JoinStep,
        hop: TransformFrameworkStep,
        execution_plan: Sequence[JoinStep | FeatureGroupStep],
        only_consumers: frozenset[UUID] | None = None,
    ) -> None:
        """Point the join's consumers on its destination framework at the hop's frame."""
        readers: set[UUID] = set()
        for inner_ep in execution_plan:
            if (
                isinstance(inner_ep, FeatureGroupStep)
                and inner_ep.compute_framework == ep.destination_framework
                and ep.reads_join_frame(inner_ep.required_uuids)
                and (only_consumers is None or inner_ep.get_uuids() & only_consumers)
            ):
                inner_ep.tfs_ids = {hop.uuid}
                inner_ep.features.any_uuid = hop.uuid
                readers |= inner_ep.get_uuids()
        hop.copy_readers = frozenset(readers)

    @staticmethod
    def _same_framework_mids(ep: JoinStep, consumers: frozenset[UUID], graph: Graph) -> set[UUID]:
        """Consumer parents on the destination framework that write into a link side's frame."""
        nodes = graph.get_nodes()
        side_members = ep.destination_framework_uuids | ep.source_framework_uuids
        return {
            parent
            for consumer in consumers
            for parent in graph.parent_to_children_mapping[consumer]
            if parent not in side_members and nodes[parent].feature.get_compute_framework() == ep.destination_framework
        }

    def _carrier_scan(
        self,
        children_uuids: set[UUID],
        split: DeclaredSideSplit,
        graph: Graph,
        frameworks: set[type[ComputeFramework]],
        side_frameworks: set[type[ComputeFramework]] | None = None,
    ) -> tuple[frozenset[UUID], set[bool], set[UUID]]:
        """Non-raising scan: carriers, the sides they descend from, and the children reading through them."""
        nodes = graph.get_nodes()
        side_members = split.left_uuids_any_distance | split.right_uuids_any_distance
        carriers: set[UUID] = set()
        sides: set[bool] = set()
        carried_children: set[UUID] = set()
        for child in children_uuids:
            parents = graph.parent_to_children_mapping[child]
            for parent in parents - self.get_parent_parents(parents, graph):
                if parent in side_members or nodes[parent].feature.get_compute_framework() in frameworks:
                    continue
                above = graph.parent_to_children_mapping.get(parent, set())
                from_left = bool(above & split.left_uuids_any_distance)
                from_right = bool(above & split.right_uuids_any_distance)
                # Same-framework join: one-sided mids over other-framework members arrive via another join of the link.
                if (
                    side_frameworks is not None
                    and not (from_left and from_right)
                    and not any(
                        nodes[a].feature.get_compute_framework() in side_frameworks for a in above & side_members
                    )
                ):
                    continue
                if from_left or from_right:
                    carriers.add(parent)
                    carried_children.add(child)
                    sides.add(from_left)
                    if from_left and from_right:
                        sides.add(False)
        return frozenset(carriers), sides, carried_children

    def _join_carriers(
        self,
        children_uuids: set[UUID],
        split: DeclaredSideSplit,
        frameworks: set[type[ComputeFramework]],
        graph: Graph,
        side_frameworks: set[type[ComputeFramework]] | None = None,
    ) -> tuple[frozenset[UUID], bool]:
        """Consumer parents on another framework that descend from a link side, and whether that side is the left."""
        carriers, sides, _ = self._carrier_scan(children_uuids, split, graph, frameworks, side_frameworks)
        if len(sides) > 1:
            nodes = graph.get_nodes()
            names = sorted(format_feature_group_class(nodes[c].feature_group_class) for c in carriers)
            raise ValueError(
                f"The consumers read both sides of a link through features on another compute framework "
                f"({', '.join(names)}), which the join cannot read. Compute them on one compute framework."
            )
        return carriers, next(iter(sides), True)

    def add_tfs(
        self, execution_plan: list[JoinStep | FeatureGroupStep], graph: Graph
    ) -> list[TransformFrameworkStep | JoinStep | FeatureGroupStep]:
        new_execution_plan: list[TransformFrameworkStep | JoinStep | FeatureGroupStep] = []

        left_join_frameworks: set[JoinStep] = {ep for ep in execution_plan if isinstance(ep, JoinStep)}
        need_to_upload_collector: set[UUID] = set()

        # Which JoinStep uuids a canonical cross-framework join hop serves; a join hop's owed-token
        # route tokens are these, not its own uuid (see the owed-token block below).
        hop_serves: dict[UUID, set[UUID]] = defaultdict(set)

        # Features produced together by one FeatureGroupStep live on the same physical source cfw
        # instance, so a hop should key on the owning step, not each member feature's own uuid.
        owning_step_of: dict[UUID, UUID] = {
            feature_uuid: ep.uuid
            for ep in execution_plan
            if isinstance(ep, FeatureGroupStep)
            for feature_uuid in ep.get_uuids()
        }

        root_steps: set[UUID] = {
            ep.uuid
            for ep in execution_plan
            if isinstance(ep, FeatureGroupStep)
            and not any(graph.parent_to_children_mapping.get(uuid) for uuid in ep.get_uuids())
        }

        def root_steps_of(parent: UUID) -> set[UUID]:
            closure = {parent} | graph.parent_to_children_mapping.get(parent, set())
            return {owning_step_of.get(uuid, uuid) for uuid in closure} & root_steps

        memo = _LineageMemo()
        # (consumer step, its hop, same-framework parent) triples, resolved once the plan list is complete.
        hop_same_framework_pairs: list[tuple[FeatureGroupStep, TransformFrameworkStep, UUID]] = []
        for ep in execution_plan:
            if isinstance(ep, JoinStep):
                if ep.destination_framework != ep.source_framework:
                    new_tfs = self.fill_tfs_by_joinstep(ep)

                    # link_id is the join token, so each join owns its hop and re-finds its hopped cfw by it.
                    self.tfs_collection[new_tfs] = new_tfs
                    new_execution_plan.append(new_tfs)
                    ep.required_uuids.add(new_tfs.uuid)
                    hop_serves[new_tfs.uuid].add(ep.uuid)

                    if ep.shared_destination or ep.carriers:
                        destination_hop = self._destination_hop(
                            ep, graph, owning_step_of, not ep.carriers or ep.split_consumers is not None
                        )
                        new_execution_plan.append(destination_hop)
                        ep.required_uuids.add(destination_hop.uuid)
                        ep.destination_hop_uuid = destination_hop.uuid
                        need_to_upload_collector.update(ep.carriers or ep.destination_framework_uuids)
                        self._redirect_consumers_to_hop(ep, destination_hop, execution_plan, ep.split_consumers)

                    need_to_upload_collector.update(ep.source_framework_uuids)

                    # We are updating the required uuids after the tfs is added as this makes sure, that the TFS can run in parallel before the join.
                    ep.required_uuids.update(self.joinstep_collection.get_required_join_uuids(ep))
                else:
                    # We need to do two things:
                    # 1) source feature group of the join step needs to know of the link, so that the cfw can be used by the joinstep
                    # 2) The child feature using this join needs to know which cfw to use. We use the tfs vehicle for this.
                    store_val = None

                    if ep.carriers or ep.shared_destination:
                        destination_hop = self._destination_hop(
                            ep, graph, owning_step_of, not ep.carriers or ep.split_consumers is not None
                        )
                        new_execution_plan.append(destination_hop)
                        ep.required_uuids.add(destination_hop.uuid)
                        ep.destination_hop_uuid = destination_hop.uuid
                        need_to_upload_collector.update(ep.carriers)
                        if not ep.carriers:
                            mids = self._same_framework_mids(ep, ep.split_consumers or frozenset(), graph)
                            destination_hop.order_after_uuids |= mids
                            need_to_upload_collector.update(mids | ep.destination_framework_uuids)
                        self._redirect_consumers_to_hop(ep, destination_hop, execution_plan, ep.split_consumers)

                    for inner_ep in execution_plan:
                        if isinstance(inner_ep, FeatureGroupStep):
                            # 1) We do 1 here:
                            for uuid in inner_ep.get_uuids():
                                if uuid in ep.source_framework_uuids:
                                    # add the JoinStep token to the children_if_root of the source feature group
                                    inner_ep.add_value_to_children_if_root(ep.uuid)

                                    # add to upload as this source feature group gets accessed in mp by other process
                                    need_to_upload_collector.update(ep.source_framework_uuids)
                                    break

                                if uuid in ep.destination_framework_uuids:
                                    # remember the destination feature group's uuid for the JoinStep token

                                    store_val = uuid

                            if store_val is None:
                                continue

                            # Check if any element of ep.destination_framework_uuids is in inner_ep.required_uuids
                            # same for source framework
                            if (
                                ep.destination_hop_uuid is None
                                and any(elem in inner_ep.required_uuids for elem in ep.destination_framework_uuids)
                                and any(elem in inner_ep.required_uuids for elem in ep.source_framework_uuids)
                            ):
                                if ep.link.jointype in (JoinType.APPEND, JoinType.UNION):
                                    self.set_store_value_to_left_most_index_and_update_feature_group(
                                        inner_ep, store_val
                                    )
                                else:
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
                    direct_parents = member_parents - self.get_parent_parents(member_parents, graph)
                    parents |= direct_parents

                names_by_step: dict[UUID, set[str]] = defaultdict(set)
                for parent in parents:
                    names_by_step[owning_step_of.get(parent, parent)].add(str(graph.get_nodes()[parent].feature.name))
                consumed_names_by_step = {step: frozenset(names) for step, names in names_by_step.items()}

                # Explicit hops and join-served parents (delivered pre-merged by a JoinStep, no hop built)
                # both compete for this step's one binding, so both get grouped by the same linkage test
                # below. Order-independent: collected here, grouped once after the loop.
                bound_entries: list[tuple[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent, UUID]] = []
                join_served_entries: list[tuple[type[FeatureGroup], UUID]] = []
                same_framework_entries: list[tuple[type[FeatureGroup], UUID]] = []
                seen_hop_uuids: set[UUID] = set()

                for parent in parents:
                    parent_node_property = graph.get_nodes()[parent]
                    matching_join_steps = [
                        js
                        for js in left_join_frameworks
                        if js.matched(ep.compute_framework, parent_node_property.feature.uuid, ep.required_uuids)
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

                        # Records every parent the hop covers. On its own this doesn't change the
                        # scheduling gate, since canonical_tfs is still the one owning step here;
                        # but the subclass-cluster widening further below can still merge this
                        # hop's required_uuids with a SIBLING hop's, so the set can end up spanning
                        # more than one owning step/framework after all.
                        canonical_tfs.required_uuids.add(parent)
                        ep.required_uuids.add(canonical_tfs.uuid)

                        # Record the surviving hop's uuid so the step resolves its compute framework from it.
                        ep.tfs_ids.add(canonical_tfs.uuid)

                        need_to_upload_collector.add(parent)
                    else:
                        same_framework_entries.append((parent_node_property.feature_group_class, parent))

                def _conflicting_variants(
                    parent_a: UUID, parent_b: UUID
                ) -> tuple[Feature, Feature, frozenset[Any]] | None:
                    """Two same-name parents of the step differing in data type or options."""
                    nodes = graph.get_nodes()
                    return self._variant_conflict(nodes[parent_a].feature, nodes[parent_b].feature)

                # Group entries by transitive linkage: same feature-group class (unless split across
                # unrelated root steps), one entry's own class a subclass (or superclass) of the other's
                # (catches a case-override hop whose parent lost the JoinStep's own uuid to a same-role sibling,
                # see `_case_override_beats_nearer_wrong_framework_left`, without also bridging two entries that
                # merely share an unrelated common ancestor via some third join's declared side), or
                # `_parents_linked_by_join`. A subclass pairing must additionally share genuine graph ancestry
                # unless it is join-served, so two plain hops that merely subclass one another over otherwise
                # unrelated roots are not merged.
                def _entries_linked(
                    entry_a: tuple[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent, UUID],
                    entry_b: tuple[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent, UUID],
                ) -> bool:
                    hop_a, parent_a = entry_a
                    hop_b, parent_b = entry_b
                    if hop_a.from_feature_group is hop_b.from_feature_group:
                        # Steps match by column name only, a heuristic: same names means either step
                        # supplies every column, except a conflicting variant pair read by one consumer feature.
                        if root_steps_of(parent_a) & root_steps_of(parent_b):
                            return True
                        step_a = owning_step_of.get(parent_a, parent_a)
                        step_b = owning_step_of.get(parent_b, parent_b)
                        if (
                            consumed_names_by_step[step_a] == consumed_names_by_step[step_b]
                            and _conflicting_variants(parent_a, parent_b) is None
                        ):
                            return True
                    if isinstance(hop_a, _SameFrameworkParent) or isinstance(hop_b, _SameFrameworkParent):
                        closure_a = {parent_a} | graph.parent_to_children_mapping.get(parent_a, set())
                        closure_b = {parent_b} | graph.parent_to_children_mapping.get(parent_b, set())
                        owners_a = {owning_step_of.get(u, u) for u in closure_a}
                        if any(owning_step_of.get(u, u) in owners_a for u in closure_b):
                            return True
                    if issubclass(hop_a.from_feature_group, hop_b.from_feature_group) or issubclass(
                        hop_b.from_feature_group, hop_a.from_feature_group
                    ):
                        join_adjacent = isinstance(hop_a, _JoinServedParent) or isinstance(hop_b, _JoinServedParent)
                        if join_adjacent or self._shares_graph_ancestor(parent_a, parent_b, graph):
                            return True
                    return self._parents_linked_by_join(parent_a, parent_b, left_join_frameworks, graph, memo)

                def _add_to_groups(
                    groups: list[list[tuple[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent, UUID]]],
                    entry: tuple[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent, UUID],
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

                hop_groups: list[
                    list[tuple[TransformFrameworkStep | _JoinServedParent | _SameFrameworkParent, UUID]]
                ] = []
                for entry in bound_entries:
                    _add_to_groups(hop_groups, entry)

                # Join-served parents compete for the same binding, so merge them into the same groups too.
                for served_feature_group, served_parent in join_served_entries:
                    _add_to_groups(hop_groups, (_JoinServedParent(served_feature_group), served_parent))

                for same_feature_group, same_parent in same_framework_entries:
                    _add_to_groups(hop_groups, (_SameFrameworkParent(same_feature_group), same_parent))

                edge_parents = memo.edge_parents(graph)
                nodes = graph.get_nodes()
                cycle_errors = [
                    self._read_through_own_input_error(
                        ep, nodes[declared].feature_group_class, nodes[via].feature_group_class
                    )
                    for member_uuid in ep.get_uuids()
                    for via in edge_parents.get(member_uuid, set())
                    if nodes[via].feature.get_compute_framework() == ep.compute_framework
                    and any(not edge_parents.get(u) for u in self._same_framework_lineage(via, graph, memo) - {via})
                    for declared in edge_parents.get(member_uuid, set()) & edge_parents.get(via, set())
                    if nodes[declared].feature.get_compute_framework() != ep.compute_framework
                    and not any(
                        declared in js.carriers and js.matched(ep.compute_framework, declared, ep.required_uuids)
                        for js in left_join_frameworks
                    )
                ]
                if cycle_errors:
                    raise ValueError(min(cycle_errors))

                for group in hop_groups:
                    for same_hop, same_parent_uuid in group:
                        if isinstance(same_hop, _SameFrameworkParent):
                            hop_same_framework_pairs.extend(
                                (ep, hop, same_parent_uuid)
                                for hop, _hop_parent in group
                                if isinstance(hop, TransformFrameworkStep)
                            )

                # Two distinct hops linked only because one's from_feature_group subclasses the other's
                # read the same physical source cfw instance at runtime, but each
                # only ever waited on its own parent; whichever a later runtime lookup happens to pick
                # may transform a stale snapshot, taken before the sibling's parent landed in it. Widen
                # each to wait on the other's parent too, scoped to the subclass pairing specifically:
                # the same-class and join-bridged linkages `_entries_linked` also groups by already have
                # their own, narrower reasons to keep separate required_uuids (see the two multi-member
                # tests in test_add_tfs_multi_member_parents.py). This widening does not re-check
                # `_shares_graph_ancestor` for a pair only transitively grouped via a third member.
                def _subclass_only_linked(hop_a: TransformFrameworkStep, hop_b: TransformFrameworkStep) -> bool:
                    if hop_a.from_feature_group is hop_b.from_feature_group:
                        return False
                    return issubclass(hop_a.from_feature_group, hop_b.from_feature_group) or issubclass(
                        hop_b.from_feature_group, hop_a.from_feature_group
                    )

                for group in hop_groups:
                    tfs_members = [hop for hop, _member_parent in group if isinstance(hop, TransformFrameworkStep)]
                    subclass_clusters: list[list[TransformFrameworkStep]] = []
                    for tfs in tfs_members:
                        linked_clusters = [
                            cluster
                            for cluster in subclass_clusters
                            if any(_subclass_only_linked(tfs, member) for member in cluster)
                        ]
                        if linked_clusters:
                            linked_clusters[0].append(tfs)
                            for other_cluster in linked_clusters[1:]:
                                linked_clusters[0].extend(other_cluster)
                                subclass_clusters.remove(other_cluster)
                        else:
                            subclass_clusters.append([tfs])

                    for cluster in subclass_clusters:
                        if len(cluster) > 1:
                            shared_required_uuids: set[UUID] = set().union(*(tfs.required_uuids for tfs in cluster))
                            for tfs in cluster:
                                # A step must never wait for a token it produces itself.
                                tfs.required_uuids = shared_required_uuids - tfs.get_uuids()

                # Hops of differing classes bridged by a join read one physical frame: each waits on every
                # sibling's parent, else a hop may snapshot the frame before a sibling's parent landed in it.
                for group in hop_groups:
                    bridged: list[tuple[TransformFrameworkStep, UUID]] = [
                        (bh, bp) for bh, bp in group if isinstance(bh, TransformFrameworkStep)
                    ]
                    snapshot = {id(bh): frozenset(bh.required_uuids) for bh, _ in bridged}
                    for bh, bp in bridged:
                        for other, other_parent in bridged:
                            if (
                                other is not bh
                                and other.from_feature_group is not bh.from_feature_group
                                and other.from_framework == bh.from_framework
                                and self._parents_linked_by_join(bp, other_parent, left_join_frameworks, graph, memo)
                            ):
                                bh.required_uuids |= snapshot[id(other)] - bh.get_uuids()

                if len(hop_groups) > 1:
                    variant_errors: list[tuple[str, str, str]] = []
                    for index, group_a in enumerate(hop_groups):
                        for group_b in hop_groups[index + 1 :]:
                            for hop_a, parent_a in group_a:
                                for hop_b, parent_b in group_b:
                                    if hop_a.from_feature_group is not hop_b.from_feature_group:
                                        continue
                                    conflict = _conflicting_variants(parent_a, parent_b)
                                    if conflict is not None:
                                        message = self._conflicting_variants_error(
                                            ep, hop_a.from_feature_group, conflict[0], conflict[1], conflict[2]
                                        )
                                        variant_errors.append(
                                            (hop_a.from_feature_group.get_class_name(), str(conflict[0].name), message)
                                        )
                    if variant_errors:
                        raise ValueError(min(variant_errors)[2])
                    # Set iteration order of parents varies with the hash seed, so name groups in a stable order.
                    reps = sorted(
                        (group[0][0] for group in hop_groups),
                        key=lambda hop: (hop.from_feature_group.get_class_name(), hop.from_feature_group.__module__),
                    )
                    raise ValueError(self._conflicting_transform_hops_error(ep, reps[0], reps[1], reps[2:]))

            else:
                raise ValueError(f"Element {ep} is not a valid element.")
            new_execution_plan.append(ep)

        pair_outcomes: list[str] = []
        tfs_by_uuid = {step.uuid: step for step in new_execution_plan if isinstance(step, TransformFrameworkStep)}
        redirects: dict[UUID, tuple[FeatureGroupStep, dict[UUID, tuple[TransformFrameworkStep, str]]]] = {}
        steps_by_uuid = {step.uuid: step for step in new_execution_plan}
        for pair_consumer, pair_hop, pair_parent in hop_same_framework_pairs:
            pair_outcome = self._order_hop_after_same_framework_parent(
                pair_consumer,
                pair_hop,
                pair_parent,
                new_execution_plan,
                steps_by_uuid,
                owning_step_of,
                graph,
                memo,
                join_steps=left_join_frameworks,
            )
            if isinstance(pair_outcome, tuple):
                by_hop_entry = redirects.setdefault(pair_consumer.uuid, (pair_consumer, {}))[1]
                previous = by_hop_entry.get(pair_hop.uuid)
                if previous is not None:
                    pair_outcome = (pair_outcome[0], min(previous[1], pair_outcome[1]))
                by_hop_entry[pair_hop.uuid] = pair_outcome
                need_to_upload_collector.add(pair_parent)
            elif pair_outcome is not None:
                pair_outcomes.append(pair_outcome)
            else:
                need_to_upload_collector.add(pair_parent)
        for redirect_consumer, by_hop in redirects.values():
            mirror_uuids = {mirror.uuid for mirror, _text in by_hop.values()}
            ambiguous = any(
                sum(1 for c, h, _p in hop_same_framework_pairs if c is redirect_consumer and h.uuid == hop_uuid) > 1
                for hop_uuid in by_hop
            )
            if ambiguous or len(mirror_uuids) != 1 or set(by_hop) != set(redirect_consumer.tfs_ids):
                pair_outcomes.append(min(text for _mirror, text in by_hop.values()))
        if pair_outcomes:
            raise ValueError(min(pair_outcomes))
        for redirect_consumer, by_hop in redirects.values():
            mirror = next(iter(by_hop.values()))[0]
            redirect_consumer.tfs_ids = {mirror.uuid}
            redirect_consumer.features.any_uuid = mirror.uuid
            if mirror.private_copy:
                mirror.copy_readers |= redirect_consumer.get_uuids()
            for dropped_uuid in sorted(by_hop):
                dropped = tfs_by_uuid[dropped_uuid]
                mirror.order_after_uuids |= dropped.required_uuids
                redirect_consumer.required_uuids.discard(dropped.uuid)
                if not any(
                    dropped.uuid in step.required_uuids
                    or dropped.uuid in getattr(step, "order_after_uuids", ())
                    or (isinstance(step, FeatureGroupStep) and dropped.uuid in step.tfs_ids)
                    for step in new_execution_plan
                    if step is not dropped
                ) and not any(dropped.uuid in served for served in hop_serves.values()):
                    new_execution_plan.remove(dropped)
                    self.tfs_collection.pop(dropped, None)
            redirect_consumer.required_uuids.add(mirror.uuid)

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

        # A hop's SOURCE-side cfw is credited via owed_tokens, letting it drop once read. Multi-route rule: a consumer
        # reaching two or more hops out of one source framework is credited by none, since one hop finishing could drop
        # the cfw while another still reads it. Per-source guard: a hop is ineligible when its OWN source data - not
        # merely its framework class - is also the source or destination of another JoinStep. The identity check is
        # OWNING-STEP granularity (which FeatureGroupStep produced the data), not true physical-cfw granularity:
        # several FeatureGroupSteps in one same-framework chain under a root can share ONE physical cfw instance at
        # runtime (ComputeFrameworkExecutor.add_compute_framework reuses an existing cfw whenever CfwManager.get_cfw_uuid
        # finds the feature in some cfw's children_if_root), so two different owning steps can disagree with true cfw
        # sharing. That can only make the guard MORE conservative, never less: the transitive children_if_root
        # membership check the runtime drop path applies downstream (see run.py's _drop_tfs_source_if_possible /
        # _drop_join_source_if_possible) still gates the actual drop on every real reader of the shared cfw, whatever
        # this guard decided. Scope: a join hop credits only its destination-framework consumer; APPEND/UNION hops
        # stay finalize-only.
        joinsteps_by_uuid: dict[UUID, JoinStep] = {js.uuid: js for js in left_join_frameworks}

        def _owning_step(feature_uuid: UUID) -> UUID:
            """The uuid of the FeatureGroupStep that owns a feature uuid; a bare feature uuid
            owns itself when it has no FeatureGroupStep of its own (hand-built steps, tests)."""
            return owning_step_of.get(feature_uuid, feature_uuid)

        # Route tokens and from_framework are recorded for every hop, eligible or not: an ineligible hop
        # still reads its source cfw, so it still counts as a route for the multi-route rule below.
        route_tokens_by_hop: dict[UUID, set[UUID]] = {}
        hop_from_framework: dict[UUID, type[ComputeFramework]] = {}
        eligible_hops: dict[UUID, TransformFrameworkStep] = {}
        # A hop's own source, keyed by hop uuid; always resolved (see the invariant checks below),
        # read by the transitive reach walk further down.
        hop_source_owner_by_hop: dict[UUID, UUID] = {}
        for _ep in new_execution_plan:
            if not isinstance(_ep, TransformFrameworkStep):
                continue

            if _ep.link_id is None:
                served_joinstep_uuids: set[UUID] = set()
                route_tokens = {_ep.uuid}
            else:
                served_joinstep_uuids = hop_serves.get(_ep.uuid, set())
                route_tokens = set(served_joinstep_uuids)

            route_tokens_by_hop[_ep.uuid] = route_tokens
            hop_from_framework[_ep.uuid] = _ep.from_framework

            served_joinsteps = [joinsteps_by_uuid[uuid] for uuid in served_joinstep_uuids]
            if any(js.link.jointype in (JoinType.APPEND, JoinType.UNION) for js in served_joinsteps):
                continue

            # The hop's own source identity, normalized to the owning FeatureGroupStep: a plain
            # hop's source_step_uuid is already that owner; a join hop's source_framework_uuid is
            # the raw feature uuid the served JoinStep reads from, so it needs the same normalizing.
            # Both are always resolvable given how hops are built: a plain hop's source_step_uuid is
            # set unconditionally above (owning_step_of.get(parent, parent) never returns None), and
            # _validate_join_step_uuids guarantees every JoinStep - hence fill_tfs_by_joinstep's hop -
            # has a non-empty source_framework_uuids, so a join hop's source_framework_uuid is never
            # None either.
            if _ep.link_id is None:
                if _ep.source_step_uuid is None:
                    raise ValueError(
                        internal_invariant_error(
                            "Plain hop has no source_step_uuid.",
                            f"hop uuid={_ep.uuid}, from_framework={_ep.from_framework.get_class_name()}",
                            "add_tfs always sets source_step_uuid when building a plain hop.",
                        )
                    )
                hop_source_owner: UUID = _ep.source_step_uuid
            else:
                if _ep.source_framework_uuid is None:
                    raise ValueError(
                        internal_invariant_error(
                            "Join hop has no source_framework_uuid.",
                            f"hop uuid={_ep.uuid}, link_id={_ep.link_id}",
                            "_validate_join_step_uuids guarantees every JoinStep has a non-empty "
                            "source_framework_uuids.",
                        )
                    )
                hop_source_owner = _owning_step(_ep.source_framework_uuid)

            hop_source_owner_by_hop[_ep.uuid] = hop_source_owner

            other_join_source_owners = {
                _owning_step(uuid)
                for js in joinsteps_by_uuid.values()
                if js.uuid not in served_joinstep_uuids
                for uuid in (js.source_framework_uuids | js.destination_framework_uuids)
            }
            if hop_source_owner in other_join_source_owners:
                continue

            eligible_hops[_ep.uuid] = _ep

        hop_uuid_by_route_token: dict[UUID, UUID] = {
            token: hop_uuid for hop_uuid, tokens in route_tokens_by_hop.items() for token in tokens
        }

        feature_group_steps_by_uuid: dict[UUID, FeatureGroupStep] = {
            _ep.uuid: _ep for _ep in new_execution_plan if isinstance(_ep, FeatureGroupStep)
        }

        # A hop's credit extends past its own direct consumer to that consumer's same-framework
        # descendants, but only along the part of a descendant's dependency
        # chain that never branches away back to the hop's source, or to another hop out of a source
        # framework the multi-route rule below already treats as ambiguous. `_reach` walks a
        # FeatureGroupStep's required_uuids once, memoized: a hop-route token is a hop boundary,
        # crossed unconditionally and tagged with the hop's own source owner, while a same-framework
        # producer token is crossed only when frameworks match (tagged `None`, a bypass marker) and
        # folds in whatever that dependency itself already reaches. A raw ancestor token whose owner's
        # framework does NOT match is the ancestor set's own leftover bookkeeping - the same
        # dependency is already captured, framework-correctly, by its own hop token elsewhere in
        # required_uuids - so it is skipped rather than read as a spurious bypass.
        #
        # Complexity: memoization visits each step once, but FeatureGroupStep.required_uuids holds
        # the FULL TRANSITIVE ancestor set, not just direct parents, so a chain of N same-framework
        # steps makes this walk do O(N) token-loop work per step across O(N) steps, merging O(N)-sized
        # by_owner dicts at each - roughly O(N^3) for one long single-framework chain, not the
        # O(steps * features) bound elsewhere in this block.
        #
        # Recursion depth: an explicit worklist, not native recursion, walks the dependency DAG below,
        # so an arbitrarily long single-framework chain cannot raise RecursionError; this assumes the
        # step dependency graph is acyclic, an invariant the rest of add_tfs already relies on.
        reach_cache: dict[UUID, dict[UUID, frozenset[UUID | None]]] = {}
        all_hops_cache: dict[UUID, frozenset[UUID]] = {}

        def _reach(start_uuid: UUID) -> dict[UUID, frozenset[UUID | None]]:
            cached = reach_cache.get(start_uuid)
            if cached is not None:
                return cached

            work: list[tuple[UUID, bool]] = [(start_uuid, False)]
            while work:
                step_uuid, dependencies_ready = work.pop()
                if step_uuid in reach_cache:
                    continue

                step = feature_group_steps_by_uuid.get(step_uuid)
                if step is None:
                    reach_cache[step_uuid] = {}
                    all_hops_cache[step_uuid] = frozenset()
                    continue

                if not dependencies_ready:
                    pending: list[UUID] = []
                    for token in step.required_uuids:
                        hop_uuid = hop_uuid_by_route_token.get(token)
                        if hop_uuid is not None:
                            owner = hop_source_owner_by_hop.get(hop_uuid)
                            if owner is not None and owner not in reach_cache:
                                pending.append(owner)
                            continue

                        dep_uuid = owning_step_of.get(token)
                        if dep_uuid is None or dep_uuid == step_uuid:
                            continue
                        dep_step = feature_group_steps_by_uuid.get(dep_uuid)
                        if dep_step is None or dep_step.compute_framework != step.compute_framework:
                            continue
                        if dep_uuid not in reach_cache:
                            pending.append(dep_uuid)

                    if pending:
                        work.append((step_uuid, True))
                        work.extend((dep_uuid, False) for dep_uuid in pending)
                        continue

                by_owner: dict[UUID, set[UUID | None]] = defaultdict(set)
                hops: set[UUID] = set()
                for token in step.required_uuids:
                    hop_uuid = hop_uuid_by_route_token.get(token)
                    if hop_uuid is not None:
                        hops.add(hop_uuid)
                        owner = hop_source_owner_by_hop.get(hop_uuid)
                        if owner is not None:
                            by_owner[owner].add(hop_uuid)
                            for other_owner, markers in reach_cache[owner].items():
                                by_owner[other_owner] |= markers
                            hops |= all_hops_cache[owner]
                        continue

                    dep_uuid = owning_step_of.get(token)
                    if dep_uuid is None or dep_uuid == step_uuid:
                        continue
                    dep_step = feature_group_steps_by_uuid.get(dep_uuid)
                    if dep_step is None or dep_step.compute_framework != step.compute_framework:
                        continue
                    by_owner[dep_uuid].add(None)
                    for other_owner, markers in reach_cache[dep_uuid].items():
                        by_owner[other_owner] |= markers
                    hops |= all_hops_cache[dep_uuid]

                reach_cache[step_uuid] = {owner: frozenset(markers) for owner, markers in by_owner.items()}
                all_hops_cache[step_uuid] = frozenset(hops)

            return reach_cache[start_uuid]

        owed_by_hop: dict[UUID, set[UUID]] = {hop_uuid: set() for hop_uuid in eligible_hops}
        for _consumer in new_execution_plan:
            if not isinstance(_consumer, FeatureGroupStep):
                continue

            owner_reach = _reach(_consumer.uuid)
            reached_hop_uuids = all_hops_cache[_consumer.uuid]
            frameworks_reached: dict[type[ComputeFramework], int] = defaultdict(int)
            for hop_uuid in reached_hop_uuids:
                frameworks_reached[hop_from_framework[hop_uuid]] += 1

            for markers in owner_reach.values():
                if len(markers) != 1:
                    continue
                (sole_marker,) = markers
                if sole_marker is None:
                    continue
                if frameworks_reached[hop_from_framework[sole_marker]] > 1:
                    continue
                hop = eligible_hops.get(sole_marker)
                if hop is None or (hop.link_id is not None and _consumer.compute_framework != hop.to_framework):
                    continue
                owed_by_hop[sole_marker].update(_consumer.get_uuids())

        for hop_uuid, owed in owed_by_hop.items():
            eligible_hops[hop_uuid].owed_tokens = frozenset(owed)

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
    ) -> list[JoinStep]:
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

        if link.jointype == JoinType.RIGHT and link_fw[1] != link_fw[2] and children_uuids:
            child_frameworks = {graph.get_nodes()[u].feature.get_compute_framework() for u in children_uuids}
            if child_frameworks == {link_fw[1]}:
                # Consumers sit on the left side's framework only: the join runs there, like INNER.
                destination_framework = link_fw[1]
                source_framework = link_fw[2]

        swap_merge_sides = False

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

        join_steps: list[JoinStep] = []
        parts: list[tuple[set[UUID], JoinSide | None, bool, bool]] = []
        for component, varying_side in self._independent_link_children(link, children_uuids, graph):
            pieces = self._split_by_carrier_frame(
                link, component, varying_side, {destination_framework, source_framework}, graph
            )
            parts.extend((members, side, beside, len(pieces) > 1) for members, side, beside in pieces)
        for component, varying_side, beside_carried, is_split in parts:
            js = self._plan_link_join(
                link_fw,
                link_trekker,
                graph,
                pre_execution_plan,
                component,
                destination_framework,
                source_framework,
                swap_merge_sides,
                attempted_key,
                varying_side,
                beside_carried,
            )
            if js is not None:
                if is_split:
                    js.split_consumers = frozenset(component)
                join_steps.append(js)
        return join_steps

    def _split_by_carrier_frame(
        self,
        link: Link,
        component: set[UUID],
        varying_side: JoinSide | None,
        frameworks: set[type[ComputeFramework]],
        graph: Graph,
    ) -> list[tuple[set[UUID], JoinSide | None, bool]]:
        """Split consumers reading a side through one carrier step and directly; the direct part is flagged."""
        unsplit = [(component, varying_side, False)]
        if link.jointype in (JoinType.APPEND, JoinType.UNION) or len(component) < 2:
            return unsplit
        by_carriers: dict[frozenset[UUID], set[UUID]] = {}
        for child in sorted(component):
            split = split_by_declared_side(link, set(graph.parent_to_children_mapping[child]), graph)
            carriers, sides, _ = self._carrier_scan({child}, split, graph, frameworks)
            if len(sides) > 1:
                return unsplit
            by_carriers.setdefault(carriers, set()).add(child)
        carried_keys = [key for key in by_carriers if key]
        if len(by_carriers) < 2 or len(carried_keys) != 1:
            return unsplit
        carried_children = by_carriers[carried_keys[0]]
        carried_split = split_by_declared_side(
            link, {p for c in carried_children for p in graph.parent_to_children_mapping[c]}, graph
        )
        _, carriers_on_left = self._join_carriers(carried_children, carried_split, frameworks, graph)
        side = JoinSide.LEFT if carriers_on_left else JoinSide.RIGHT
        return [(members, side, not key) for key, members in by_carriers.items()]

    def _independent_link_children(
        self, link: Link, children_uuids: set[UUID], graph: Graph
    ) -> list[tuple[set[UUID], JoinSide | None]]:
        """Split children reading disjoint side steps (e.g. option variants), each getting a join, with the varying side."""
        step_of = {uuid: index for index, uuids in enumerate(self.feature_set_collections) for uuid in uuids}
        components: list[tuple[set[UUID], set[int]]] = []
        side_steps: dict[UUID, tuple[frozenset[int], frozenset[int]]] = {}
        for child in sorted(children_uuids):
            split = split_by_declared_side(link, set(graph.parent_to_children_mapping[child]), graph)
            left = frozenset(step_of[uuid] for uuid in split.left_uuids_any_distance if uuid in step_of)
            right = frozenset(step_of[uuid] for uuid in split.right_uuids_any_distance if uuid in step_of)
            steps = set(left | right)
            if not steps:
                return [(children_uuids, None)]
            side_steps[child] = (left, right)
            linked = [component for component in components if component[1] & steps]
            merged = (
                {child}.union(*(component[0] for component in linked)),
                steps.union(*(component[1] for component in linked)),
            )
            components = [component for component in components if component not in linked] + [merged]
        nodes = graph.get_nodes()
        result: list[tuple[set[UUID], JoinSide | None]] = []
        for members, _ in components:
            by_variant: dict[tuple[type[FeatureGroup], str], list[UUID]] = {}
            for member in sorted(members):
                key = (nodes[member].feature_group_class, str(nodes[member].name))
                by_variant.setdefault(key, []).append(member)
            conflicting = any(
                len({side_steps[v][0] for v in variants}) > 1 or len({side_steps[v][1] for v in variants}) > 1
                for variants in by_variant.values()
            )
            if not conflicting:
                result.append((members, None))
                continue
            fan_out = self._split_by_varying_side(members, side_steps)
            result.extend(fan_out)
        return result

    @staticmethod
    def _split_by_varying_side(
        members: set[UUID], side_steps: dict[UUID, tuple[frozenset[int], frozenset[int]]]
    ) -> list[tuple[set[UUID], JoinSide]]:
        """One part per group of members sharing a varying-side step."""
        if len({side_steps[m][0] for m in members}) == 1:
            index, varying = 1, JoinSide.RIGHT
        elif len({side_steps[m][1] for m in members}) == 1:
            index, varying = 0, JoinSide.LEFT
        else:
            raise ValueError(
                internal_invariant_error(
                    "option variants read differing variants of both join sides in one component.",
                    f"members={sorted(members)}",
                )
            )
        parts: list[tuple[set[UUID], set[int]]] = []
        for member in sorted(members):
            steps = set(side_steps[member][index])
            linked = [part for part in parts if part[1] & steps]
            merged = ({member}.union(*(p[0] for p in linked)), steps.union(*(p[1] for p in linked)))
            parts = [part for part in parts if part not in linked] + [merged]
        return [(part[0], varying) for part in parts]

    @staticmethod
    def _named(graph: Graph, uuids: set[UUID] | frozenset[UUID]) -> str:
        nodes = graph.get_nodes()
        return ", ".join(
            sorted(
                f"{format_feature_group_class(nodes[u].feature_group_class)} "
                f"({nodes[u].feature.get_compute_framework().get_class_name()})"
                for u in uuids
            )
        )

    @staticmethod
    def _polymorphic_member(
        split: DeclaredSideSplit, graph: Graph, frameworks: set[type[ComputeFramework]]
    ) -> UUID | None:
        """First side member beyond the nearest split that runs on one of `frameworks`."""
        nodes = graph.get_nodes()
        beyond = (split.left_uuids_any_distance | split.right_uuids_any_distance) - split.left_uuids - split.right_uuids
        return next((u for u in sorted(beyond) if nodes[u].feature.get_compute_framework() in frameworks), None)

    def _plan_link_join(
        self,
        link_fw: LinkFrameworkTrekker,
        link_trekker: LinkTrekker,
        graph: Graph,
        pre_execution_plan: list[LinkFrameworkTrekker | FeatureGroupStep],
        children_uuids: set[UUID],
        destination_framework: type[ComputeFramework],
        source_framework: type[ComputeFramework],
        swap_merge_sides: bool,
        attempted_key: LinkFrameworkTrekker,
        varying_side: JoinSide | None = None,
        beside_carried: bool = False,
    ) -> JoinStep | None:
        link = link_fw[0]
        entered_same_framework = destination_framework == source_framework

        # This gets the parent ids of the joinstep, which needs to be calculated before the link.
        required_uuids: set[UUID] = set()
        for uuid in children_uuids:
            required_uuids.update(graph.parent_to_children_mapping[uuid])

        # Split before the order-edge link uuids join required_uuids.
        split = split_by_declared_side(link, required_uuids, graph)

        declared_left_frameworks = {graph.get_nodes()[u].feature.get_compute_framework() for u in split.left_uuids}
        declared_right_frameworks = {graph.get_nodes()[u].feature.get_compute_framework() for u in split.right_uuids}

        declared_carried_uuids: frozenset[UUID] | None = None
        declared_other_uuids: frozenset[UUID] = frozenset()
        declared_carried_left = False
        if link.jointype not in (JoinType.APPEND, JoinType.UNION) and destination_framework != source_framework:
            scanned, scanned_sides, _ = self._carrier_scan(
                children_uuids, split, graph, {destination_framework, source_framework}
            )
            if scanned and len(scanned_sides) == 1:
                # The join runs on the consumers' framework; the carrier frame hops in there.
                consumer_frameworks = {graph.get_nodes()[c].feature.get_compute_framework() for c in children_uuids}
                carried_left = next(iter(scanned_sides))
                carried_fws, other_fws = (
                    (declared_left_frameworks, declared_right_frameworks)
                    if carried_left
                    else (declared_right_frameworks, declared_left_frameworks)
                )
                if len(consumer_frameworks) == 1 and consumer_frameworks <= {destination_framework, source_framework}:
                    consumer_framework = next(iter(consumer_frameworks))
                    if consumer_framework in other_fws and consumer_framework not in carried_fws:
                        destination_framework = source_framework = consumer_framework
                        declared_carried_left = carried_left
                        declared_carried_uuids, declared_other_uuids = (
                            (split.left_uuids, split.right_uuids)
                            if carried_left
                            else (split.right_uuids, split.left_uuids)
                        )
                    elif consumer_framework in carried_fws:
                        if consumer_framework == source_framework:
                            source_framework = destination_framework
                        destination_framework = consumer_framework
                    else:
                        member = self._polymorphic_member(split, graph, {consumer_framework})
                        name = (
                            format_feature_group_class(graph.get_nodes()[member].feature_group_class)
                            if member is not None
                            else "A side member"
                        )
                        raise ValueError(
                            f"The consumers {self._named(graph, children_uuids)} of {link} read a link side through "
                            f"the carriers {self._named(graph, scanned)}, but {name} is a member of that side on "
                            f"{consumer_framework.get_class_name()}, and one consumer cannot join members of one link "
                            "side on different compute frameworks. Read one member of the link side per consumer."
                        )
                else:
                    member = self._polymorphic_member(split, graph, {destination_framework, source_framework})
                    message = (
                        f"The consumers {self._named(graph, children_uuids)} of {link} read through the carriers "
                        f"{self._named(graph, scanned)}, but the join runs on "
                        f"{destination_framework.get_class_name()} and {source_framework.get_class_name()}."
                    )
                    if member is not None:
                        name = format_feature_group_class(graph.get_nodes()[member].feature_group_class)
                        message += (
                            f" {name} is a polymorphic member of a link side and sets one of those frameworks."
                            " Read one member of the link side per consumer."
                        )
                    else:
                        message += " Compute the consumer on one of the join's frameworks."
                    raise ValueError(message)

        # This filters the required_uuids to only the one with the final compute framework.
        destination_framework_uuids: set[UUID] = set()
        source_framework_uuids: set[UUID] = set()

        for uuid in required_uuids:
            node_framework = graph.get_nodes()[uuid].feature.get_compute_framework()

            if node_framework == destination_framework:
                destination_framework_uuids.add(uuid)

            if node_framework == source_framework:
                source_framework_uuids.add(uuid)
        if declared_carried_uuids is not None:
            destination_framework_uuids, source_framework_uuids = set(declared_carried_uuids), set(declared_other_uuids)
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
        if declared_carried_uuids is not None:
            swap_sides = not declared_carried_left

        # This part is for handling specific join cases. Currently, we only deal with equal feature groups.
        for children_uuid in children_uuids:
            children_fw = graph.get_nodes()[children_uuid].feature.get_compute_framework()

            # This runs with the assumption that children_uuids is exactly 1.
            # result = True
            result = self.is_valid_join_step(link_fw, children_fw, children_uuid, graph)
            if result is False:
                if attempted_key not in self.declined_orientations:
                    self.declined_orientations.append(attempted_key)
                return None
            elif result is True:
                pass
            elif declared_carried_uuids is None:
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
        carriers: frozenset[UUID] = frozenset()
        shared_source = False
        shared_destination = False
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

            carriers, carriers_on_left = self._join_carriers(
                children_uuids,
                split,
                {destination_framework, source_framework},
                graph,
                {destination_framework} if entered_same_framework else None,
            )
            if carriers and (side is JoinSide.LEFT) != carriers_on_left:
                side = JoinSide.LEFT if carriers_on_left else JoinSide.RIGHT

            # Joins fanned out over one shared side must not all merge into that side's frame.
            if beside_carried and not carriers:
                shared_destination = True
            elif varying_side is not None and not carriers and side is not varying_side:
                if destination_framework == source_framework:
                    side = varying_side
                else:
                    shared_destination = True
            shared_source = varying_side is not None and destination_framework == source_framework

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
            carriers=carriers,
            shared_source=shared_source,
            shared_destination=shared_destination,
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
        For a key in declared order, `run_link` sets destination_framework to link_fw[2] for JoinType.RIGHT
        (unless it moved the join onto link_fw[1] because the consumers only run there), and link_fw[2] is
        the declared right framework, so membership settles on right. For a key
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

        moved_left = destination_framework == trekker_left_framework
        if jointype == JoinType.RIGHT and destination_framework != source_framework and not moved_left:
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
                    new_children_uuids.discard(c_o_c)

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
        return Link.matches_discriminator(discriminator, graph.nodes[uuid].feature.options)

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
        nodes: dict[UUID, NodeProperties] | None = None,
    ) -> dict[Any, FeatureGroupStep]:
        feature_group, features = feature_group_features[0], feature_group_features[1]
        features_grouped_by_framework_and_options: dict[Any, set[Feature]] = (
            self.group_features_by_compute_framework_and_options(features)
        )
        if isinstance(feature_group.input_data(), ApiInputData) and self.api_input_data_collection is not None:
            api_groups: dict[tuple[int, str], set[Feature]] = defaultdict(set)
            for f_hash, grouped_features in features_grouped_by_framework_and_options.items():
                for feature in grouped_features:
                    source_key, _ = self.api_input_data_collection.get_name_cls_by_matching_column_name(feature.name)
                    api_groups[(f_hash, source_key)].add(feature)
            features_grouped_by_framework_and_options = api_groups

        # Only a root feature group (no upstream ancestors) can be the actual cause of a
        # missing-Links error: a split with ancestors of its own is never the source read directly.
        is_root = not any(parent_to_children_mapping.get(feature.uuid) for feature in features)
        if is_root:
            split_keys = frozenset(key for feature in features for key in feature.options.inherited_context_keys)
            self._option_split_keys[feature_group] = self._option_split_keys.get(feature_group, frozenset()) | (
                split_keys
            )
            split_buckets = self._option_split_buckets.setdefault(feature_group, {})
            for f_hash, grouped_features in features_grouped_by_framework_and_options.items():
                representative = next(iter(grouped_features))
                bucket = split_buckets.setdefault(f_hash, (representative, set()))
                bucket[1].update(feature.uuid for feature in grouped_features)

        fg_steps: dict[Any, FeatureGroupStep] = {}

        root_parent_children_mapping = self.get_parent_children_mapping(parent_to_children_mapping)

        split_groups: list[tuple[Any, int, set[Feature]]] = []
        for f_hash, features in features_grouped_by_framework_and_options.items():
            if nodes is None:
                split_groups.append((f_hash, 0, features))
                continue
            for bucket_idx, members in enumerate(
                self._split_by_variant_conflicts(features, parent_to_children_mapping, nodes)
            ):
                split_groups.append((f_hash, bucket_idx, members))

        for f_hash, bucket_idx, features in split_groups:
            sub_groups = self._split_features_by_dependency_levels(features, parent_to_children_mapping)

            for level_idx, sub_features in enumerate(sub_groups):
                pre_calculated = self.retrieve_nodes_which_must_be_calculated_before(
                    sub_features, parent_to_children_mapping
                )
                pre_calculated.update(copy(pre_required_uuids))

                chosen = {f.get_compute_framework() for f in sub_features}
                if len(chosen) != 1:
                    names = sorted(c.get_class_name() for c in chosen)
                    raise ValueError(f"Step of {feature_group.get_class_name()} mixes compute frameworks {names}.")
                cf = next(iter(chosen))

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
                    edge_pairs: list[tuple[str, frozenset[str]]] = []
                    for feature in sub_features:
                        resolved_names = self.resolved_input_feature_names.get(feature.uuid) or frozenset()
                        union.update(resolved_names)
                        edge_pairs.append((str(feature.name), resolved_names))
                    feature_set.declared_input_feature_names = frozenset(union) or None
                    feature_set.declared_input_feature_edges = merge_input_feature_edges(edge_pairs)
                    feature_set.declared_input_features_resolved = True

                if self.specialized_from is not None:
                    feature_set.specialized_from = tuple(
                        sorted({name for f in sub_features for name in self.specialized_from.get(f.uuid, ())})
                    )

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

                fg_steps[(f_hash, bucket_idx, level_idx)] = feature_group_step
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

    @staticmethod
    def group_features_by_compute_framework_and_options(features: set[Feature]) -> dict[int, set[Feature]]:
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
