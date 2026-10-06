from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import logging
from typing import Any, cast
from uuid import UUID
import uuid

from mloda.core.abstract_plugins.components.index.add_index_feature import create_index_feature
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.prepare.accessible_plugins import EnvironmentPreconditionError, PreFilterPlugins
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.connection_requirement import ConnectionRequirement
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import (
    Extender,
    ExtenderHook,
    _invoke_extender,
    build_hook_extenders,
)
from mloda.core.abstract_plugins.hook_context import HookContext, _no_rows, instrument
from mloda.core.abstract_plugins.plugin_version import resolve_plugin_version
from mloda.core.abstract_plugins.plan_context import PlanContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.build_graph import BuildGraph
from mloda.core.prepare.resolve_graph import ResolveGraph
from mloda.core.prepare.resolution_failure_renderer import _candidate_sort_key
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.core.prepare.resolution_types import (
    EvaluationResult,
    ResolutionRecord,
)
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.core.abstract_plugins.feature_group import FeatureGroup, format_feature_group_class
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.hashable_dict import _deep_equal
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.abstract_plugins.components.validators.link_validator import LinkValidator


logger = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        features: Features,
        compute_frameworks: set[type[ComputeFramework]],
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None = None,
        global_filter: GlobalFilter | None = None,
        api_input_data_collection: ApiInputDataCollection | None = None,
        plugin_collector: PluginCollector | None = None,
        column_ordering: str | None = None,
        function_extender: set[Extender] | None = None,
        plan_context: PlanContext | None = None,
        framework_preference: Mapping[type[ComputeFramework], int] | None = None,
        output_framework: type[ComputeFramework] | None = None,
    ) -> None:
        self.output_framework = output_framework
        self.framework_positions: Mapping[type[ComputeFramework], int] = framework_preference or {}
        self.filter_ties: list[tuple[UUID, UUID]] = []
        self.connection_dropped: dict[UUID, frozenset[type[ComputeFramework]]] = {}
        # setup variables which track the primary sources and the compute platforms
        self.function_extender = function_extender if function_extender is not None else set()
        self._hook_extenders = build_hook_extenders(self.function_extender)
        self.plan_context = plan_context
        self.run_context = RunContext(plan_id=plan_context.plan_id if plan_context else None)
        # Holds the Feature objects ResolveComputeFrameworks.links rewrites: hash-stale after planning, so only read it before planning (as today).
        self.feature_group_collection: dict[type[FeatureGroup], set[Feature]] = defaultdict(set)

        # use global filters
        self.global_filter = global_filter

        # Tracks feature relation to its parents
        self.feature_link_parents: dict[UUID, set[UUID]] = defaultdict(set)

        # get accessible feature groups and their compute platforms
        self.accessible_plugins = PreFilterPlugins(compute_frameworks, plugin_collector).get_accessible_plugins()
        # get links
        LinkValidator.validate_links(links)
        self.links = set(links) if links is not None else None

        # set api input collection if relevant
        self.api_input_data_collection = api_input_data_collection

        self.plugin_collector = plugin_collector

        self.data_access_collection = data_access_collection
        self.output_connection = self._resolve_output_connection()
        self.column_ordering = column_ordering
        self.request_feature_order: list[str] = [str(f.name) for f in features]
        self._dual_consumption_warned: set[tuple[str, str, frozenset[str]]] = set()
        self._property_mapping_keys_cache: dict[type[FeatureGroup], frozenset[str]] = {}
        # Intake materialization memo: the strong source reference keeps its id stable, so features
        # sharing one pre-default Options share one effective Options.
        self._intake_options_memo: dict[tuple[type[FeatureGroup], int], tuple[Options, Options]] = {}
        # Declared (pre-default) options per surviving feature uuid, for default-equivalent merge warnings.
        self._declared_options_by_uuid: dict[UUID, Options] = {}
        # Per feature uuid, _handle_input_features_recursion's result (None: root; injected filter/index: no entry).
        self.resolved_input_feature_names: dict[UUID, frozenset[str] | None] = {}
        # Per surviving feature uuid, the parents its winning group replaced, unioned over merged duplicates.
        self.specialized_from: dict[UUID, tuple[type[FeatureGroup], ...]] = {}
        self.resolution_records: list[ResolutionRecord] = []
        self.execution_planner = self.create_setup_execution_plan(features)
        if self.function_extender:
            self.run_context = replace(self.run_context, plugin_versions=self._resolve_plugin_versions())
        self.tfs_connection_map = self._resolve_tfs_connection_map()

    def _resolve_plugin_versions(self) -> dict[str, str | None] | None:
        """Resolves each planned feature group module's owning-distribution version at plan time, so hooks only read it."""
        versions = {
            step.feature_group.__module__: resolve_plugin_version(step.feature_group.__module__)
            for step in self.execution_planner
            if isinstance(step, FeatureGroupStep)
        }
        return versions or None

    def _resolve_output_connection(self) -> Any:
        if self.output_framework is None:
            return None
        connection = self.output_framework.pick_connection_from_dac(self.data_access_collection)
        if connection is None and self.output_framework.connection_requirement() is ConnectionRequirement.REQUIRED:
            raise ValueError(
                f"output_framework {self.output_framework.get_class_name()} requires a connection, "
                "but none was found in the data access collection."
            )
        return connection

    def _resolve_tfs_connection_map(self) -> dict[type[ComputeFramework], Any]:
        """Resolve a connection per TFS destination framework at setup time.

        Raises EnvironmentPreconditionError when a converting hop into a REQUIRED-connection framework has none.
        """
        connection_map: dict[type[ComputeFramework], Any] = {}
        for tfs in self.execution_planner.tfs_collection.values():
            cfw_class = tfs.to_framework
            if cfw_class in connection_map:
                continue
            conn = cfw_class.pick_connection_from_dac(self.data_access_collection)
            if conn is not None:
                connection_map[cfw_class] = conn
            elif (
                cfw_class.connection_requirement() is ConnectionRequirement.REQUIRED
                and tfs.from_framework.expected_data_framework() is not cfw_class.expected_data_framework()
            ):
                raise EnvironmentPreconditionError(
                    f"The transform from {tfs.from_feature_group.get_class_name()} on "
                    f"{tfs.from_framework.get_class_name()} to {tfs.to_feature_group.get_class_name()} on "
                    f"{cfw_class.get_class_name()} needs a connection, so add a matching connection "
                    "to the DataAccessCollection."
                )
        return connection_map

    def get_function_extender(self, hook: ExtenderHook) -> Extender | None:
        """Select the extender(s) registered for hook, from the table built at init."""
        return self._hook_extenders.get(hook)

    def compute(self, flight_server: ParallelRunnerFlightServer | None = None) -> ExecutionOrchestrator:
        execution_plan_copy = deepcopy(self.execution_planner)
        orchestrator = ExecutionOrchestrator(
            execution_plan_copy,
            flight_server,
            column_ordering=self.column_ordering,
            request_feature_order=self.request_feature_order,
            tfs_connection_map=self.tfs_connection_map,
            run_context=self.run_context,
            output_framework=self.output_framework,
            output_connection=self.output_connection,
        )
        if isinstance(orchestrator, ExecutionOrchestrator):
            return orchestrator
        raise ValueError("ExecutionOrchestrator setup failed.")

    def create_setup_execution_plan(self, features: Features) -> ExecutionPlan:
        if self.global_filter:
            self.global_filter.reset_match_tracking()

        self.setup_features_recursion(features)
        self._fold_into_narrower_twins()

        if self.global_filter:
            self.global_filter.warn_on_unmatched_filters()

        graph_builder = BuildGraph(self.feature_link_parents, self.feature_group_collection)
        graph_builder.build_graph_from_feature_links()
        graph = graph_builder.graph

        # resolve graph into a queue
        resolver = ResolveGraph(
            graph,
            self.links,
            self.filter_ties,
            self.framework_positions,
            output_framework=self.output_framework,
            connection_dropped=self.connection_dropped,
            connected=self._connected_frameworks(),
        )
        resolver.create_initial_queue()

        resolver.set_nodes_per_feature_group()

        planned_queue = resolver.resolve_links()

        planned_queue = resolver.resolver_compute_framework.links(
            planned_queue, resolver.resolver_links.get_link_trekker()
        )

        if self.global_filter:
            # Setup still shifts a stored hash: a copied option Feature reaches the host's Feature via child_options.
            self.global_filter.rehash_stored_filters()

        execution_planner = ExecutionPlan(
            self.global_filter,
            self.api_input_data_collection,
            self.resolved_input_feature_names,
            {
                uuid: tuple(f"{c.__module__}.{c.__qualname__}" for c in classes)
                for uuid, classes in self.specialized_from.items()
            },
        )
        execution_planner.create_execution_plan(
            planned_queue,
            graph,
            resolver.resolver_links.get_link_trekker(),
            resolver.resolver_compute_framework.get_declared_frameworks(),
        )
        return execution_planner

    def setup_features_recursion(
        self,
        features: Features,
        requested: bool = True,
        depth: int = 0,
        consumer: str | None = None,
        path: tuple[str, ...] = (),
    ) -> None:
        # Register every sibling's own link before processing any, so index injection and feature-group
        # resolution see the whole batch regardless of order. Does not cover a link nested in a
        # co-sibling's input_features() subtree (see xfail
        # test_same_class_feature_link_nested_in_co_siblings_input_features_is_not_seen_in_time): only
        # index injection could defer that way, since identify_feature_group.py's
        # _filter_feature_group_by_links gate runs during resolution itself, before recursion completes.
        for feature in features:
            self.add_feature_link_to_links(feature)
        for feature in features:
            # Stamped right before resolution: a reused instance must name this consumer, not an earlier one.
            feature.resolving_consumer = consumer
            feature.resolving_path = path
            self._process_feature(feature, features, requested, depth)

    def _process_feature(self, feature: Feature, features: Features, requested: bool, depth: int = 0) -> None:
        """Processes a single feature by delegating tasks to helper methods."""

        # Feature-group matchers write into the options; those writes are never the feature's own declaration.
        feature.options.lock_own_keys()
        feature_group_class, compute_frameworks, result = self._identify_feature_group_and_frameworks(feature, depth)
        self.resolution_records.append(ResolutionRecord(str(feature.name), requested, result))
        self._warn_on_dual_option_consumption(feature, feature_group_class)
        feature_group = feature_group_class()

        self._set_feature_name(feature, feature_group)
        self._set_compute_framework_and_data_type(feature, compute_frameworks, feature_group_class)

        # Stash the declared pre-default options: dependency declaration and child inheritance observe
        # them; intake materialization canonicalizes default-equivalent twins.
        declared_options = feature.options
        added = self.add_feature_to_collection(
            feature_group_class, feature, features.child_uuid, specialized_from=result.specialized_from
        )

        if added:
            parent_domain = feature.domain.name if feature.domain else None
            self.resolved_input_feature_names[feature.uuid] = self._handle_input_features_recursion(
                feature_group_class,
                feature.uuid,
                declared_options,
                feature.name,
                parent_domain=parent_domain,
                depth=depth,
                path=feature.resolving_path,
            )

        if self.global_filter:
            self._add_filter_feature(feature_group_class, feature_group, feature, features)

        if feature_group.index_columns():
            self._add_index_feature(feature_group_class, feature_group, feature, features)
        elif self.links:
            self._add_index_feature_from_links(feature_group_class, feature_group, feature, features)

    def _set_feature_name(self, feature: Feature, feature_group: FeatureGroup) -> None:
        """Sets the feature name using the feature group's logic."""
        feature.name = feature_group.set_feature_name(feature.options, feature.name)

    def _set_compute_framework_and_data_type(
        self,
        feature: Feature,
        compute_frameworks: set[type[ComputeFramework]],
        feature_group_class: type[FeatureGroup],
    ) -> None:
        """Sets the compute framework and data type for the feature."""
        feature = self.set_compute_framework(feature, compute_frameworks, feature_group_class)
        feature.data_type = self.set_data_type(feature, feature_group_class)

    def _property_mapping_keys(self, feature_group_class: type[FeatureGroup]) -> frozenset[str]:
        """Returns the PROPERTY_MAPPING keys of a feature group class as a frozenset of str."""
        cached = self._property_mapping_keys_cache.get(feature_group_class)
        if cached is not None:
            return cached
        keys = feature_group_class.declared_option_keys()
        self._property_mapping_keys_cache[feature_group_class] = keys
        return keys

    def _warn_on_dual_option_consumption(self, feature: Feature, feature_group_class: type[FeatureGroup]) -> None:
        """Warns when a forwarded option key is declared by both the consumer and the resolved child group."""
        if not feature.consumer_attributions or not feature.options.inherited_group_keys:
            return
        child_keys = self._property_mapping_keys(feature_group_class)
        child_class_name = feature_group_class.get_class_name()
        for consumer_name, entry_keys in feature.consumer_attributions:
            overlap = entry_keys & child_keys
            if not overlap:
                continue
            warned_key = (consumer_name, child_class_name, overlap)
            if warned_key in self._dual_consumption_warned:
                continue
            self._dual_consumption_warned.add(warned_key)
            logger.warning(
                f"Option key(s) {sorted(overlap)} forwarded from consumer feature group "
                f"{consumer_name} to input feature '{feature.name}' resolved by "
                f"{child_class_name} are declared in the PROPERTY_MAPPING of both feature "
                f"groups, so the forwarded value now configures both. If this is unintended, set "
                f"forward_group_exclude={{...}}, an allowlist, or forward_group=False on the child in "
                f"input_features."
            )

    def _identify_feature_group_and_frameworks(
        self, feature: Feature, depth: int = 0
    ) -> tuple[type[FeatureGroup], set[type[ComputeFramework]], EvaluationResult]:
        """Identify the winning feature group via the shared helper; on failure it raises the enriched error."""
        extender = self.get_function_extender(ExtenderHook.FEATURE_GROUP_MATCHED)
        if extender is None:
            result = resolve_or_raise(
                feature,
                self.accessible_plugins,
                self.links,
                self.data_access_collection,
                partial_records=self.resolution_records,
            )
        else:
            result = self._resolve_with_match_hook(extender, feature, depth)
        feature_group_class, compute_frameworks = next(iter(result.identified.items()))
        return feature_group_class, compute_frameworks, result

    def _resolve_with_match_hook(self, extender: Extender, feature: Feature, depth: int) -> EvaluationResult:
        """Run resolve_or_raise through the extender under a HookContext.
        feature_group_class is None until the match resolves, then written post-hoc."""
        plan = self.plan_context
        context = HookContext(
            hook=ExtenderHook.FEATURE_GROUP_MATCHED,
            feature_group_class=None,
            feature_group_version=None,
            plugin_version=None,
            feature_names=(str(feature.name),),
            input_features=None,
            compute_framework_name=None,
            run_id=None,
            plan_id=self.run_context.plan_id,
            carrier=None,
            tenant_id=plan.tenant_id if plan else None,
            project_id=plan.project_id if plan else None,
            principal=plan.principal if plan else None,
            worker_index=None,
            plan_feature_count=len(self.resolution_records),
            plan_node_count=sum(len(features) for features in self.feature_group_collection.values()),
            plan_depth=depth,
        )

        def _resolve(*args: Any, **kwargs: Any) -> EvaluationResult:
            result = resolve_or_raise(*args, **kwargs)
            winner = next(iter(result.identified.items()))[0]
            # Sealed fields: the engine alone writes these post-hoc, bypassing the frozen guard.
            object.__setattr__(context, "feature_group_class", f"{winner.__module__}.{winner.__qualname__}")
            object.__setattr__(
                context,
                "specialized_from",
                tuple(sorted(f"{c.__module__}.{c.__qualname__}" for c in result.specialized_from)),
            )
            return result

        with context.activate():
            return cast(
                EvaluationResult,
                _invoke_extender(
                    extender,
                    instrument(context, _resolve, row_count=_no_rows),
                    feature,
                    self.accessible_plugins,
                    self.links,
                    self.data_access_collection,
                    partial_records=self.resolution_records,
                ),
            )

    def _add_index_feature(
        self,
        feature_group_class: type[FeatureGroup],
        feature_group: FeatureGroup,
        feature: Feature,
        features: Features,
    ) -> None:
        indexes = feature_group_class.index_columns()
        if indexes is None:
            raise ValueError(f"Feature group {format_feature_group_class(feature_group_class)} has no indexes defined.")

        if self.links is None:
            return

        for index in indexes:
            self._process_index_feature(feature_group_class, feature_group, feature, features, index)

    def _link_sides(self, link: Link, feature_group_class: type[FeatureGroup], feature: Feature) -> tuple[bool, bool]:
        """Sides match by class or subclass; discriminators narrow only when the class matches both sides."""
        left = issubclass(feature_group_class, link.left_feature_group)
        right = issubclass(feature_group_class, link.right_feature_group)

        if left and right:
            left = link.left_discriminator is None or Link.matches_discriminator(
                link.left_discriminator, feature.options
            )
            right = link.right_discriminator is None or Link.matches_discriminator(
                link.right_discriminator, feature.options
            )

        return left, right

    def _add_index_feature_from_links(
        self,
        feature_group_class: type[FeatureGroup],
        feature_group: FeatureGroup,
        feature: Feature,
        features: Features,
    ) -> None:
        """Adds index features derived from JoinSpec links when the feature group has no index_columns defined."""
        if self.links is None:
            return

        feature_name_str = feature.name

        for link in self.links:
            if link.jointype in (JoinType.APPEND, JoinType.UNION):
                continue
            left, right = self._link_sides(link, feature_group_class, feature)
            if left and feature_name_str in link.left_index.index:
                return
            if right and feature_name_str in link.right_index.index:
                return

        for link in self.links:
            if link.jointype in (JoinType.APPEND, JoinType.UNION):
                continue
            left, right = self._link_sides(link, feature_group_class, feature)
            if left:
                self._create_and_add_index_feature(
                    feature_group_class, feature_group, feature, features, link.left_index
                )
            if right:
                self._create_and_add_index_feature(
                    feature_group_class, feature_group, feature, features, link.right_index
                )

    def _process_index_feature(
        self,
        feature_group_class: type[FeatureGroup],
        feature_group: FeatureGroup,
        feature: Feature,
        features: Features,
        index: Index,
    ) -> None:
        """Processes the index feature for both left and right links."""
        if self.links is None:
            return

        for link in self.links:
            left, right = self._link_sides(link, feature_group_class, feature)
            if left and link.left_index == index:
                self._create_and_add_index_feature(feature_group_class, feature_group, feature, features, index)

            if right and link.right_index == index:
                self._create_and_add_index_feature(feature_group_class, feature_group, feature, features, index)

    def _create_and_add_index_feature(
        self,
        feature_group_class: type[FeatureGroup],
        feature_group: FeatureGroup,
        feature: Feature,
        features: Features,
        index: Index,
    ) -> None:
        """Creates and adds a new index feature to the collection."""
        new_index_feature = create_index_feature(index, feature_group, feature)
        self.add_feature_to_collection(feature_group_class, new_index_feature, features.child_uuid, True)
        if feature.uuid in self.connection_dropped:
            self.connection_dropped[new_index_feature.uuid] = self.connection_dropped[feature.uuid]

    def _add_filter_feature(
        self,
        feature_group_class: type[FeatureGroup],
        feature_group: FeatureGroup,
        feature: Feature,
        features: Features,
    ) -> None:
        if self.global_filter:
            matched_filters = self.global_filter.identify_matched_filters(
                feature_group_class, feature, self.data_access_collection
            )

            for match in matched_filters:
                match.filter_feature.name = feature_group.set_feature_name(
                    match.filter_feature.options, match.filter_feature.name
                )
                # We assign a new UUID to the filter feature to ensure
                # it is treated as a separate instance from the original filter feature
                match.filter_feature.uuid = uuid.uuid4()
                # Intake may materialize declared defaults into the filter feature's options: group fills shift
                # SingleFilter's hash, context fills shift its equality, so intake must run before it is stored.
                self.add_feature_to_collection(feature_group_class, match.filter_feature, features.child_uuid)
                declared = next((f for f in self.global_filter.filters if f.uuid == match.uuid), None)
                if declared is None:
                    raise ValueError(f"Matched filter on {feature.name} is not declared in the global filter.")
                if declared.filter_feature.compute_frameworks:
                    feature = self._narrow_host_to_pin(feature_group_class, feature, match.filter_feature)
                survivor = next(
                    f for f in self.feature_group_collection[feature_group_class] if f == match.filter_feature
                )
                self.filter_ties.append((feature.uuid, survivor.uuid))
                # The stored filter needs its own Feature: planner rewrites of the queue twin must not shift its hash.
                # handle_filter_feature copies via Feature.__copy__, which owns the containers that decide the hash.
                # The copy keeps the queue twin's uuid on purpose: nothing reads the stored filter feature's uuid.
                match.filter_feature = match.handle_filter_feature(match.filter_feature)
                self.global_filter.add_filter_to_collection(feature_group_class, feature.name, match)

            # After the loop, so the recorded filters are the renamed ones.
            self.global_filter.record_probe(feature_group_class, feature.name, feature.uuid, matched_filters)

    def _narrow_host_to_pin(
        self, feature_group_class: type[FeatureGroup], host: Feature, filter_feature: Feature
    ) -> Feature:
        """A pinned filter moves its host onto the pin; on a collision the host merges into the equal feature."""
        pin = filter_feature.compute_frameworks
        if not pin or host.compute_frameworks == pin:
            return host
        collection = self.feature_group_collection[feature_group_class]
        if not any(f is host for f in collection):
            return host
        collection.discard(host)
        host.compute_frameworks = set(pin)
        existing = next((f for f in collection if f == host), None)
        if existing is None:
            host.framework_pinned = True
            collection.add(host)
            return host
        self._merge_host_into(existing, host)
        existing.framework_pinned = True
        return existing

    def _fold_into_narrower_twins(self) -> None:
        """Folds each feature into its unique minimal narrower-framework twin; decided from a snapshot, order-free."""
        folds: list[tuple[type[FeatureGroup], Feature, Feature]] = []
        for group_class, collection in self.feature_group_collection.items():
            buckets: dict[int, list[tuple[Feature, frozenset[type[ComputeFramework]]]]] = defaultdict(list)
            for member in collection:
                if member.compute_frameworks is not None:
                    buckets[member.hash_ignoring_compute_frameworks()].append(
                        (member, frozenset(member.compute_frameworks))
                    )
            for bucket in buckets.values():
                for host, host_cf in bucket:
                    narrower = [
                        (t, t_cf)
                        for t, t_cf in bucket
                        if t is not host and t_cf < host_cf and t.equals_ignoring_compute_frameworks(host)
                    ]
                    minimal = [t for t, t_cf in narrower if not any(o_cf < t_cf for _, o_cf in narrower)]
                    if len(minimal) == 1 and self._same_matched_filters(host, minimal[0], group_class):
                        folds.append((group_class, host, minimal[0]))
        if not folds:
            return
        folds.sort(
            key=lambda fold: (
                fold[0].__module__,
                fold[0].__qualname__,
                str(fold[1].name),
                sorted(cf.__qualname__ for cf in fold[1].compute_frameworks or ()),
            )
        )
        fold_map = {host.uuid: survivor.uuid for _, host, survivor in folds}
        for group_class, host, survivor in folds:
            collection = self.feature_group_collection[group_class]
            self.feature_group_collection[group_class] = {f for f in collection if f is not host}
            self._merge_host_into(survivor, host)
            survivor.framework_pinned = survivor.framework_pinned or host.framework_pinned
            if self.global_filter is not None:
                self.global_filter.probes.pop((group_class, host.name, host.uuid), None)
        ties = [(fold_map.get(a, a), fold_map.get(b, b)) for a, b in self.filter_ties]
        self.filter_ties = list(dict.fromkeys(ties))

    def _same_matched_filters(self, host: Feature, target: Feature, group_class: type[FeatureGroup]) -> bool:
        """True when host and target matched the same global filters."""
        if self.global_filter is None:
            return True
        probes = self.global_filter.probes

        def ids(f: Feature) -> set[UUID]:
            return {sf.uuid for sf in probes.get((group_class, f.name, f.uuid), set())}

        return ids(host) == ids(target)

    def _merge_host_into(self, existing: Feature, host: Feature) -> None:
        """Folds a displaced host into its equal feature, as the duplicate path of add_feature_to_collection does."""
        existing.options.union_own_keys(host.options)
        for name, keys in host.consumer_attributions:
            existing.add_consumer_attribution(name, keys)
        if host.initial_requested_data:
            existing.initial_requested_data = True
        merged = set(self.specialized_from.get(existing.uuid, ())) | set(self.specialized_from.pop(host.uuid, ()))
        if merged:
            self.specialized_from[existing.uuid] = tuple(sorted(merged, key=_candidate_sort_key))
        if host.uuid in self.resolved_input_feature_names:
            self.resolved_input_feature_names.setdefault(existing.uuid, self.resolved_input_feature_names[host.uuid])
            del self.resolved_input_feature_names[host.uuid]
        self._declared_options_by_uuid.pop(host.uuid, None)
        host_parents = self.feature_link_parents.pop(host.uuid, set())
        self.feature_link_parents[existing.uuid] |= host_parents - {existing.uuid}
        for parents in self.feature_link_parents.values():
            if host.uuid in parents:
                parents.discard(host.uuid)
                parents.add(existing.uuid)

    def add_feature_link_to_links(self, feature: Feature) -> None:
        """With this functionality, we can add links with a feature instead via mloda API."""

        if feature.link is None:
            return

        if self.links is not None and feature.link in self.links:
            return

        candidate = {feature.link} if self.links is None else self.links | {feature.link}
        LinkValidator.validate_links(candidate)
        self.links = candidate

    def add_feature_to_collection(
        self,
        feature_group_class: type[FeatureGroup],
        feature: Feature,
        child_uuid: UUID | None,
        if_index_feature: bool = False,
        specialized_from: tuple[type[FeatureGroup], ...] = (),
    ) -> bool:
        # Materialize declared defaults at intake: default-equivalent twins become equal and merge
        # via the duplicate path below; identity no-op without concrete defaults.
        declared_options = feature.options
        memo_key = (feature_group_class, id(declared_options))
        entry = self._intake_options_memo.get(memo_key)
        if entry is None:
            entry = (declared_options, feature_group_class.options_with_defaults(declared_options))
            self._intake_options_memo[memo_key] = entry
        feature.options = entry[1]
        feature_collection = self.feature_group_collection[feature_group_class]

        if feature not in feature_collection:
            self.add_feature_link_to_links(feature)

            self.feature_link_parents[feature.uuid] = set()
            feature_collection.add(feature)
            self._declared_options_by_uuid[feature.uuid] = declared_options
            if specialized_from:
                self.specialized_from[feature.uuid] = specialized_from
            return True

        existing_feature = next((f for f in feature_collection if feature == f), None)

        if existing_feature is not None:
            if specialized_from:
                merged = set(self.specialized_from.get(existing_feature.uuid, ())) | set(specialized_from)
                self.specialized_from[existing_feature.uuid] = tuple(sorted(merged, key=_candidate_sort_key))
            existing_feature.options.union_own_keys(feature.options)
            for name, keys in feature.consumer_attributions:
                existing_feature.add_consumer_attribution(name, keys)
            self._warn_on_default_equivalent_merge(feature, declared_options, existing_feature)
            # Propagate the requested flag: filter twins must not displace requested output columns (issue #712).
            if feature.initial_requested_data and not existing_feature.initial_requested_data:
                existing_feature.initial_requested_data = True

            # An index twin is never a graph parent: wiring only repeat intakes made its position order-dependent.
            if child_uuid and not if_index_feature:
                self._update_feature_link_parents(child_uuid, feature.uuid, existing_feature.uuid)

        return False

    def _warn_on_default_equivalent_merge(
        self, feature: Feature, declared_options: Options, existing_feature: Feature
    ) -> None:
        """Warn when a merge holds only post-materialization: the declared (pre-default) options differ."""
        survivor_declared = self._declared_options_by_uuid.get(existing_feature.uuid, existing_feature.options)
        # Reached only when the feature equality probe matched, so cyclic values arrive here too.
        if _deep_equal(declared_options.group, survivor_declared.group) and _deep_equal(
            declared_options.context, survivor_declared.context
        ):
            return
        logger.warning(
            f"Feature '{feature.name}' was requested twice with default-equivalent options: the requests "
            f"differ only in explicitly-set declared defaults and merged into one feature at intake. "
            f"Deduplicate the request; dependency declaration follows the first-listed request."
        )

    def _update_feature_link_parents(self, child_uuid: UUID, original_uuid: UUID, wanted_uuid: UUID) -> None:
        """Points the child at the surviving feature instead of its merged duplicate."""
        self.feature_link_parents[child_uuid].discard(original_uuid)
        self.feature_link_parents[child_uuid].add(wanted_uuid)

    def _handle_input_features_recursion(
        self,
        feature_group_class: type[FeatureGroup],
        uuid: UUID,
        options: Options,
        feature_name: FeatureName,
        parent_domain: str | None = None,
        depth: int = 0,
        path: tuple[str, ...] = (),
    ) -> frozenset[str] | None:
        """Handles recursion for input features of a feature group."""
        feature_group = feature_group_class()

        # options = deepcopy(options)

        try:
            input_features = feature_group.input_features(options, feature_name)
        except NotImplementedError:  # This means, it is a root feature.
            input_features = None

        if not input_features:
            return None

        features = Features(list(input_features), child_options=options, child_uuid=uuid, parent_domain=parent_domain)
        consumer_name = feature_group_class.get_class_name()
        consumer_property_keys = self._property_mapping_keys(feature_group_class)
        for input_feature in features.collection:
            forwarded_declared_keys = input_feature.forwarded_group_keys & consumer_property_keys
            input_feature.add_consumer_attribution(consumer_name, forwarded_declared_keys)
        if features.child_uuid is None:
            raise ValueError(f"Features {features} has no parent uuid although it should have one.")
        self.feature_link_parents[features.child_uuid] = features.parent_uuids
        self.setup_features_recursion(
            features, requested=False, depth=depth + 1, consumer=consumer_name, path=(*path, str(feature_name))
        )
        return frozenset(str(f.name) for f in features.collection)

    def set_compute_framework(
        self,
        feature: Feature,
        compute_frameworks: set[type[ComputeFramework]],
        feature_group_class: type[FeatureGroup],
    ) -> Feature:
        """
        This function ensures that the feature always has a compute framework set!
        """
        if feature.compute_frameworks:
            if not feature.compute_frameworks & compute_frameworks:
                raise ValueError(
                    f"Feature {feature.name} does not support compute framework {feature.compute_frameworks}."
                )
        else:
            # Hash-safe only because this runs before add_feature_to_collection stores the feature.
            feature.compute_frameworks = self._drop_unconnected_required(
                feature, compute_frameworks, feature_group_class
            )
            dropped = frozenset(compute_frameworks - feature.compute_frameworks)
            if dropped:
                self.connection_dropped[feature.uuid] = dropped
        return feature

    def _connected_frameworks(self) -> frozenset[type[ComputeFramework]]:
        """Dropped frameworks for which the DataAccessCollection holds exactly one matching connection."""
        dac = self.data_access_collection
        if dac is None:
            return frozenset()
        return frozenset(
            cfw
            for cfw in set().union(*self.connection_dropped.values())
            if sum(cfw._connection_matches(conn) for conn in dac.connections.values()) == 1
        )

    @staticmethod
    def _drop_unconnected_required(
        feature: Feature,
        compute_frameworks: set[type[ComputeFramework]],
        feature_group_class: type[FeatureGroup],
    ) -> set[type[ComputeFramework]]:
        """Drop REQUIRED frameworks lacking a matching connection in the options; keep the set if none would remain."""
        conn = feature.options.get(feature_group_class.get_class_name())
        kept = {
            cfw
            for cfw in compute_frameworks
            if cfw.connection_requirement() is not ConnectionRequirement.REQUIRED or cfw._connection_matches(conn)
        }
        return kept or compute_frameworks

    def set_data_type(self, feature: Feature, feature_group_class: type[FeatureGroup]) -> DataType | None:
        fg_data_type = feature_group_class.return_data_type_rule(feature)
        if feature.data_type and fg_data_type:
            if feature.data_type != fg_data_type:
                raise ValueError(
                    f"Feature {feature.name} has a data type mismatch with feature group {feature_group_class}."
                )
            return fg_data_type
        return fg_data_type or feature.data_type
