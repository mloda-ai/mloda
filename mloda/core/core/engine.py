from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import replace
import logging
from typing import Any, cast
from uuid import UUID
import uuid

from mloda.core.abstract_plugins.components.index.add_index_feature import create_index_feature
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.input_data.reader_probe import (
    ReaderProbeScope,
    SourceProbeOutcome,
    probe_source_group,
    reader_probe_scope,
)
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.prepare.accessible_plugins import EnvironmentPreconditionError, PreFilterPlugins
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.data_access_collection import (
    DataAccessCollection,
    resolve_connection_handle,
)
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
from mloda.core.abstract_plugins.implementation import (
    accessible_implementations,
    current_implementations,
    serving_implementation,
)
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.read_file_fg import ReadFileFG
from mloda.core.prepare.resolve_implementation_pins import ResolveImplementationPins
from mloda.core.prepare.resolution_failure_renderer import _candidate_sort_key
from mloda.core.prepare.stable_text import stable_text
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.core.prepare.input_chain import (
    EnvironmentKey,
    declared_inputs,
    input_resolution_scope,
    normalize_input_name,
    resolution_environment,
)
from mloda.core.prepare.resolution_types import (
    EvaluationResult,
    ResolutionRecord,
)
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.core.abstract_plugins.feature_group import FeatureGroup, format_feature_group_class
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.hashable_dict import _deep_equal, _deep_hashable
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.abstract_plugins.components.match_hook import call_match_hook
from mloda.core.abstract_plugins.components.validators.link_validator import LinkValidator


logger = logging.getLogger(__name__)


def _typed_reuse_key(feature: Feature) -> tuple[str, str]:
    return str(feature.data_type), stable_text(feature.child_options)


class Engine:
    def __init__(
        self,
        features: Features,
        compute_frameworks: set[type[ComputeFramework]],
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None = None,
        global_filter: GlobalFilter | None = None,
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
        self.connection_reasons: dict[UUID, dict[type[ComputeFramework], str]] = {}
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
        pre_filter = PreFilterPlugins(compute_frameworks, plugin_collector)
        self.accessible_plugins = pre_filter.get_accessible_plugins()
        self.implementations = pre_filter.implementations
        self._connection_frameworks = sorted(
            {
                cfw
                for cfws in self.accessible_plugins.values()
                for cfw in cfws
                if cfw.connection_requirement() is not ConnectionRequirement.NONE
            },
            key=lambda cfw: cfw.get_class_name(),
        )
        self.excluded_feature_groups = pre_filter.excluded_feature_groups
        # get links
        LinkValidator.validate_links(links)
        self.links = set(links) if links is not None else None

        self.plugin_collector = plugin_collector

        self.data_access_collection = data_access_collection
        self.column_ordering = column_ordering
        self.request_feature_order: list[str] = [str(f.name) for f in features]
        self._dual_consumption_warned: set[tuple[str, str, frozenset[str]]] = set()
        self._property_mapping_keys_cache: dict[type[FeatureGroup], frozenset[str]] = {}
        # Intake materialization memo: the strong source reference keeps its id stable, so features
        # sharing one pre-default Options share one effective Options.
        self._intake_options_memo: dict[tuple[type[FeatureGroup], int, Any], tuple[Options, Options]] = {}
        # Declared (pre-default) options per surviving feature uuid, for default-equivalent merge warnings.
        self._declared_options_by_uuid: dict[UUID, Options] = {}
        # Per feature uuid, _handle_input_features_recursion's result (None: root; injected filter/index: no entry).
        self.resolved_input_feature_names: dict[UUID, frozenset[str] | None] = {}
        # Per feature uuid, the sorted columns its declared inputs bind to in the incoming data.
        self.input_feature_columns: dict[UUID, tuple[str, ...]] = {}
        # Per feature uuid, the cascade_context_keys it had when its inputs were resolved.
        self._cascade_keys_at_resolution: dict[UUID, frozenset[str]] = {}
        # Per surviving feature uuid, the parents its winning group replaced, unioned over merged duplicates.
        self.specialized_from: dict[UUID, tuple[type[FeatureGroup], ...]] = {}
        self.resolution_records: list[ResolutionRecord] = []
        with run_match_cache(), accessible_implementations(self.implementations), input_resolution_scope(self):
            self.execution_planner = self.create_setup_execution_plan(features)
        if self.function_extender:
            self.run_context = replace(self.run_context, plugin_versions=self._resolve_plugin_versions())
        self.connections = self._resolve_connections()
        self.output_connection = self._resolve_output_connection()

    def _resolve_plugin_versions(self) -> dict[str, str | None] | None:
        """Resolves each planned feature group module's owning-distribution version at plan time, so hooks only read it."""
        versions = {
            step.feature_group.__module__: resolve_plugin_version(step.feature_group.__module__)
            for step in self.execution_planner
            if isinstance(step, FeatureGroupStep)
        }
        return versions or None

    def _resolve_output_connection(self) -> Any:
        """Resolve the output_framework connection, needed only when a requested step's result is converted."""
        if self.output_framework is None:
            return None
        output = self.output_framework
        converting = [
            step
            for step in self.execution_planner
            if isinstance(step, FeatureGroupStep)
            and step.features.get_initial_requested_features()
            and step.compute_framework.expected_data_framework() is not output.expected_data_framework()
        ]
        if not converting:
            return None
        dac = self.data_access_collection
        picked = [
            output.pick_connection_from_dac(dac, s.features.options) if dac is not None else None for s in converting
        ]
        connection = picked[0]
        if any(other is not connection for other in picked):
            handles = sorted({str(self._resolve_handle(output, s.features.options)[0]) for s in converting})
            raise ValueError(
                f"The results converted to output_framework {output.get_class_name()} select different connections: "
                + " and ".join(f"'{h}'" for h in handles)
                + "."
            )
        if connection is None and output.connection_requirement() is ConnectionRequirement.REQUIRED:
            labels = dict.fromkeys(
                f"{s.feature_group.get_class_name()} on {s.compute_framework.get_class_name()}" for s in converting
            )
            steps = ", ".join(labels)
            raise EnvironmentPreconditionError(
                f"The result of {steps} converts to output_framework {output.get_class_name()}, "
                "which needs a connection, so add a matching connection to the DataAccessCollection.",
                partial_records=self.resolution_records,
            )
        return connection

    def _resolve_connections(self) -> dict[str, Any]:
        """Map each handle a step uses to its DataAccessCollection connection.

        Raises EnvironmentPreconditionError when a converting hop into a REQUIRED-connection framework has none.
        """
        for tfs in self.execution_planner.tfs_collection.values():
            if (
                tfs.connection_handle is None
                and tfs.to_framework.connection_requirement() is ConnectionRequirement.REQUIRED
                and tfs.from_framework.expected_data_framework() is not tfs.to_framework.expected_data_framework()
            ):
                raise EnvironmentPreconditionError(
                    f"The transform from {tfs.from_feature_group.get_class_name()} on "
                    f"{tfs.from_framework.get_class_name()} to {tfs.to_feature_group.get_class_name()} on "
                    f"{tfs.to_framework.get_class_name()} needs a connection, so add a matching connection "
                    "to the DataAccessCollection.",
                    partial_records=self.resolution_records,
                )
        dac = self.data_access_collection
        if dac is None:
            return {}
        handles = {step.connection_handle for step in self.execution_planner if step.connection_handle is not None}
        return {handle: dac.connections[handle] for handle in sorted(handles)}

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
            connections=self.connections,
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

        match_extender = self.get_function_extender(ExtenderHook.INPUT_DATA_MATCH)
        plan = self.plan_context
        probe_scope = (
            None
            if match_extender is None
            else ReaderProbeScope(
                match_extender,
                self.run_context.plan_id,
                plan.tenant_id if plan else None,
                plan.project_id if plan else None,
                plan.principal if plan else None,
            )
        )
        with reader_probe_scope(probe_scope):
            self.setup_features_recursion(features)
        self._fold_into_narrower_twins()
        with reader_probe_scope(probe_scope):
            self._load_result_index_columns()

        if self.global_filter:
            self.global_filter.warn_on_unmatched_filters()

        graph_builder = BuildGraph(self.feature_link_parents, self.feature_group_collection)
        graph_builder.build_graph_from_feature_links()
        graph = graph_builder.graph

        pin_frameworks: dict[UUID, frozenset[type[ComputeFramework]]] = {}
        pinned: set[UUID] = set()
        if any(node.feature.implementation_pin is not None for node in graph.nodes.values()):
            pin_frameworks, pinned = ResolveImplementationPins(graph, self.implementations).resolve()

        # resolve graph into a queue
        resolver = ResolveGraph(
            graph,
            self.links,
            self.filter_ties,
            self.framework_positions,
            output_framework=self.output_framework,
            connection_dropped=self.connection_dropped,
            pin_frameworks=pin_frameworks,
            pinned=pinned,
            implementations=self.implementations,
            handle_of=self._resolve_handle,
            connection_reasons=self.connection_reasons,
        )
        resolver.create_initial_queue()

        resolver.set_nodes_per_feature_group()

        planned_queue = resolver.resolve_links()
        resolver.resolver_links.warn_on_unjoined_links()

        planned_queue = resolver.resolver_compute_framework.links(
            planned_queue, resolver.resolver_links.get_link_trekker()
        )

        if self.global_filter:
            # Setup still shifts a stored hash: a copied option Feature reaches the host's Feature via child_options.
            self.global_filter.rehash_stored_filters()

        execution_planner = ExecutionPlan(
            self.global_filter,
            self.resolved_input_feature_names,
            {
                uuid: tuple(f"{c.__module__}.{c.__qualname__}" for c in classes)
                for uuid, classes in self.specialized_from.items()
            },
            self.implementations,
            self.data_access_collection,
            input_feature_columns=self.input_feature_columns,
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
        self._raise_on_connection_in_options(feature)
        feature_group_class, compute_frameworks, result = self._identify_feature_group_and_frameworks(feature, depth)
        self.resolution_records.append(ResolutionRecord(str(feature.name), requested, result))
        self._warn_on_dual_option_consumption(feature, feature_group_class)
        feature_group = feature_group_class()

        self._set_feature_name(feature, feature_group, result.output_parts.get(feature_group_class))
        self._set_compute_framework_and_data_type(feature, compute_frameworks, feature_group_class)

        feature.result_index_columns = self._result_index_columns(feature, feature_group, feature_group_class)

        # Stash the declared pre-default options: dependency declaration and child inheritance observe
        # them; intake materialization canonicalizes default-equivalent twins.
        declared_options = feature.options
        added = self.add_feature_to_collection(
            feature_group_class,
            feature,
            features.child_uuid,
            specialized_from=result.specialized_from,
            check_cascade=True,
        )

        if added:
            self._cascade_keys_at_resolution[feature.uuid] = feature.options.cascade_context_keys
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

    def _set_feature_name(
        self, feature: Feature, feature_group: FeatureGroup, output_part: tuple[str, str] | None
    ) -> None:
        """Rename the feature to its base name and record the requested output part."""
        normalize_input_name(self._input_environment(), feature, feature_group, output_part)

    def _input_environment(self) -> EnvironmentKey:
        return resolution_environment(
            self.accessible_plugins, self.links, self.data_access_collection, self.excluded_feature_groups
        )

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
                excluded=self.excluded_feature_groups,
                _input_resolver=self._resolve_input_candidate,
                _input_depth=depth,
            )
        else:
            result = self._resolve_with_match_hook(extender, feature, depth)
        feature_group_class, compute_frameworks = next(iter(result.identified.items()))
        return feature_group_class, compute_frameworks, result

    def _resolve_input_candidate(self, feature: Feature, links: set[Link] | None, depth: int) -> EvaluationResult:
        """Use the run's match extender for an input attempt, without committing nodes or records."""
        extender = self.get_function_extender(ExtenderHook.FEATURE_GROUP_MATCHED)
        if extender is not None:
            return self._resolve_with_match_hook(extender, feature, depth, probe=True, links=links)
        return resolve_or_raise(
            feature,
            self.accessible_plugins,
            links,
            self.data_access_collection,
            excluded=self.excluded_feature_groups,
            _input_resolver=self._resolve_input_candidate,
            _input_depth=depth,
        )

    def _resolve_with_match_hook(
        self,
        extender: Extender,
        feature: Feature,
        depth: int,
        *,
        probe: bool = False,
        links: set[Link] | None = None,
    ) -> EvaluationResult:
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
                    links if probe else self.links,
                    self.data_access_collection,
                    partial_records=() if probe else self.resolution_records,
                    excluded=self.excluded_feature_groups,
                    _input_resolver=self._resolve_input_candidate,
                    _input_depth=depth,
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

    @staticmethod
    def _link_may_raise(link: Link, feature: Feature, discriminator: dict[str, Any] | None, one_sided: bool) -> bool:
        """A missing link key column is an error for a matched discriminator or a one-sided link the feature declares."""
        matched = bool(discriminator and Link.matches_discriminator(discriminator, feature.options))
        return matched or (one_sided and feature.link == link)

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
                    feature_group_class,
                    feature_group,
                    feature,
                    features.child_uuid,
                    link.left_index,
                    link,
                    self._link_may_raise(link, feature, link.left_discriminator, left != right),
                )
            if right:
                self._create_and_add_index_feature(
                    feature_group_class,
                    feature_group,
                    feature,
                    features.child_uuid,
                    link.right_index,
                    link,
                    self._link_may_raise(link, feature, link.right_discriminator, left != right),
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
                self._create_and_add_index_feature(
                    feature_group_class,
                    feature_group,
                    feature,
                    features.child_uuid,
                    index,
                    link,
                    self._link_may_raise(link, feature, link.left_discriminator, left != right),
                )

            if right and link.right_index == index:
                self._create_and_add_index_feature(
                    feature_group_class,
                    feature_group,
                    feature,
                    features.child_uuid,
                    index,
                    link,
                    self._link_may_raise(link, feature, link.right_discriminator, left != right),
                )

    def _create_and_add_index_feature(
        self,
        feature_group_class: type[FeatureGroup],
        feature_group: FeatureGroup,
        feature: Feature,
        child_uuid: UUID | None,
        index: Index,
        link: Link | None,
        may_raise: bool = False,
    ) -> None:
        """Creates and adds a new index feature; a link's key columns are probed through the INPUT_DATA_MATCH gate."""
        if link is not None:
            for column in index.index:
                outcome = self._probe_has_column(feature_group_class, feature, column, read=may_raise)
                if outcome is None:
                    break
                if outcome.refused is None:
                    if outcome.result is False:
                        raise ValueError(self._missing_link_column_message(feature_group_class, feature, column, link))
                else:
                    raise ValueError(
                        f"The link index column '{column}' of feature '{feature.name}' on "
                        f"{feature_group_class.get_class_name()} (link {link}) was refused at INPUT_DATA_MATCH: "
                        f"{outcome.refused}"
                    ) from outcome.error
        new_index_feature = create_index_feature(index, feature_group, feature)
        new_index_feature.link_key_only = link is not None
        self.add_feature_to_collection(feature_group_class, new_index_feature, child_uuid, True)
        if feature.uuid in self.connection_dropped:
            self.connection_dropped[new_index_feature.uuid] = self.connection_dropped[feature.uuid]
        if feature.uuid in self.connection_reasons:
            self.connection_reasons[new_index_feature.uuid] = self.connection_reasons[feature.uuid]

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
                survivor = self._find_existing_feature(
                    match.filter_feature, self.feature_group_collection[feature_group_class], allow_untyped_input=True
                )
                if survivor is None:
                    raise ValueError(f"Filter feature {match.filter_feature.name} is missing from its group.")
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
        if not pin:
            return host
        if host.compute_frameworks == pin:
            host.framework_pinned = True
            return host
        collection = self.feature_group_collection[feature_group_class]
        if not any(f is host for f in collection):
            return host
        collection.discard(host)
        host.compute_frameworks = set(pin)
        existing = self._find_existing_feature(host, collection)
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

    def _result_index_columns(
        self, feature: Feature, feature_group: FeatureGroup, feature_group_class: type[FeatureGroup]
    ) -> tuple[str, ...]:
        """The key columns this request keeps in its result frame: Feature.index, else the group's index_columns."""
        if not (feature.keep_index and feature.initial_requested_data):
            return ()
        if feature.index is not None:
            key = tuple(feature.index.index)
        else:
            key = tuple(dict.fromkeys(c for index in feature_group.index_columns() or [] for c in index.index))
        if not key:
            raise ValueError(
                f"Feature '{feature.name}' sets keep_index, but no key is known for it in feature group "
                f"{format_feature_group_class(feature_group_class)}; set Feature.index or declare index_columns()."
            )
        return key

    def _missing_link_column_message(self, group: type[FeatureGroup], feature: Feature, column: str, link: Link) -> str:
        head = f"The link index column '{column}' of feature '{feature.name}' (link {link}) is in none of the sources"
        if feature.input_data_match is None or not issubclass(group, SourceFeatureGroup):
            return head
        match = feature.input_data_match[1]
        pointed = group.is_pointed(str(feature.name), feature.options, self.data_access_collection)
        source = scrub_credentials(match.source) if pointed else group._hidden_source(match)
        return f"{head} of {group.get_class_name()}: {source}"

    def _probe_has_column(
        self, group: type[FeatureGroup], feature: Feature, column: str, read: bool = True
    ) -> SourceProbeOutcome | None:
        """Asks the matched source whether it has the column, through the INPUT_DATA_MATCH gate."""
        if feature.input_data_match is None or not issubclass(group, SourceFeatureGroup):
            return None
        reader, match = feature.input_data_match
        return probe_source_group(
            group,
            lambda: None if not read or reader._declared_name(column) else reader.has_column(match, column),
            self.data_access_collection,
            column,
            feature.options._fork(),
        )

    def _root_column_rank(self, group: type[FeatureGroup], root: Feature, column: str) -> int | None:
        """0 if the root may confirm the column, 1 if its source cannot list columns, None if excluded."""
        outcome = self._probe_has_column(group, root, column)
        if outcome is None:
            return 0
        if outcome.refused is not None or outcome.result is False:
            return None
        declared = root.input_data_match is not None and root.input_data_match[0]._declared_name(column)
        return 1 if outcome.result is None and not declared else 0

    def _load_result_index_columns(self) -> None:
        """Inject each kept key column into the root features of the features that keep it."""
        owners = {f.uuid: (group, f) for group, fs in self.feature_group_collection.items() for f in fs}
        for _, feature in list(owners.values()):
            if not feature.result_index_columns:
                continue
            roots: dict[UUID, tuple[type[FeatureGroup], Feature]] = {}
            seen: set[UUID] = set()
            todo = [feature.uuid]
            while todo:
                current = todo.pop()
                if current in seen:
                    continue
                seen.add(current)
                if not self.resolved_input_feature_names.get(current):
                    roots[current] = owners[current]
                todo.extend(self.feature_link_parents.get(current, ()))
            visited = {str(owners[u][1].name) for u in seen}
            ordered = sorted(roots.values(), key=lambda gr: (format_feature_group_class(gr[0]), str(gr[1].name)))
            for column in feature.result_index_columns:
                if column in visited:
                    continue
                match = None
                for group, root in ordered:
                    rank = self._root_column_rank(group, root, column)
                    if rank is None or (rank == 1 and match is not None):
                        continue
                    if not call_match_hook(group, column, root.options._fork(), self.data_access_collection).matched:
                        continue
                    if rank == 0:
                        match = (group, root)
                        break
                    if match is None:
                        match = (group, root)
                if match is None:
                    raise ValueError(
                        f"Feature '{feature.name}' keeps index column '{column}', but no input feature group "
                        f"provides it; the column must be matched by one of the feature's root input feature groups."
                    )
                self._create_and_add_index_feature(match[0], match[0](), match[1], None, Index((column,)), None)

    def _merge_host_into(self, existing: Feature, host: Feature) -> None:
        """Folds a displaced host into its equal feature, as the duplicate path of add_feature_to_collection does."""
        existing.options.union_own_keys(host.options)
        existing.absorb_parts(host)
        if not host.link_key_only:
            existing.link_key_only = False
        existing.absorb_result_index_columns(host)
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
        if host.uuid in self.input_feature_columns:
            self.input_feature_columns.setdefault(existing.uuid, self.input_feature_columns[host.uuid])
            del self.input_feature_columns[host.uuid]
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
        *,
        check_cascade: bool = False,
    ) -> bool:
        # Materialize declared defaults at intake: default-equivalent twins become equal and merge
        # via the duplicate path below; name-bound values fill too (into context); identity no-op otherwise.
        declared_options = feature.options
        bindings = feature_group_class.name_bindings(feature.name)
        memo_key = (feature_group_class, id(declared_options), _deep_hashable(bindings))
        entry = self._intake_options_memo.get(memo_key)
        if entry is None:
            entry = (
                declared_options,
                feature_group_class.options_with_defaults(declared_options, bindings),
            )
            self._intake_options_memo[memo_key] = entry
        feature.options = entry[1]
        feature_collection = self.feature_group_collection[feature_group_class]

        existing_feature = self._find_existing_feature(feature, feature_collection, allow_untyped_input=True)
        if (
            existing_feature is not None
            and existing_feature != feature
            and self._cascade_keys_at_resolution.get(existing_feature.uuid, feature.options.cascade_context_keys)
            != feature.options.cascade_context_keys
        ):
            existing_feature = None

        if existing_feature is None:
            self.add_feature_link_to_links(feature)

            self.feature_link_parents[feature.uuid] = set()
            feature_collection.add(feature)
            self._declared_options_by_uuid[feature.uuid] = declared_options
            if specialized_from:
                self.specialized_from[feature.uuid] = specialized_from
            return True

        if existing_feature is not None:
            if check_cascade and self.resolved_input_feature_names.get(existing_feature.uuid):
                resolved_keys = self._cascade_keys_at_resolution.get(
                    existing_feature.uuid, existing_feature.options.cascade_context_keys
                )
                if resolved_keys != feature.options.cascade_context_keys:
                    raise ValueError(
                        f"Feature '{feature.name}' is requested with different cascade_context_keys: "
                        f"{sorted(resolved_keys)} and {sorted(feature.options.cascade_context_keys)}. "
                        f"Its inputs were already resolved, so the cascade would depend on request order. "
                        f"Declare the same cascade_context_keys on every request that reaches the feature, "
                        f"and list the cascading request first when a Feature object is shared between requests."
                    )
            if existing_feature.data_type is None and feature.data_type is not None:
                # Other Features may hold the old object in their identity. Copy it without changing the graph UUID.
                promoted = copy(existing_feature)
                promoted.data_type = feature.data_type
                feature_collection = {promoted if f is existing_feature else f for f in feature_collection}
                self.feature_group_collection[feature_group_class] = feature_collection
                existing_feature = promoted
            if specialized_from:
                merged = set(self.specialized_from.get(existing_feature.uuid, ())) | set(specialized_from)
                self.specialized_from[existing_feature.uuid] = tuple(sorted(merged, key=_candidate_sort_key))
            existing_feature.options.union_own_keys(feature.options)
            existing_feature.absorb_parts(feature)
            if not feature.link_key_only:
                existing_feature.link_key_only = False
            existing_feature.absorb_result_index_columns(feature)
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

    def _find_existing_feature(
        self, feature: Feature, feature_collection: Iterable[Feature], *, allow_untyped_input: bool = False
    ) -> Feature | None:
        """Prefer an equal twin; optionally let a typed request satisfy an untyped input."""
        if feature in feature_collection:
            exact = next((f for f in feature_collection if feature == f), None)
            if exact is not None:
                return exact
        compatible = None
        for f in feature_collection:
            if (
                f is feature
                or f.options.cascade_context_keys != feature.options.cascade_context_keys
                or f.requested_parts != feature.requested_parts
                or f.parts_only != feature.parts_only
            ):
                continue
            if f.child_options_differ(feature) and f.equals_ignoring_child_options(feature):
                return f
            if (
                allow_untyped_input
                and (
                    (
                        f.initial_requested_data
                        and f.data_type is not None
                        and not feature.initial_requested_data
                        and feature.data_type is None
                    )
                    or (
                        feature.initial_requested_data
                        and feature.data_type is not None
                        and not f.initial_requested_data
                        and f.data_type is None
                    )
                )
                # Cross-type reuse needs completed inputs, not a placeholder or an ancestor still resolving.
                and f.uuid in self.resolved_input_feature_names
                and f.equals_ignoring_child_options(feature, include_data_type=False)
            ):
                if compatible is None or _typed_reuse_key(f) < _typed_reuse_key(compatible):
                    compatible = f
        return compatible

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
            f"differ only in explicitly-set declared defaults or name-bound values and merged into one feature at intake. "
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
        try:
            input_features = declared_inputs(
                self._input_environment(), feature_group_class, options, feature_name, materialize=True
            )
        except NotImplementedError:  # This means, it is a root feature.
            input_features = None

        if not input_features:
            return None

        for entry in input_features:
            if isinstance(entry, Feature) and entry.connection is not None:
                raise ValueError(
                    f"{feature_group_class.get_class_name()}.input_features returned '{entry.name}' with a "
                    "connection; register it in the DataAccessCollection and set the 'connection_handle' option."
                )

        ordered: list[Feature | str] = list(input_features)
        if isinstance(input_features, (set, frozenset)):  # set order is hash-seed dependent
            ordered.sort(key=lambda f: str(f.name) if isinstance(f, Feature) else str(f))
        # Captured before recursion renames digit-only ~N part requests to their base (issue #2031).
        declared_names = {str(entry) if isinstance(entry, str) else str(entry.name) for entry in ordered}
        declared_order = [str(entry) if isinstance(entry, str) else str(entry.name) for entry in ordered]
        features = Features(ordered, child_options=options, child_uuid=uuid, parent_domain=parent_domain)
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
        columns: set[str] = set()
        for declared, resolved in zip(declared_order, features.collection, strict=True):
            renamed = str(resolved.name)
            columns.add(declared if renamed == declared or declared.startswith(renamed + "~") else renamed)
        if columns:
            self.input_feature_columns[uuid] = tuple(sorted(columns))
        resolved_names = {str(f.name) for f in features.collection}
        evicted = {FeatureGroup.get_column_base_feature(name) for name in declared_names} - declared_names
        return frozenset((resolved_names - evicted) | declared_names)

    def set_compute_framework(
        self,
        feature: Feature,
        compute_frameworks: set[type[ComputeFramework]],
        feature_group_class: type[FeatureGroup],
    ) -> Feature:
        """
        This function ensures that the feature always has a compute framework set!
        """
        infeasible = self._infeasible_file_frameworks(compute_frameworks, feature_group_class)
        if feature.compute_frameworks:
            if not feature.compute_frameworks & compute_frameworks:
                raise ValueError(
                    f"Feature {feature.name} does not support compute framework {feature.compute_frameworks}."
                )
            for cfw in feature.compute_frameworks & compute_frameworks:
                if cfw in infeasible:
                    raise ValueError(f"Feature {feature.name} cannot use {infeasible[cfw]}")
        else:
            if infeasible and not compute_frameworks - infeasible.keys():
                raise ValueError(f"Feature {feature.name} cannot use {'; '.join(sorted(infeasible.values()))}")
            # Hash-safe only because this runs before add_feature_to_collection stores the feature.
            feature.compute_frameworks = self._drop_unconnected_required(
                feature, compute_frameworks - infeasible.keys(), feature_group_class
            )
            for cfw, reason in infeasible.items():
                self.connection_reasons.setdefault(feature.uuid, {})[cfw] = reason
            dropped = frozenset(compute_frameworks - feature.compute_frameworks - infeasible.keys())
            if dropped:
                self.connection_dropped[feature.uuid] = dropped
        return feature

    @staticmethod
    def _infeasible_file_frameworks(
        compute_frameworks: set[type[ComputeFramework]], feature_group_class: type[FeatureGroup]
    ) -> dict[type[ComputeFramework], str]:
        """REQUIRED frameworks a neutral FileSource load of a file root cannot reach, with the reason."""
        if not (
            isinstance(feature_group_class, type)
            and issubclass(feature_group_class, ReadFileFG)
            and not feature_group_class.overrides_load_neutral(ReadFileFG)
        ):
            return {}
        if not any(cfw.connection_requirement() is ConnectionRequirement.REQUIRED for cfw in compute_frameworks):
            return {}
        transformer = ComputeFrameworkTransformer()
        implementations = list(current_implementations())
        infeasible: dict[type[ComputeFramework], str] = {}
        for cfw in compute_frameworks:
            if cfw.connection_requirement() is not ConnectionRequirement.REQUIRED:
                continue
            if serving_implementation(feature_group_class, cfw, implementations, stop_after="load_neutral"):
                continue
            target = cfw.expected_data_framework()
            if not ComputeFramework.descriptor_reaches(transformer, FileSource, target):
                infeasible[cfw] = (
                    f"{cfw.get_class_name()}: {ComputeFramework.no_transformer_reason(FileSource, target)}"
                )
        return infeasible

    def _raise_on_connection_in_options(self, feature: Feature) -> None:
        """A connection object in the options is no longer read; it belongs in the DataAccessCollection."""
        if not self._connection_frameworks:
            return
        for scope in (feature.options.group, feature.options.context):
            for key, value in scope.items():
                for cfw in self._connection_frameworks:
                    if cfw._connection_matches(value):
                        raise ValueError(
                            f"Option '{key}' of feature '{feature.name}' holds a {cfw.get_class_name()} connection. "
                            "Put it in the DataAccessCollection and select it with the 'connection_handle' option."
                        )

    def _resolve_handle(self, framework: type[ComputeFramework], options: Any) -> tuple[str | None, bool]:
        """The DataAccessCollection handle for a framework and whether the options named it."""
        handle, explicit, _ = resolve_connection_handle(framework, self.data_access_collection, options)
        return handle, explicit

    def _drop_unconnected_required(
        self,
        feature: Feature,
        compute_frameworks: set[type[ComputeFramework]],
        feature_group_class: type[FeatureGroup],
    ) -> set[type[ComputeFramework]]:
        """Drop REQUIRED frameworks without a connection handle; keep the set if none would remain."""
        kept: set[type[ComputeFramework]] = set()
        for cfw in compute_frameworks:
            if cfw.connection_requirement() is not ConnectionRequirement.REQUIRED:
                kept.add(cfw)
                continue
            handle, _, reason = resolve_connection_handle(cfw, self.data_access_collection, feature.options)
            if handle is not None:
                kept.add(cfw)
            elif reason is not None:
                self.connection_reasons.setdefault(feature.uuid, {})[cfw] = f"{cfw.get_class_name()}: {reason}"
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
