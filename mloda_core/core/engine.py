from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, Set, Type
from uuid import UUID
import uuid
from mloda_core.abstract_plugins.components.index.add_index_feature import create_index_feature
from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.filter.global_filter import GlobalFilter
from mloda_core.prepare.accessible_plugins import PreFilterPlugins
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.prepare.execution_plan import ExecutionPlan
from mloda_core.prepare.graph.build_graph import BuildGraph
from mloda_core.prepare.resolve_graph import ResolveGraph
from mloda_core.runtime.run import Runner
from mloda_core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda_core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.link import Link


class Engine:
    def __init__(
        self,
        features: Features,
        compute_frameworks: Set[Type[ComputeFrameWork]],
        links: Optional[Set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_input_data_collection: Optional[ApiInputDataCollection] = None,
        plugin_collector: Optional[PlugInCollector] = None,
    ) -> None:
        # setup variables which track the primary sources and the compute platforms
        self.feature_group_collection: Dict[Type[AbstractFeatureGroup], Set[Feature]] = defaultdict(set)

        # use global filters
        self.global_filter = global_filter

        # Tracks feature relation to its parents
        self.feature_link_parents: Dict[UUID, Set[UUID]] = defaultdict(set)

        # get accessible feature groups and their compute platforms
        self.accessible_plugins = PreFilterPlugins(compute_frameworks, plugin_collector).get_accessible_plugins()
        # get links
        Link.validate(links)
        self.links = links

        # set api input collection if relevant
        self.api_input_data_collection = api_input_data_collection

        # TODO
        self.plugin_collector = plugin_collector
        self.copy_compute_frameworks = deepcopy(compute_frameworks)

        self.data_access_collection = data_access_collection
        self.execution_planner = self.create_setup_execution_plan(features)

    def compute(self, flight_server: Optional[ParallelRunnerFlightServer] = None) -> Runner:
        runner = Runner(self.execution_planner, flight_server)
        if isinstance(runner, Runner):
            return runner
        raise ValueError("Runner setup failed.")

    def create_setup_execution_plan(self, features: Features) -> ExecutionPlan:
        self.setup_features_recursion(features)

        graph_builder = BuildGraph(self.feature_link_parents, self.feature_group_collection)
        graph_builder.build_graph_from_feature_links()
        graph = graph_builder.graph

        # resolve graph into a queue
        resolver = ResolveGraph(graph, self.links)
        resolver.create_initial_queue()

        resolver.set_nodes_per_feature_group()

        planned_queue = resolver.resolve_links()

        planned_queue = resolver.resolver_compute_framework.links(
            planned_queue, resolver.resolver_links.get_link_trekker()
        )
        # need to resolve compute frameworks
        # 1) first run -> joins: adjust to the provided links in the planned queue. Maybe also in the trekker? link trekker should have child uuid plus link
        # 2) second run -> reduce by most common compute framework

        execution_planner = ExecutionPlan(self.global_filter, self.api_input_data_collection)
        execution_planner.create_execution_plan(planned_queue, graph, resolver.resolver_links.get_link_trekker())
        return execution_planner

    def setup_features_recursion(self, features: Features) -> None:
        for feature in features:
            self.accessible_plugins = PreFilterPlugins(
                self.copy_compute_frameworks, self.plugin_collector
            ).get_accessible_plugins()
            feature_group_class, compute_frameworks = IdentifyFeatureGroupClass(
                feature, self.accessible_plugins, self.links, self.data_access_collection
            ).get()

            feature_group = feature_group_class()

            feature.name = feature_group.set_feature_name(feature.options, feature.name)

            feature = self.set_compute_framework(feature, compute_frameworks)
            feature.data_type = self.set_data_type(feature, feature_group_class)

            added = self.add_feature_to_collection(feature_group_class, feature, features.child_uuid)

            if added:
                self.call_recursion_by_using_input_feature_definition(
                    feature_group_class, feature.uuid, feature.options, feature.name
                )

            if self.global_filter:
                self.add_filter_feature(feature_group_class, feature_group, feature, features)

            if feature_group.index_columns():
                self.add_index_feature(feature_group_class, feature_group, feature, features)

    def add_index_feature(
        self,
        feature_group_class: Type[AbstractFeatureGroup],
        feature_group: AbstractFeatureGroup,
        feature: Feature,
        features: Features,
    ) -> None:
        indexes = feature_group_class.index_columns()
        if indexes is None:
            raise ValueError(f"Feature group {feature_group_class} has no indexes defined.")

        if self.links is None:
            return

        for index in indexes:
            for link in self.links:
                if link.left_feature_group == feature_group_class:
                    if link.left_index == index:
                        new_index_feature = create_index_feature(index, feature_group, feature)
                        self.add_feature_to_collection(
                            feature_group_class, new_index_feature, features.child_uuid, True
                        )

                if link.right_feature_group == feature_group_class:
                    if link.right_index == index:
                        new_index_feature = create_index_feature(index, feature_group, feature)
                        self.add_feature_to_collection(
                            feature_group_class, new_index_feature, features.child_uuid, True
                        )

    def add_filter_feature(
        self,
        feature_group_class: Type[AbstractFeatureGroup],
        feature_group: AbstractFeatureGroup,
        feature: Feature,
        features: Features,
    ) -> None:
        if self.global_filter:
            matched_filters = self.global_filter.identity_matched_filters(
                feature_group_class, feature, self.data_access_collection
            )

            for match in matched_filters:
                match.filter_feature.name = feature_group.set_feature_name(
                    match.filter_feature.options, match.filter_feature.name
                )
                # We assign a new UUID to the filter feature to ensure
                # it is treated as a separate instance from the original filter feature
                match.filter_feature.uuid = uuid.uuid4()
                self.global_filter.add_filter_to_collection(feature_group_class, feature.name, match)
                self.add_feature_to_collection(feature_group_class, match.filter_feature, features.child_uuid)

    def add_feature_link_to_links(self, feature: Feature) -> None:
        """With this functionality, we can add links with a feature instead via mloda API."""

        if feature.link is None:
            return

        if self.links is None:
            self.links = {feature.link}
        else:
            self.links.add(feature.link)

    def add_feature_to_collection(
        self,
        feature_group_class: Type[AbstractFeatureGroup],
        feature: Feature,
        child_uuid: Optional[UUID],
        if_index_feature: bool = False,
    ) -> bool:
        feature_collection = self.feature_group_collection[feature_group_class]

        if feature not in feature_collection:
            self.add_feature_link_to_links(feature)

            self.feature_link_parents[feature.uuid] = set()
            feature_collection.add(feature)
            return True

        if child_uuid:
            # Find the wanted_uuid in feature_collection
            wanted_uuid = next((f.uuid for f in feature_collection if feature == f), None)

            if wanted_uuid is not None:
                if not if_index_feature:
                    self.feature_link_parents[child_uuid].remove(feature.uuid)
                    self.feature_link_parents[child_uuid].add(wanted_uuid)
                else:
                    self.feature_link_parents[child_uuid].add(wanted_uuid)

        return False

    def call_recursion_by_using_input_feature_definition(
        self, feature_group_class: Type[AbstractFeatureGroup], uuid: UUID, options: Options, feature_name: FeatureName
    ) -> None:
        feature_group = feature_group_class()

        options = deepcopy(options)

        try:
            input_features = feature_group.input_features(options, feature_name)
        except NotImplementedError:  # This means, it is a root feature.
            input_features = None

        if input_features:
            features = Features(list(input_features), child_options=options, child_uuid=uuid)
            if features.child_uuid is None:
                raise ValueError(f"Features {features} has no parent uuid although it should have one.")
            self.feature_link_parents[features.child_uuid] = features.parent_uuids
            self.setup_features_recursion(features)

    def set_compute_framework(self, feature: Feature, compute_frameworks: Set[Type[ComputeFrameWork]]) -> Feature:
        """
        This function leads to that the feature has always a compute framework set!
        """
        if feature.compute_frameworks:
            if feature.get_compute_framework() not in compute_frameworks:
                raise ValueError(
                    f"Feature {feature.name} does not support compute framework {feature.compute_frameworks}."
                )
        else:
            feature.compute_frameworks = compute_frameworks
        return feature

    def set_data_type(self, feature: Feature, feature_group_class: Type[AbstractFeatureGroup]) -> Optional[DataType]:
        fg_data_type = feature_group_class.return_data_type_rule(feature)
        if feature.data_type and fg_data_type:
            if feature.data_type != fg_data_type:
                raise ValueError(
                    f"Feature {feature.name} has a data type mismatch with feature group {feature_group_class}."
                )
            return fg_data_type
        return fg_data_type or feature.data_type
