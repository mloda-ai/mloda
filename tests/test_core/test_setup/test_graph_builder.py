from collections import defaultdict
from typing import Dict, List, Optional, Set, Type, Union
from uuid import UUID
import uuid

from mloda.user import DataAccessCollection
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.core.prepare.graph.build_graph import BuildGraph
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Options
from mloda.user import Index
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1


class BaseTestGraphFeatureGroup3(BaseTestFeatureGroup1):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "GraphFeature" in feature_name.name:  # type: ignore
            return True
        return False

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {GraphComputeFramework2, GraphComputeFramework3}


class BaseTestGraphFeatureGroup4(BaseTestGraphFeatureGroup3):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "GraphFeatureIndex" in feature_name.name:  # type: ignore
            return True
        return False

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        index_1 = Index(tuple(["Index1"]))
        return [index_1]


class GraphComputeFramework1(ComputeFramework):
    pass


class GraphComputeFramework2(ComputeFramework):
    pass


class GraphComputeFramework3(ComputeFramework):
    pass


class TestGraphBuildGraph:
    def get_empty_feature_link_parents(self) -> Dict[UUID, Set[UUID]]:
        return defaultdict(set)

    def test_init_graph(self) -> None:
        feature_link_parents = self.get_empty_feature_link_parents()
        feature_link_parents[uuid.UUID(int=0)] = set()
        feature_link_parents[uuid.UUID(int=1)] = {uuid.UUID(int=2)}
        feature_link_parents[uuid.UUID(int=3)] = {uuid.UUID(int=2), uuid.UUID(int=3)}
        feature_link_parents[uuid.UUID(int=4)] = {uuid.UUID(int=3)}

        feature_group_collection: Dict[Type[FeatureGroup], Set[Feature]] = defaultdict(set)
        f0, f1, f2, f3, f4 = (
            Feature("GraphFeature2")._set_uuid(uuid.UUID(int=0)),
            Feature("GraphFeature3")._set_uuid(uuid.UUID(int=1)),
            Feature("GraphFeature4")._set_uuid(uuid.UUID(int=2)),
            Feature("GraphFeature6")._set_uuid(uuid.UUID(int=3)),
            Feature("GraphFeature5")._set_uuid(uuid.UUID(int=4)),
        )
        feature_group_collection[BaseTestFeatureGroup1] = {f0, f1, f2, f3, f4}

        build_graph = BuildGraph(feature_link_parents, feature_group_collection)
        build_graph.build_graph_from_feature_links()

        graph = build_graph.graph
        assert len(graph.get_nodes()) == 5
        assert len(graph.get_edges()) == 4

    def test_add_fg_to_graph(self) -> None:
        feature_group_collection: Dict[Type[FeatureGroup], Set[Feature]] = defaultdict(set)

        f1 = Feature("BaseTestFeature1", compute_framework="GraphComputeFramework1")
        g2, g3 = Feature("GraphFeature2"), Feature("GraphFeature3")

        feature_group_collection[BaseTestFeatureGroup1] = {f1}
        feature_group_collection[BaseTestGraphFeatureGroup3] = {g2, g3}

        feature_link_parents = self.get_empty_feature_link_parents()
        feature_link_parents[f1.uuid] = set()
        feature_link_parents[g3.uuid] = {g2.uuid}

        build_graph = BuildGraph(feature_link_parents, feature_group_collection)
        build_graph.build_graph_from_feature_links()

        graph = build_graph.graph

        # Assert nodes
        nodes = graph.get_nodes()
        assert len(nodes) == 3
        assert nodes[f1.uuid].feature.name == "BaseTestFeature1"
        assert nodes[f1.uuid].feature_group_class == BaseTestFeatureGroup1
        assert nodes[g2.uuid].feature_group_class == BaseTestGraphFeatureGroup3
        assert nodes[g3.uuid].feature_group_class == BaseTestGraphFeatureGroup3

        assert {fw.get_class_name() for fw in nodes[f1.uuid].feature.compute_frameworks} == {  # type: ignore
            GraphComputeFramework1.get_class_name()
        }
        assert nodes[g2.uuid].feature.compute_frameworks is None, "Should be None as it is not set in this test!"

        # Assert edges
        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[g2.uuid, g3.uuid].parent_feature_group_class == BaseTestGraphFeatureGroup3
        assert edges[g2.uuid, g3.uuid].child_feature_group_class == BaseTestGraphFeatureGroup3

    def test_add_compute_framework(self) -> None:
        """
        This test expects that the compute frameworks for the features were set before correctly.
        """
        feature_group_collection: Dict[Type[FeatureGroup], Set[Feature]] = defaultdict(set)

        f1 = Feature("BaseTestFeature1", compute_framework="GraphComputeFramework1")
        g2, g3 = Feature("GraphFeature2"), Feature("GraphFeature3")

        g2.compute_frameworks = {GraphComputeFramework2, GraphComputeFramework3}
        g3.compute_frameworks = {GraphComputeFramework3}

        feature_group_collection[BaseTestFeatureGroup1] = {f1}
        feature_group_collection[BaseTestGraphFeatureGroup3] = {g2, g3}

        feature_link_parents = self.get_empty_feature_link_parents()
        feature_link_parents[f1.uuid] = set()
        feature_link_parents[g2.uuid] = {g3.uuid}

        build_graph = BuildGraph(feature_link_parents, feature_group_collection)
        build_graph.build_graph_from_feature_links()

        graph = build_graph.graph

        # Assert nodes
        nodes = graph.get_nodes()
        assert len(nodes) == 3
        assert nodes[f1.uuid].feature.name == "BaseTestFeature1"
        assert nodes[f1.uuid].feature_group_class == BaseTestFeatureGroup1
        assert nodes[g2.uuid].feature_group_class == BaseTestGraphFeatureGroup3
        assert nodes[g3.uuid].feature_group_class == BaseTestGraphFeatureGroup3

        assert {fw.get_class_name() for fw in nodes[f1.uuid].feature.compute_frameworks} == {  # type: ignore
            GraphComputeFramework1.get_class_name()
        }
        assert nodes[g2.uuid].feature.compute_frameworks == {GraphComputeFramework2, GraphComputeFramework3}
        assert nodes[g3.uuid].feature.compute_frameworks == {GraphComputeFramework3}
