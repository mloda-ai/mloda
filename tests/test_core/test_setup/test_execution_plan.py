from typing import Tuple
import uuid
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.resolve_graph import PlannedQueue
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda.user import Features
from mloda.user import Index
from mloda.user import Link, JoinSpec
from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import BaseTestComputeFramework1
from tests.test_core.test_setup.test_link_resolver import BaseLinkTestFeatureGroup1
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import (
    BaseTestFeatureGroup1,
    BaseTestFeatureGroup2,
)


class TestExecutionPlan:
    def get_link_trekker(self) -> Tuple[LinkTrekker, LinkFrameworkTrekker]:
        link_trekker = LinkTrekker()
        link = Link.inner(
            JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
            JoinSpec(BaseTestFeatureGroup2, Index(tuple(["Index1"]))),
        )

        link_framework_trekker = (link, BaseTestComputeFramework1, BaseTestComputeFramework1)
        link_trekker.update(link_framework_trekker, uuid.UUID(int=1))
        return link_trekker, link_framework_trekker

    def test_execution_plan_init(self) -> None:
        planned_queue: PlannedQueue = []
        link_trekker, link_framework_trekker = self.get_link_trekker()
        planned_queue.append(link_framework_trekker)

        features = set(Features(["BaseTestFeature1", "BaseTestFeature2"]))
        for feat in features:
            feat.compute_frameworks = {BaseTestComputeFramework1}

        feature_group_features = (BaseTestFeatureGroup1, features)
        planned_queue.append(feature_group_features)

        ExecutionPlan()
