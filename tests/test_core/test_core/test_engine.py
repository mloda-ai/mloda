from typing import Any
from unittest.mock import patch
from mloda_core.abstract_plugins.components.data_types import DataType

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.core.engine import Engine
from mloda_core.prepare.execution_plan import ExecutionPlan
from mloda_core.core.step.feature_group_step import FeatureGroupStep
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFrameWork1,
    BaseTestComputeFrameWork2,
)
from tests.test_core.test_setup.test_graph_builder import BaseTestGraphFeatureGroup3
from tests.test_core.test_setup.test_link_resolver import BaseLinkTestFeatureGroup1
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import (
    BaseTestFeatureGroup1,
    BaseTestFeatureGroup2,
)


class TestEngine:
    def test_init_engine(self) -> None:
        with (
            patch(
                "mloda_core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda_core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFrameWork1, BaseTestComputeFrameWork2],
                BaseTestFeatureGroup2: [BaseTestComputeFrameWork1],
            }

            compute_framework = {BaseTestComputeFrameWork1, BaseTestComputeFrameWork2}
            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            links = {
                Link.inner(
                    (BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                    (BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
                )
            }
            Engine(features, compute_framework, links)

    def test_setup_features_recursion(self) -> None:
        with (
            patch(
                "mloda_core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda_core.core.engine.Engine.create_setup_execution_plan"),
        ):
            # setup
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFrameWork1, BaseTestComputeFrameWork2],
                BaseTestFeatureGroup2: [BaseTestComputeFrameWork1],
            }

            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            compute_framework = {BaseTestComputeFrameWork1, BaseTestComputeFrameWork2}

            # test init
            engine = Engine(features, compute_framework, None)
            mocked_derived_accessible_plugins.assert_called_once()
            assert engine.feature_group_collection == {}

            # run
            engine.setup_features_recursion(features)

            type_result: Any = []
            for feature_group_class, set_feature in engine.feature_group_collection.items():
                if feature_group_class == BaseTestFeatureGroup1:
                    assert len(set_feature) == 3
                    type_result.extend(feature.data_type for feature in set_feature)
                elif feature_group_class == BaseTestFeatureGroup2:
                    assert len(set_feature) == 1
                    assert next(iter(set_feature)).data_type == DataType.STRING

            assert None in type_result
            assert DataType.STRING in type_result
            assert DataType.INT32 in type_result

    def test_create_setup_execution_plan(self) -> None:
        with patch(
            "mloda_core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ) as mocked_derived_accessible_plugins:
            # setup
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: {BaseTestComputeFrameWork1, BaseTestComputeFrameWork2},
                BaseTestFeatureGroup2: {BaseTestComputeFrameWork2},
            }

            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            compute_framework = {BaseTestComputeFrameWork1, BaseTestComputeFrameWork2}

            links = {
                Link.inner(
                    (BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                    (BaseTestFeatureGroup2, Index(tuple(["Index1"]))),
                )
            }

            engine = Engine(features, compute_framework, links)

            # run
            data_types = set()

            for step in engine.execution_planner:
                if isinstance(step, FeatureGroupStep):
                    for f in step.features.features:
                        if f.name == FeatureName("BaseTestFeature1"):
                            data_types.add(f.data_type)

                        if f.name == FeatureName("BaseTestFeature1"):
                            assert not step.required_uuids
                        else:
                            assert step.required_uuids

            assert len(data_types) == 3
            assert isinstance(engine.execution_planner, ExecutionPlan)
