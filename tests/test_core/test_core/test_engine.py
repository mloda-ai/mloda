from typing import Any
from unittest.mock import MagicMock, call, patch
from mloda.user import DataType

from mloda.user import FeatureName
from mloda.core.core.engine import Engine
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import Features
from mloda.user import Index
from mloda.user import Link, JoinSpec

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
    BaseTestComputeFramework2,
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
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFramework1, BaseTestComputeFramework2],
                BaseTestFeatureGroup2: [BaseTestComputeFramework1],
            }

            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}
            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            links = {
                Link.inner(
                    JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                    JoinSpec(BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
                )
            }
            Engine(features, compute_framework, links)

    def test_setup_features_recursion(self) -> None:
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            # setup
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFramework1, BaseTestComputeFramework2],
                BaseTestFeatureGroup2: [BaseTestComputeFramework1],
            }

            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}

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
            "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ) as mocked_derived_accessible_plugins:
            # setup
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: {BaseTestComputeFramework1, BaseTestComputeFramework2},
                BaseTestFeatureGroup2: {BaseTestComputeFramework2},
            }

            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}

            links = {
                Link.inner(
                    JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                    JoinSpec(BaseTestFeatureGroup2, Index(tuple(["Index1"]))),
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

    def test_compute_returns_independent_execution_planner_state(self) -> None:
        """Engine.compute() must return orchestrators with independent execution planner state.

        Currently Engine.compute() passes self.execution_planner by reference to
        ExecutionOrchestrator. If step_is_done is set to True on steps during the
        first compute() call, those mutations leak into subsequent compute() calls
        because they share the same object. Each call to compute() should get an
        independent copy of the execution plan so that step_is_done starts as False.
        """
        with patch(
            "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ) as mocked_derived_accessible_plugins:
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: {BaseTestComputeFramework1, BaseTestComputeFramework2},
                BaseTestFeatureGroup2: {BaseTestComputeFramework2},
            }

            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}

            links = {
                Link.inner(
                    JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                    JoinSpec(BaseTestFeatureGroup2, Index(tuple(["Index1"]))),
                )
            }

            engine = Engine(features, compute_framework, links)

            # Verify the execution planner has steps with step_is_done = False
            steps = list(engine.execution_planner)
            assert len(steps) > 0, "Execution plan must have at least one step"
            for step in steps:
                assert step.step_is_done is False

            # First call to compute() - get orchestrator1
            orchestrator1 = engine.compute()
            assert isinstance(orchestrator1, ExecutionOrchestrator)

            # Simulate what happens during execution: mark all steps as done
            for step in orchestrator1.execution_planner:
                step.step_is_done = True

            # Verify the mutation took effect on orchestrator1
            for step in orchestrator1.execution_planner:
                assert step.step_is_done is True

            # Second call to compute() - get orchestrator2
            orchestrator2 = engine.compute()
            assert isinstance(orchestrator2, ExecutionOrchestrator)

            # The critical assertion: orchestrator2 steps must have step_is_done = False.
            # Engine.compute() deepcopies self.execution_planner, so orchestrator2
            # gets independent step state from orchestrator1.
            for step in orchestrator2.execution_planner:
                assert step.step_is_done is False, (
                    f"step_is_done leaked between compute() calls: "
                    f"step {step} has step_is_done={step.step_is_done}, expected False"
                )

    def test_setup_features_recursion_does_not_recreate_prefilter_plugins(self) -> None:
        """PreFilterPlugins should be created once in __init__, not per feature in setup_features_recursion."""
        original_init = PreFilterPlugins.__init__

        init_call_count = 0

        def tracking_init(self_pfp: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal init_call_count
            init_call_count += 1
            original_init(self_pfp, *args, **kwargs)

        with (
            patch.object(PreFilterPlugins, "__init__", tracking_init),
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            features = Features(["BaseTestFeature1", "BaseTestFeature2"])
            compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}

            engine = Engine(features, compute_framework, None)

            # Reset counter after __init__ to isolate setup_features_recursion calls
            assert init_call_count == 1, (
                f"PreFilterPlugins should be instantiated exactly once during Engine.__init__, "
                f"but was instantiated {init_call_count} times"
            )

            init_call_count = 0
            engine.setup_features_recursion(features)

            assert init_call_count == 0, (
                f"PreFilterPlugins should not be instantiated during setup_features_recursion, "
                f"but was instantiated {init_call_count} time(s). "
                f"setup_features_recursion should reuse self.accessible_plugins from __init__"
            )
