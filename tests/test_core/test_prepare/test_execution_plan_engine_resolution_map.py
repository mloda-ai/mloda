"""ExecutionPlan takes the engine's input_features() resolution keyed by feature uuid and unions it
onto each step's FeatureSet; a plan built without one keeps the runtime fallback; Feature itself
carries no engine bookkeeping.
"""

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class _ResolutionMapFeatureGroup(FeatureGroup):
    """Stand-in feature group; calculate_feature is never invoked by these tests."""


class TestExecutionPlanEngineResolutionMap:
    """The plan's per-step resolution comes from the engine map by uuid, never from the Feature object."""

    def test_step_unions_engine_resolution_and_ignores_unresolved_members(self) -> None:
        framework_name = PythonDictFramework.get_class_name()
        resolved = Feature("resolved", compute_framework=framework_name)
        root = Feature("root", compute_framework=framework_name)
        injected = Feature("injected", compute_framework=framework_name)

        plan = ExecutionPlan(
            resolved_input_feature_names={
                resolved.uuid: frozenset({"engine_parent"}),
                root.uuid: None,
            }
        )
        fg_steps = plan.run_feature_group(
            (_ResolutionMapFeatureGroup, {resolved, root, injected}),
            parent_to_children_mapping={},
            pre_required_uuids=set(),
        )

        assert len(fg_steps) == 1, f"Expected all features batched into a single step, got: {fg_steps}"
        feature_set = next(iter(fg_steps.values())).features
        assert resolved in feature_set.features
        assert root in feature_set.features
        assert injected in feature_set.features

        assert feature_set.declared_input_feature_names == frozenset({"engine_parent"}), (
            "A step's declared_input_feature_names must be the union of the engine map entries of its members; "
            "a None entry (root feature) and a missing entry (injected feature) contribute nothing."
        )
        assert feature_set.declared_input_features_resolved is True, (
            "A plan given the engine's resolution map must mark every step's FeatureSet resolved."
        )

    def test_plan_without_engine_map_leaves_runtime_fallback(self) -> None:
        framework_name = PythonDictFramework.get_class_name()
        first = Feature("first", compute_framework=framework_name)
        second = Feature("second", compute_framework=framework_name)

        plan = ExecutionPlan()
        fg_steps = plan.run_feature_group(
            (_ResolutionMapFeatureGroup, {first, second}),
            parent_to_children_mapping={},
            pre_required_uuids=set(),
        )

        assert len(fg_steps) == 1, f"Expected both features batched into a single step, got: {fg_steps}"
        feature_set = next(iter(fg_steps.values())).features

        assert feature_set.declared_input_features_resolved is False, (
            "A plan built without an engine resolution map must leave the FeatureSet un-resolved so "
            "ComputeFramework._declared_input_feature_names's runtime fallback runs."
        )
        assert feature_set.declared_input_feature_names is None

    def test_feature_carries_no_engine_bookkeeping(self) -> None:
        feature = Feature("plain_feature")

        assert not hasattr(feature, "declared_input_feature_names"), (
            "Feature must not carry the engine's input_features() resolution; it lives on the engine keyed by uuid."
        )
        assert not hasattr(feature, "declared_input_feature_names_resolved"), (
            "Feature must not carry an engine resolution flag; it lives on the engine keyed by uuid."
        )
