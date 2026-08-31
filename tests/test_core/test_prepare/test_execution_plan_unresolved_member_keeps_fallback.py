"""Regression: a step's FeatureSet must not be marked plan-resolved when one of its
members is an injected feature (added via Engine._add_filter_feature or
_create_and_add_index_feature, bypassing Engine._process_feature) that never had its
own input_features() resolution stamped onto it during planning.
"""

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class _MixedResolutionFeatureGroup(FeatureGroup):
    """Stand-in feature group; calculate_feature is never invoked by this test."""


class TestExecutionPlanUnresolvedMemberKeepsFallback:
    """A FeatureSet mixing a plan-resolved feature with an unresolved (injected) one must stay unresolved."""

    def test_unresolved_member_prevents_plan_resolved_marking(self) -> None:
        framework_name = PythonDictFramework.get_class_name()
        engine_resolved_feature = Feature("engine_resolved_feature", compute_framework=framework_name)
        # Simulates Engine._process_feature's post-processing stamp.
        engine_resolved_feature.declared_input_feature_names_resolved = True

        injected_feature = Feature("injected_feature", compute_framework=framework_name)
        # No declared_input_feature_names_resolved stamp: Engine._add_filter_feature and
        # _create_and_add_index_feature add features via add_feature_to_collection, bypassing
        # _process_feature entirely, so a feature like this is never stamped.

        plan = ExecutionPlan()
        fg_steps = plan.run_feature_group(
            (_MixedResolutionFeatureGroup, {engine_resolved_feature, injected_feature}),
            parent_to_children_mapping={},
            pre_required_uuids=set(),
        )

        assert len(fg_steps) == 1, f"Expected both features batched into a single step, got: {fg_steps}"
        feature_set = next(iter(fg_steps.values())).features
        assert engine_resolved_feature in feature_set.features
        assert injected_feature in feature_set.features

        assert feature_set.declared_input_features_resolved is False, (
            "A FeatureSet containing an unresolved (injected) feature must be left un-resolved so "
            "ComputeFramework._declared_input_feature_names's runtime fallback still runs for this step. "
            "run_feature_group currently marks the whole FeatureSet resolved unconditionally, regardless "
            "of whether every member feature actually had its input_features() resolved during planning."
        )
