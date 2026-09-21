"""ExecutionPlan takes the engine's input_features() resolution keyed by feature uuid and unions it
onto each step's FeatureSet; a plan built without one keeps the runtime fallback; Feature itself
carries no engine bookkeeping.
"""

from mloda.core.abstract_plugins.components.feature_set import merge_input_feature_edges
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

    def test_step_builds_edges_from_engine_resolution_keyed_by_feature_name(self) -> None:
        framework_name = PythonDictFramework.get_class_name()
        first = Feature("first", compute_framework=framework_name)
        second = Feature("second", compute_framework=framework_name)
        root = Feature("root", compute_framework=framework_name)
        injected = Feature("injected", compute_framework=framework_name)

        plan = ExecutionPlan(
            resolved_input_feature_names={
                first.uuid: frozenset({"src_b", "src_a"}),
                second.uuid: frozenset({"src_c"}),
                root.uuid: None,
            }
        )
        fg_steps = plan.run_feature_group(
            (_ResolutionMapFeatureGroup, {first, second, root, injected}),
            parent_to_children_mapping={},
            pre_required_uuids=set(),
        )

        feature_set = next(iter(fg_steps.values())).features

        assert feature_set.declared_input_feature_edges == {"first": ("src_a", "src_b"), "second": ("src_c",)}, (
            "Edges are keyed by str(feature.name) with sorted plain-str values; a None entry (root) and a "
            "missing entry (injected) are absent from the mapping."
        )
        assert feature_set.declared_input_feature_names == frozenset({"src_a", "src_b", "src_c"})

    def test_edges_of_features_sharing_a_name_are_union_merged_sorted(self) -> None:
        edges = merge_input_feature_edges(
            [
                ("twin", frozenset({"src_b", "src_a"})),
                ("twin", frozenset({"src_c", "src_a"})),
                ("other", ["src_z"]),
                ("root", frozenset()),
            ]
        )

        assert edges == {"twin": ("src_a", "src_b", "src_c"), "other": ("src_z",)}, (
            "Same-named pairs are union-merged into a sorted tuple; a pair with no declared inputs is absent."
        )
        assert edges is not None
        assert all(type(key) is str for key in edges)
        assert all(type(name) is str for names in edges.values() for name in names)

    def test_merge_input_feature_edges_returns_none_when_nothing_declared(self) -> None:
        assert merge_input_feature_edges([]) is None
        assert merge_input_feature_edges([("root", frozenset())]) is None

    def test_different_options_twins_carry_their_own_edges_in_separate_steps(self) -> None:
        framework_name = PythonDictFramework.get_class_name()
        twin_one = Feature("twin", compute_framework=framework_name, options={"variant": 1})
        twin_two = Feature("twin", compute_framework=framework_name, options={"variant": 2})

        plan = ExecutionPlan(
            resolved_input_feature_names={
                twin_one.uuid: frozenset({"src_b", "src_a"}),
                twin_two.uuid: frozenset({"src_c", "src_a"}),
            }
        )
        fg_steps = plan.run_feature_group(
            (_ResolutionMapFeatureGroup, {twin_one, twin_two}),
            parent_to_children_mapping={},
            pre_required_uuids=set(),
        )

        step_edges = [step.features.declared_input_feature_edges for step in fg_steps.values()]

        assert len(fg_steps) == 2, f"Different options must split into separate steps, got: {fg_steps}"
        assert sorted(step_edges, key=str) == [{"twin": ("src_a", "src_b")}, {"twin": ("src_a", "src_c")}]

    def test_step_without_any_engine_entries_has_no_edges(self) -> None:
        framework_name = PythonDictFramework.get_class_name()
        root = Feature("root", compute_framework=framework_name)
        injected = Feature("injected", compute_framework=framework_name)

        plan = ExecutionPlan(resolved_input_feature_names={root.uuid: None})
        fg_steps = plan.run_feature_group(
            (_ResolutionMapFeatureGroup, {root, injected}),
            parent_to_children_mapping={},
            pre_required_uuids=set(),
        )

        feature_set = next(iter(fg_steps.values())).features

        assert feature_set.declared_input_feature_edges is None

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
        assert feature_set.declared_input_feature_edges is None

    def test_feature_carries_no_engine_bookkeeping(self) -> None:
        feature = Feature("plain_feature")

        assert not hasattr(feature, "declared_input_feature_names"), (
            "Feature must not carry the engine's input_features() resolution; it lives on the engine keyed by uuid."
        )
        assert not hasattr(feature, "declared_input_feature_names_resolved"), (
            "Feature must not carry an engine resolution flag; it lives on the engine keyed by uuid."
        )
