"""Tests for the public resolved-execution-plan API (issue #647).

Contract under test:
  * ``mloda.core.api.plan_info.PlanStep`` is a frozen dataclass describing one step of the
    resolved execution plan: ``step_kind``, ``feature_names``, ``feature_group``,
    ``compute_framework``, ``source_feature_group``, ``source_compute_framework`` plus the
    convenience properties ``feature_group_name`` and ``compute_framework_name``.
  * ``mlodaAPI.resolved_plan()`` returns ``list[PlanStep]`` on a prepared session, both before
    and after ``run()``, in execution-plan order.
  * ``mlodaAPI.explain(features, ...)`` mirrors the ``run_all`` parameter shape, prepares without
    executing and returns the same records as ``prepare(...).resolved_plan()``.
  * ``PlanStep`` is exported from both ``mloda.user`` and ``mloda.steward``.

A caller must reach all of this without importing anything from the internal execution-step
package; ``TestCallerNeedsNoInternalImport`` locks that in for this module itself.
"""

import ast
import dataclasses
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pytest

# Aliased: a bare ``import mloda.user`` would bind the name ``mloda`` to the package and collide
# with the ``mloda`` mlodaAPI alias imported below.
import mloda.steward as mloda_steward
import mloda.user as mloda_user
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import (
    Feature,
    FeatureName,
    Index,
    JoinSpec,
    Link,
    Options,
    PlanStep,
    PluginCollector,
    mloda,
    mlodaAPI,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


# ---------------------------------------------------------------------------
# Test feature groups
# ---------------------------------------------------------------------------


class PlanInfoPandasSource(FeatureGroup):
    """Pandas root source feeding the chained aggregation request."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"sales", "price"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"sales": [100, 200, 300], "price": [1.0, 2.0, 3.0]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class PlanInfoArrowSource(FeatureGroup):
    """PyArrow root source: forces a transform step into the pandas aggregation group."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"arrow_sales"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"arrow_sales": [100, 200, 300]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlanInfoNeverExecutes(FeatureGroup):
    """Plans fine, but blows up if anything actually executes it.

    ``explain()`` must never reach ``calculate_feature``, so a successful explain call over this
    feature group proves that explain does not execute the plan.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"never_executed"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("explain() must not execute the plan: calculate_feature was called")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class PlanInfoLeftSource(FeatureGroup):
    """Left side of the join scenario."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"jid", "left_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"jid": [1, 2, 3], "left_val": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("jid",))]


class PlanInfoRightSource(FeatureGroup):
    """Right side of the join scenario."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"jid", "right_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"jid": [1, 2, 3], "right_val": [1.5, 2.5, 3.5]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("jid",))]


class PlanInfoJoinConsumer(FeatureGroup):
    """Consumes features from both join sides, which forces a join step into the plan."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("left_val"), Feature("right_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PlanInfoJoinConsumer"] = data["left_val"] * data["right_val"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_AGGREGATION_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoPandasSource, PandasAggregatedFeatureGroup})
_TRANSFORM_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoArrowSource, PandasAggregatedFeatureGroup})
_NEVER_EXECUTES_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoNeverExecutes})
_JOIN_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoLeftSource, PlanInfoRightSource, PlanInfoJoinConsumer})

# The chained request from the issue: an aggregated feature over a source feature.
_CHAINED_FEATURES: list[Feature | str] = ["sales__mean_aggr"]


def _prepare_chained_session() -> mlodaAPI:
    return mloda.prepare(
        _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        plugin_collector=_AGGREGATION_PLUGINS,
    )


# ---------------------------------------------------------------------------
# PlanStep dataclass
# ---------------------------------------------------------------------------


class TestPlanStepDataclass:
    """PlanStep is a frozen dataclass with the documented fields and name properties."""

    def test_plan_step_is_a_dataclass(self) -> None:
        # Kept apart from the frozen check: is_dataclass() narrows PlanStep for mypy, which
        # would make every later PlanStep(...) construction in the same function untyped.
        assert dataclasses.is_dataclass(PlanStep)

    def test_plan_step_is_frozen(self) -> None:
        step = PlanStep(
            step_kind="compute",
            feature_names=("sales__mean_aggr",),
            feature_group=PandasAggregatedFeatureGroup,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            step.step_kind = "join"  # type: ignore[misc]

    def test_plan_step_exposes_documented_fields(self) -> None:
        field_names = [field.name for field in dataclasses.fields(PlanStep)]

        assert field_names == [
            "step_kind",
            "feature_names",
            "feature_group",
            "compute_framework",
            "source_feature_group",
            "source_compute_framework",
        ]

    def test_name_properties_return_class_names(self) -> None:
        step = PlanStep(
            step_kind="compute",
            feature_names=("sales__mean_aggr",),
            feature_group=PandasAggregatedFeatureGroup,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

        assert step.feature_group_name == "PandasAggregatedFeatureGroup"
        assert step.compute_framework_name == "PandasDataFrame"

    def test_name_properties_are_none_when_class_is_none(self) -> None:
        step = PlanStep(
            step_kind="join",
            feature_names=(),
            feature_group=None,
            compute_framework=None,
            source_feature_group=None,
            source_compute_framework=None,
        )

        assert step.feature_group_name is None
        assert step.compute_framework_name is None

    def test_plan_steps_compare_by_value(self) -> None:
        """Records must be value-comparable so callers can diff two plans."""
        first = PlanStep(
            step_kind="compute",
            feature_names=("sales",),
            feature_group=PlanInfoPandasSource,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )
        second = PlanStep(
            step_kind="compute",
            feature_names=("sales",),
            feature_group=PlanInfoPandasSource,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

        assert first == second


# ---------------------------------------------------------------------------
# Definition of done: chained aggregated feature
# ---------------------------------------------------------------------------


class TestResolvedPlanForChainedFeature:
    """DoD: requesting a chained ``*__mean_aggr`` feature exposes the concrete aggregation
    FeatureGroup implementation on the selected compute framework."""

    def test_resolved_plan_reports_source_and_aggregation_compute_steps(self) -> None:
        session = _prepare_chained_session()

        plan = session.resolved_plan()

        assert isinstance(plan, list)
        assert all(isinstance(step, PlanStep) for step in plan)

        compute_steps = [step for step in plan if step.step_kind == "compute"]
        assert len(compute_steps) == 2, f"expected two compute steps, got {[s.step_kind for s in plan]}"

        source_step, aggregation_step = compute_steps

        # The root source feature the aggregation chains off.
        assert source_step.feature_names == ("sales",)
        assert source_step.feature_group is PlanInfoPandasSource
        assert source_step.compute_framework is PandasDataFrame

        # The concrete aggregation implementation on the selected framework.
        assert aggregation_step.feature_names == ("sales__mean_aggr",)
        assert aggregation_step.feature_group is PandasAggregatedFeatureGroup
        assert aggregation_step.compute_framework is PandasDataFrame

        assert aggregation_step.feature_group_name == "PandasAggregatedFeatureGroup"
        assert aggregation_step.compute_framework_name == "PandasDataFrame"

    def test_compute_steps_carry_no_source_fields(self) -> None:
        """source_* fields are transform/join specific and stay None on compute steps."""
        plan = _prepare_chained_session().resolved_plan()

        for step in (step for step in plan if step.step_kind == "compute"):
            assert step.source_feature_group is None
            assert step.source_compute_framework is None

    def test_feature_names_are_strings_and_sorted(self) -> None:
        """feature_names mirrors FeatureSet.get_all_names(): a sorted tuple of strings.

        Requested as ["sales", "price"], reported sorted as ("price", "sales").
        """
        session = mloda.prepare(
            ["sales", "price"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )

        plan = session.resolved_plan()
        source_steps = [step for step in plan if step.feature_group is PlanInfoPandasSource]
        assert len(source_steps) == 1

        names = source_steps[0].feature_names
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)
        assert names == ("price", "sales")
        assert list(names) == sorted(names)

    def test_plan_is_in_execution_order(self) -> None:
        """The source must be planned before the aggregation that consumes it."""
        plan = _prepare_chained_session().resolved_plan()

        groups = [step.feature_group for step in plan if step.step_kind == "compute"]
        assert groups.index(PlanInfoPandasSource) < groups.index(PandasAggregatedFeatureGroup)


# ---------------------------------------------------------------------------
# resolved_plan() before and after run()
# ---------------------------------------------------------------------------


class TestResolvedPlanBeforeAndAfterRun:
    def test_resolved_plan_available_after_prepare_without_running(self) -> None:
        session = _prepare_chained_session()

        plan = session.resolved_plan()

        assert len(plan) > 0

    def test_resolved_plan_is_unchanged_by_run(self) -> None:
        session = _prepare_chained_session()

        before_run = session.resolved_plan()
        results = session.run()
        after_run = session.resolved_plan()

        assert after_run == before_run

        # Sanity check that the session really did execute.
        assert len(results) == 1
        assert "sales__mean_aggr" in results[0].columns


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_matches_prepare_resolved_plan(self) -> None:
        explained = mlodaAPI.explain(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )
        prepared = _prepare_chained_session().resolved_plan()

        assert explained == prepared

    def test_explain_returns_plan_steps(self) -> None:
        explained = mlodaAPI.explain(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )

        assert isinstance(explained, list)
        assert all(isinstance(step, PlanStep) for step in explained)
        assert [step.feature_group for step in explained] == [PlanInfoPandasSource, PandasAggregatedFeatureGroup]

    def test_explain_does_not_execute_the_plan(self) -> None:
        """PlanInfoNeverExecutes.calculate_feature raises. explain() must still succeed."""
        explained = mlodaAPI.explain(
            ["never_executed"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_NEVER_EXECUTES_PLUGINS,
        )

        assert len(explained) == 1
        assert explained[0].feature_group is PlanInfoNeverExecutes
        assert explained[0].feature_names == ("never_executed",)

    def test_explain_is_reachable_from_the_mloda_alias(self) -> None:
        explained = mloda.explain(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )

        assert all(isinstance(step, PlanStep) for step in explained)


# ---------------------------------------------------------------------------
# transform and join step mapping
# ---------------------------------------------------------------------------


class TestTransformStepMapping:
    """A PyArrow source feeding the pandas aggregation group produces a transform step."""

    def test_transform_step_maps_from_and_to_group_and_framework(self) -> None:
        session = mloda.prepare(
            ["arrow_sales__mean_aggr"],
            compute_frameworks={PandasDataFrame, PyArrowTable},
            plugin_collector=_TRANSFORM_PLUGINS,
        )

        plan = session.resolved_plan()

        transform_steps = [step for step in plan if step.step_kind == "transform"]
        assert len(transform_steps) == 1, f"expected one transform step, got {[s.step_kind for s in plan]}"

        transform = transform_steps[0]

        # transform: feature_group/compute_framework are the destination, source_* the origin.
        assert transform.feature_names == ()
        assert transform.feature_group is PandasAggregatedFeatureGroup
        assert transform.compute_framework is PandasDataFrame
        assert transform.source_feature_group is PlanInfoArrowSource
        assert transform.source_compute_framework is PyArrowTable

    def test_transform_step_sits_between_its_compute_steps(self) -> None:
        session = mloda.prepare(
            ["arrow_sales__mean_aggr"],
            compute_frameworks={PandasDataFrame, PyArrowTable},
            plugin_collector=_TRANSFORM_PLUGINS,
        )

        kinds = [step.step_kind for step in session.resolved_plan()]

        assert kinds == ["compute", "transform", "compute"]


class TestJoinStepMapping:
    """Two linked pandas sources consumed by one feature group produce a join step."""

    def test_join_step_maps_frameworks_and_has_no_feature_group(self) -> None:
        link = Link.inner(JoinSpec(PlanInfoLeftSource, "jid"), JoinSpec(PlanInfoRightSource, "jid"))

        session = mloda.prepare(
            ["PlanInfoJoinConsumer"],
            compute_frameworks={PandasDataFrame},
            links={link},
            plugin_collector=_JOIN_PLUGINS,
        )

        plan = session.resolved_plan()

        join_steps = [step for step in plan if step.step_kind == "join"]
        assert len(join_steps) == 1, f"expected one join step, got {[s.step_kind for s in plan]}"

        join = join_steps[0]

        # join: no feature group, no feature names; frameworks are left (compute) and right (source).
        assert join.feature_names == ()
        assert join.feature_group is None
        assert join.feature_group_name is None
        assert join.compute_framework is PandasDataFrame
        assert join.source_compute_framework is PandasDataFrame
        assert join.source_feature_group is None

    def test_join_plan_contains_both_sources_and_the_consumer(self) -> None:
        link = Link.inner(JoinSpec(PlanInfoLeftSource, "jid"), JoinSpec(PlanInfoRightSource, "jid"))

        session = mloda.prepare(
            ["PlanInfoJoinConsumer"],
            compute_frameworks={PandasDataFrame},
            links={link},
            plugin_collector=_JOIN_PLUGINS,
        )

        plan = session.resolved_plan()

        compute_groups = {step.feature_group for step in plan if step.step_kind == "compute"}
        assert compute_groups == {PlanInfoLeftSource, PlanInfoRightSource, PlanInfoJoinConsumer}

        # The join must be planned before the consumer that depends on it.
        kinds = [step.step_kind for step in plan]
        last_compute = len(kinds) - 1 - kinds[::-1].index("compute")
        assert kinds.index("join") < last_compute


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPlanStepIsPubliclyExported:
    def test_plan_step_exported_from_user_and_steward(self) -> None:
        assert "PlanStep" in mloda_user.__all__
        assert "PlanStep" in mloda_steward.__all__

    def test_user_and_steward_export_the_same_object(self) -> None:
        assert mloda_user.PlanStep is mloda_steward.PlanStep
        assert mloda_user.PlanStep is PlanStep


class TestCallerNeedsNoInternalImport:
    """The whole public plan API must be reachable without touching internal execution steps."""

    def test_this_module_imports_nothing_from_the_internal_core_package(self) -> None:
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

        internal = "mloda.core.core"
        offenders = sorted(name for name in imported if name == internal or name.startswith(f"{internal}."))

        assert not offenders, f"the public plan API must not require {internal} imports, found: {offenders}"

    def test_public_entry_points_exist_on_the_public_api(self) -> None:
        assert hasattr(mlodaAPI, "resolved_plan")
        assert hasattr(mlodaAPI, "explain")
