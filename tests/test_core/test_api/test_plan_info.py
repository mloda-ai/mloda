"""Tests for the public resolved-execution-plan API (issue #647).

Contract under test:
  * ``mloda.core.api.plan_info.PlanStep`` is a frozen dataclass describing one step of the
    resolved execution plan: ``step_kind``, ``feature_names``, ``feature_group``,
    ``compute_framework``, ``source_feature_group``, ``source_compute_framework``, ``join_type``
    plus the convenience properties ``feature_group_name``, ``compute_framework_name``,
    ``source_feature_group_name`` and ``source_compute_framework_name``.
  * ``step_kind`` is typed ``Literal["compute", "join", "transform"]``.
  * Join steps are not blank records: ``feature_group``/``source_feature_group`` are the link's
    declared left/right feature groups, ``compute_framework`` is the merge destination framework
    and ``source_compute_framework`` the framework merged in, ``join_type`` the link's join type.
  * ``PlanStep`` also carries ``requested_feature_names`` and ``injected_feature_names``, both
    defaulting to ``()``. On compute steps they partition ``feature_names`` into the names the user
    asked for (``initial_requested_data`` is True) and the names the engine injected (chained
    sources, link index features). Join and transform steps keep both empty.
  * ``PlanStep.input_feature_names`` is the prepare-time twin of ``HookContext.input_features``: on a
    compute step the sorted, deduplicated names its feature group declares as input, read from the
    step's ``FeatureSet.declared_input_feature_names``. A root group (no declared inputs) and every
    join and transform step report ``()``. The step's own feature names are not subtracted.
  * Join steps also expose the resolved orientation: ``join_destination_side`` ("left"/"right") and
    ``join_token`` are dataclass fields; ``join_inverted`` is a derived read-only property
    (``join_destination_side == "right"``, or None without a side). Both default to None on
    compute/transform steps and wherever ``build_plan_steps`` gets no ``resolved_join_plan``.
    ``join_token`` is minted fresh per planning run, so it stays out of equality.
  * Join steps also expose ``declared_left_frameworks``/``declared_right_frameworks``: sorted class
    tuples of each declared side's candidate compute frameworks, plus
    ``declared_left_framework_names``/``declared_right_framework_names`` string mirrors. Both class
    tuples default to ``()`` without a resolved join plan. ``build_plan_steps`` takes an explicit
    ``resolved_join_plan`` argument; a bare ``ExecutionPlan`` or list carries none.
  * ``build_plan_steps`` raises ``ValueError`` on a step it does not know, instead of dropping it.
  * ``mlodaAPI.resolved_plan()`` returns ``list[PlanStep]`` on a prepared session, both before
    and after ``run()``, in execution-plan order, and matches the plan that actually executed.
  * ``mlodaAPI.explain(features, ...)`` mirrors the ``prepare`` parameter shape with keyword-only
    parameters after ``features``, prepares without executing and returns the same records as
    ``prepare(...).resolved_plan()``.
  * ``PlanStep`` is exported from both ``mloda.user`` and ``mloda.steward``.

A caller must reach all of this without importing anything from the internal execution-step
package; ``TestCallerNeedsNoInternalImport`` locks that in for this module itself.

Root feature names in this module carry a ``plan_info_`` prefix on purpose: a ``DataCreator``
claim is registry-wide, so generic names like ``sales`` would leak into every other test.
"""

import ast
import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional, get_args, get_origin
from uuid import UUID

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

# Aliased: a bare ``import mloda.user`` would bind the name ``mloda`` to the package and collide
# with the ``mloda`` mlodaAPI alias imported below.
import mloda.steward as mloda_steward
import mloda.user as mloda_user
from mloda.core.api.plan_info import build_plan_steps
from mloda.core.prepare.resolved_join import ResolvedJoinPlan
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook, HookContext
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
        return DataCreator({"plan_info_sales", "plan_info_price"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"plan_info_sales": [100, 200, 300], "plan_info_price": [1.0, 2.0, 3.0]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class PlanInfoArrowSource(FeatureGroup):
    """PyArrow root source: forces a transform step into the pandas aggregation group."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"plan_info_arrow_sales"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"plan_info_arrow_sales": [100, 200, 300]})

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
        return DataCreator({"plan_info_never_executed"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("explain() must not execute the plan: calculate_feature was called")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class PlanInfoLeftSource(FeatureGroup):
    """Left side of the single-framework join scenario."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"plan_info_jid", "plan_info_left_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"plan_info_jid": [1, 2, 3], "plan_info_left_val": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("plan_info_jid",))]


class PlanInfoRightSource(FeatureGroup):
    """Right side of the single-framework join scenario."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"plan_info_jid", "plan_info_right_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"plan_info_jid": [1, 2, 3], "plan_info_right_val": [1.5, 2.5, 3.5]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("plan_info_jid",))]


class PlanInfoJoinConsumer(FeatureGroup):
    """Consumes features from both join sides, which forces a join step into the plan."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("plan_info_left_val"), Feature("plan_info_right_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PlanInfoJoinConsumer"] = data["plan_info_left_val"] * data["plan_info_right_val"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class PlanInfoCrossLeftPandas(FeatureGroup):
    """Left side of the cross-framework join scenario: pandas."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"plan_info_xjid", "plan_info_xleft_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"plan_info_xjid": [1, 2, 3], "plan_info_xleft_val": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("plan_info_xjid",))]


class PlanInfoCrossRightArrow(FeatureGroup):
    """Right side of the cross-framework join scenario: pyarrow.

    Distinct framework from the left side, so a left/right swap in the plan records is
    detectable instead of being hidden behind two identical frameworks.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"plan_info_xjid", "plan_info_xright_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"plan_info_xjid": [1, 2, 3], "plan_info_xright_val": [1.5, 2.5, 3.5]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("plan_info_xjid",))]


class PlanInfoCrossConsumer(FeatureGroup):
    """Consumes one feature per side of the cross-framework join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("plan_info_xleft_val"), Feature("plan_info_xright_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PlanInfoCrossConsumer"] = data["plan_info_xleft_val"] * data["plan_info_xright_val"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class PlanInfoInvertedConsumer(FeatureGroup):
    """Consumes the same two join sides on PyArrow, which puts the merge destination on the declared right side."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("plan_info_xleft_val"), Feature("plan_info_xright_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        product = pc.multiply(data["plan_info_xleft_val"], data["plan_info_xright_val"])
        return data.append_column("PlanInfoInvertedConsumer", product)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class PlanInfoUnknownStep:
    """Not a FeatureGroupStep/TransformFrameworkStep/JoinStep: build_plan_steps must reject it."""


class PlanInfoCalculateHookRecorder(Extender):
    """Records the HookContext of every calculate hook, keyed by feature group class and feature names."""

    def __init__(self) -> None:
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result

    def input_features_by_step(self) -> dict[tuple[str, tuple[str, ...]], frozenset[str]]:
        return {
            (context.feature_group_class, context.feature_names): context.input_features or frozenset()
            for context in self.captured
        }


_AGGREGATION_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoPandasSource, PandasAggregatedFeatureGroup})
_TRANSFORM_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoArrowSource, PandasAggregatedFeatureGroup})
_NEVER_EXECUTES_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoNeverExecutes})
_JOIN_PLUGINS = PluginCollector.enabled_feature_groups({PlanInfoLeftSource, PlanInfoRightSource, PlanInfoJoinConsumer})
_CROSS_JOIN_PLUGINS = PluginCollector.enabled_feature_groups(
    {PlanInfoCrossLeftPandas, PlanInfoCrossRightArrow, PlanInfoCrossConsumer}
)
_INVERTED_JOIN_PLUGINS = PluginCollector.enabled_feature_groups(
    {PlanInfoCrossLeftPandas, PlanInfoCrossRightArrow, PlanInfoInvertedConsumer}
)

# The chained request from the issue: an aggregated feature over a source feature.
_CHAINED_FEATURES: list[Feature | str] = ["plan_info_sales__mean_aggr"]


def _prepare_chained_session() -> mlodaAPI:
    return mloda.prepare(
        _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        plugin_collector=_AGGREGATION_PLUGINS,
    )


def _prepare_chained_session_with(recorder: PlanInfoCalculateHookRecorder) -> mlodaAPI:
    return mloda.prepare(
        _CHAINED_FEATURES,
        compute_frameworks={PandasDataFrame},
        plugin_collector=_AGGREGATION_PLUGINS,
        function_extender={recorder},
    )


def _prepare_single_framework_join_session() -> mlodaAPI:
    link = Link.inner(JoinSpec(PlanInfoLeftSource, "plan_info_jid"), JoinSpec(PlanInfoRightSource, "plan_info_jid"))

    return mloda.prepare(
        ["PlanInfoJoinConsumer"],
        compute_frameworks={PandasDataFrame},
        links={link},
        plugin_collector=_JOIN_PLUGINS,
    )


def _cross_link() -> Link:
    return Link.inner(
        JoinSpec(PlanInfoCrossLeftPandas, "plan_info_xjid"),
        JoinSpec(PlanInfoCrossRightArrow, "plan_info_xjid"),
    )


def _prepare_cross_join(consumer: str, plugins: PluginCollector) -> mlodaAPI:
    return mloda.prepare(
        [consumer],
        compute_frameworks={PandasDataFrame, PyArrowTable},
        links={_cross_link()},
        plugin_collector=plugins,
    )


def _prepare_cross_framework_join_session() -> mlodaAPI:
    return _prepare_cross_join("PlanInfoCrossConsumer", _CROSS_JOIN_PLUGINS)


def _prepare_inverted_cross_framework_join_session() -> mlodaAPI:
    """Same link, PyArrow consumer: the join runs in the declared right side's framework."""
    return _prepare_cross_join("PlanInfoInvertedConsumer", _INVERTED_JOIN_PLUGINS)


def _raw_steps_by_kind(session: mlodaAPI) -> dict[str, list[Any]]:
    """Group a prepared session's raw execution-plan steps by the kind build_plan_steps gives them.

    Keeps this module free of internal execution-step imports while still reaching the raw steps.
    """
    assert session.engine is not None
    raw_steps: list[Any] = list(session.engine.execution_planner)

    grouped: dict[str, list[Any]] = {"compute": [], "join": [], "transform": []}
    for raw_step, record in zip(raw_steps, build_plan_steps(raw_steps)):
        grouped[record.step_kind].append(raw_step)
    return grouped


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
            feature_names=("plan_info_sales__mean_aggr",),
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
            "join_type",
            "requested_feature_names",
            "injected_feature_names",
            "input_feature_names",
            "join_destination_side",
            "join_token",
            "declared_left_frameworks",
            "declared_right_frameworks",
        ]

    def test_join_type_defaults_to_none(self) -> None:
        """Compute and transform steps must not need to pass join_type."""
        step = PlanStep(
            step_kind="compute",
            feature_names=("plan_info_sales",),
            feature_group=PlanInfoPandasSource,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

        assert step.join_type is None

        join_type_field = {field.name: field for field in dataclasses.fields(PlanStep)}["join_type"]
        assert join_type_field.default is None

    def test_step_kind_is_a_literal_of_the_three_kinds(self) -> None:
        """step_kind is typed, so a caller can exhaustively match on it."""
        annotation = PlanStep.__annotations__["step_kind"]

        # Annotated as Any: typeshed's get_origin overloads cannot express `is Literal`,
        # so a precisely typed operand would trip mypy's strict_equality check.
        origin: Any = get_origin(annotation)
        assert origin is Literal, f"step_kind must be a Literal, got {annotation!r}"
        assert set(get_args(annotation)) == {"compute", "join", "transform"}

    def test_name_properties_return_class_names(self) -> None:
        step = PlanStep(
            step_kind="transform",
            feature_names=(),
            feature_group=PandasAggregatedFeatureGroup,
            compute_framework=PandasDataFrame,
            source_feature_group=PlanInfoArrowSource,
            source_compute_framework=PyArrowTable,
        )

        assert step.feature_group_name == "PandasAggregatedFeatureGroup"
        assert step.compute_framework_name == "PandasDataFrame"
        assert step.source_feature_group_name == "PlanInfoArrowSource"
        assert step.source_compute_framework_name == "PyArrowTable"

    def test_name_properties_are_none_when_class_is_none(self) -> None:
        step = PlanStep(
            step_kind="compute",
            feature_names=(),
            feature_group=None,
            compute_framework=None,
            source_feature_group=None,
            source_compute_framework=None,
        )

        assert step.feature_group_name is None
        assert step.compute_framework_name is None
        assert step.source_feature_group_name is None
        assert step.source_compute_framework_name is None

    def test_plan_steps_compare_by_value(self) -> None:
        """Records must be value-comparable so callers can diff two plans."""
        first = PlanStep(
            step_kind="compute",
            feature_names=("plan_info_sales",),
            feature_group=PlanInfoPandasSource,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )
        second = PlanStep(
            step_kind="compute",
            feature_names=("plan_info_sales",),
            feature_group=PlanInfoPandasSource,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

        assert first == second

    def test_join_type_participates_in_equality(self) -> None:
        """Two otherwise identical join records with different join types must not compare equal."""
        inner = PlanStep(
            step_kind="join",
            feature_names=(),
            feature_group=PlanInfoLeftSource,
            compute_framework=PandasDataFrame,
            source_feature_group=PlanInfoRightSource,
            source_compute_framework=PandasDataFrame,
            join_type="inner",
        )
        outer = dataclasses.replace(inner, join_type="outer")

        assert inner != outer


class TestPlanStepRequestedAndInjectedFields:
    """PlanStep splits feature_names into user-requested and engine-injected names."""

    @staticmethod
    def _compute_step() -> PlanStep:
        return PlanStep(
            step_kind="compute",
            feature_names=("plan_info_sales",),
            feature_group=PlanInfoPandasSource,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

    def test_requested_and_injected_default_to_empty_tuples(self) -> None:
        """A caller constructing a PlanStep by hand must not need to pass either field."""
        step = self._compute_step()

        assert step.requested_feature_names == ()
        assert step.injected_feature_names == ()

        fields_by_name = {field.name: field for field in dataclasses.fields(PlanStep)}
        assert fields_by_name["requested_feature_names"].default == ()
        assert fields_by_name["injected_feature_names"].default == ()

    def test_requested_and_injected_participate_in_equality(self) -> None:
        """Two records that only differ in the requested/injected split must not compare equal."""
        base = self._compute_step()
        with_requested = dataclasses.replace(base, requested_feature_names=("plan_info_sales",))
        with_injected = dataclasses.replace(base, injected_feature_names=("plan_info_sales",))

        assert base != with_requested
        assert base != with_injected
        assert with_requested != with_injected


class TestPlanStepInputFeatureNamesField:
    """input_feature_names is an optional tuple[str, ...] field sitting next to the other name tuples."""

    @staticmethod
    def _compute_step() -> PlanStep:
        return PlanStep(
            step_kind="compute",
            feature_names=("plan_info_sales__mean_aggr",),
            feature_group=PandasAggregatedFeatureGroup,
            compute_framework=PandasDataFrame,
            source_feature_group=None,
            source_compute_framework=None,
        )

    def test_input_feature_names_defaults_to_an_empty_tuple(self) -> None:
        step = self._compute_step()

        assert step.input_feature_names == ()

        fields_by_name = {field.name: field for field in dataclasses.fields(PlanStep)}
        assert fields_by_name["input_feature_names"].default == ()

    def test_input_feature_names_is_annotated_as_a_tuple_of_strings(self) -> None:
        annotation = PlanStep.__annotations__["input_feature_names"]

        # Annotated as Any for the same reason as step_kind's Literal check above: typeshed's
        # get_origin overloads make a precisely typed operand trip mypy's strict_equality check.
        origin: Any = get_origin(annotation)
        assert origin is tuple, f"input_feature_names must be a tuple type, got {annotation!r}"
        assert get_args(annotation) == (str, Ellipsis)

    def test_input_feature_names_follows_injected_feature_names(self) -> None:
        """Positional construction order: the third name tuple sits directly behind the other two."""
        field_names = [field.name for field in dataclasses.fields(PlanStep)]

        assert field_names.index("input_feature_names") == field_names.index("injected_feature_names") + 1

    def test_input_feature_names_participates_in_equality(self) -> None:
        base = self._compute_step()
        with_inputs = dataclasses.replace(base, input_feature_names=("plan_info_sales",))

        assert base != with_inputs
        assert with_inputs == dataclasses.replace(base, input_feature_names=("plan_info_sales",))


class TestJoinOrientationFieldsOnTheDataclass:
    """join_destination_side, join_token, declared_left_frameworks and declared_right_frameworks are
    optional, typed dataclass fields; join_inverted is a derived read-only property, not a field."""

    @staticmethod
    def _join_step() -> PlanStep:
        return PlanStep(
            step_kind="join",
            feature_names=(),
            feature_group=PlanInfoLeftSource,
            compute_framework=PandasDataFrame,
            source_feature_group=PlanInfoRightSource,
            source_compute_framework=PandasDataFrame,
            join_type="inner",
        )

    def test_join_orientation_fields_default_to_none(self) -> None:
        step = self._join_step()

        assert step.join_destination_side is None
        assert step.join_token is None
        assert step.declared_left_frameworks == ()
        assert step.declared_right_frameworks == ()
        assert step.join_inverted is None

        fields_by_name = {field.name: field for field in dataclasses.fields(PlanStep)}
        assert fields_by_name["join_destination_side"].default is None
        assert fields_by_name["join_token"].default is None
        assert fields_by_name["declared_left_frameworks"].default == ()
        assert fields_by_name["declared_right_frameworks"].default == ()

    def test_join_orientation_fields_are_optional_scalars(self) -> None:
        annotations = PlanStep.__annotations__

        side, none_type = get_args(annotations["join_destination_side"])
        assert none_type is type(None)
        assert set(get_args(side)) == {"left", "right"}
        assert set(get_args(annotations["join_token"])) == {UUID, type(None)}

        assert "join_inverted" not in annotations
        assert "join_inverted" not in {field.name for field in dataclasses.fields(PlanStep)}

    def test_the_sides_and_declared_frameworks_compare_and_the_token_does_not(self) -> None:
        """The sides and declared frameworks are stable planner answers; the token is a fresh uuid per run."""
        base = self._join_step()
        token = UUID("00000000-0000-4000-8000-000000000001")

        assert base != dataclasses.replace(base, join_destination_side="right")
        assert base != dataclasses.replace(base, declared_left_frameworks=(PandasDataFrame,))
        assert base != dataclasses.replace(base, declared_right_frameworks=(PyArrowTable,))
        assert base == dataclasses.replace(base, join_token=token)
        assert hash(base) == hash(dataclasses.replace(base, join_token=token))
        assert hash(base) != hash(dataclasses.replace(base, declared_left_frameworks=(PandasDataFrame,)))

    def test_join_inverted_cannot_be_set_via_replace(self) -> None:
        """join_inverted is a derived property, so dataclasses.replace must reject it as a field."""
        base = self._join_step()

        with pytest.raises(TypeError):
            dataclasses.replace(base, join_inverted=True)  # type: ignore[call-arg]

    def test_join_inverted_is_derived_from_the_destination_side(self) -> None:
        base = self._join_step()

        assert base.join_inverted is None
        assert dataclasses.replace(base, join_destination_side="left").join_inverted is False
        assert dataclasses.replace(base, join_destination_side="right").join_inverted is True

    def test_a_plan_step_round_trips_positionally(self) -> None:
        step = dataclasses.replace(
            self._join_step(),
            input_feature_names=("plan_info_left_val",),
            join_destination_side="left",
            join_token=UUID("00000000-0000-4000-8000-000000000001"),
            declared_left_frameworks=(PandasDataFrame,),
            declared_right_frameworks=(PyArrowTable,),
        )

        assert PlanStep(*dataclasses.astuple(step)) == step
        assert PlanStep(*[getattr(step, field.name) for field in dataclasses.fields(PlanStep)]) == step

    def test_declared_framework_name_properties_mirror_the_classes(self) -> None:
        base = self._join_step()

        assert base.declared_left_framework_names == ()
        assert base.declared_right_framework_names == ()

        step = dataclasses.replace(
            base, declared_left_frameworks=(PandasDataFrame,), declared_right_frameworks=(PyArrowTable,)
        )

        assert step.declared_left_framework_names == ("PandasDataFrame",)
        assert step.declared_right_framework_names == ("PyArrowTable",)


# ---------------------------------------------------------------------------
# build_plan_steps rejects unknown steps
# ---------------------------------------------------------------------------


class TestBuildPlanStepsRejectsUnknownSteps:
    """An unmapped step must be loud, not silently dropped: a plan missing a step is a lie."""

    def test_unknown_step_raises_value_error(self) -> None:
        unknown_plan: Any = [PlanInfoUnknownStep()]

        with pytest.raises(ValueError, match="PlanInfoUnknownStep"):
            build_plan_steps(unknown_plan)

    def test_unknown_step_raises_even_when_mixed_with_known_steps(self) -> None:
        session = _prepare_chained_session()
        assert session.engine is not None

        mixed_plan: Any = [*session.engine.execution_planner, PlanInfoUnknownStep()]

        with pytest.raises(ValueError):
            build_plan_steps(mixed_plan)


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
        assert source_step.feature_names == ("plan_info_sales",)
        assert source_step.feature_group is PlanInfoPandasSource
        assert source_step.compute_framework is PandasDataFrame

        # The concrete aggregation implementation on the selected framework.
        assert aggregation_step.feature_names == ("plan_info_sales__mean_aggr",)
        assert aggregation_step.feature_group is PandasAggregatedFeatureGroup
        assert aggregation_step.compute_framework is PandasDataFrame

        assert aggregation_step.feature_group_name == "PandasAggregatedFeatureGroup"
        assert aggregation_step.compute_framework_name == "PandasDataFrame"

    def test_compute_steps_carry_no_source_or_join_fields(self) -> None:
        """source_* and join_type are transform/join specific and stay None on compute steps."""
        plan = _prepare_chained_session().resolved_plan()

        for step in (step for step in plan if step.step_kind == "compute"):
            assert step.source_feature_group is None
            assert step.source_compute_framework is None
            assert step.join_type is None

    def test_feature_names_are_strings_and_sorted(self) -> None:
        """feature_names mirrors FeatureSet.get_all_names(): a sorted tuple of strings.

        Requested as ["plan_info_sales", "plan_info_price"], reported sorted.
        """
        session = mloda.prepare(
            ["plan_info_sales", "plan_info_price"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )

        plan = session.resolved_plan()
        source_steps = [step for step in plan if step.feature_group is PlanInfoPandasSource]
        assert len(source_steps) == 1

        names = source_steps[0].feature_names
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)
        assert names == ("plan_info_price", "plan_info_sales")
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
        assert "plan_info_sales__mean_aggr" in results[0].columns

    def test_resolved_plan_matches_the_plan_that_actually_executed(self) -> None:
        """The reported plan must be the plan that ran.

        ``resolved_plan()`` reads the engine's planner, while ``Engine.compute`` deep-copies that
        planner before executing. Comparing against the runner's copy is what proves the reported
        records describe the executed steps, not a plan that was silently rewritten on the way in.
        """
        session = _prepare_chained_session()

        session.run()

        assert session.runner is not None
        executed_plan = build_plan_steps(session.runner.execution_planner)

        assert executed_plan == session.resolved_plan()

    def test_join_plan_matches_the_plan_that_actually_executed(self) -> None:
        """Same claim on a plan that contains a join step."""
        session = _prepare_single_framework_join_session()

        session.run()

        assert session.runner is not None
        executed_plan = build_plan_steps(
            session.runner.execution_planner, session.runner.execution_planner.resolved_join_plan
        )

        assert executed_plan == session.resolved_plan()


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

    def test_explain_matches_prepare_resolved_plan_for_a_join_request(self) -> None:
        """The join token is fresh per planning run and plan order across sessions is not pinned, so this
        compares multisets rather than requiring the two resolutions to match exactly."""
        explained = mlodaAPI.explain(
            ["PlanInfoCrossConsumer"],
            compute_frameworks={PandasDataFrame, PyArrowTable},
            links={_cross_link()},
            plugin_collector=_CROSS_JOIN_PLUGINS,
        )
        prepared = _prepare_cross_framework_join_session().resolved_plan()

        assert [step for step in explained if step.step_kind == "join"], "the fixture must plan a join"
        assert Counter(explained) == Counter(prepared)

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
            ["plan_info_never_executed"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_NEVER_EXECUTES_PLUGINS,
        )

        assert len(explained) == 1
        assert explained[0].feature_group is PlanInfoNeverExecutes
        assert explained[0].feature_names == ("plan_info_never_executed",)

    def test_explain_is_reachable_from_the_mloda_alias(self) -> None:
        explained = mloda.explain(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )

        assert all(isinstance(step, PlanStep) for step in explained)

    def test_explain_takes_only_features_positionally(self) -> None:
        """Everything after ``features`` is keyword-only.

        ``run_all`` and ``explain`` have different parameter orders (``run_all``'s 5th positional
        is ``parallelization_modes``, ``explain``'s would be ``global_filter``). A caller who
        reshapes a ``run_all`` call into an ``explain`` call must get a TypeError, not a plan
        silently built from misassigned arguments.
        """
        # Typed as Any on purpose: the point is the runtime signature, not what mypy allows.
        explain: Any = mlodaAPI.explain

        with pytest.raises(TypeError):
            explain(_CHAINED_FEATURES, {PandasDataFrame})

    def test_explain_still_accepts_its_parameters_by_keyword(self) -> None:
        explained = mlodaAPI.explain(
            _CHAINED_FEATURES,
            compute_frameworks={PandasDataFrame},
            plugin_collector=_AGGREGATION_PLUGINS,
        )

        assert len(explained) == 2


# ---------------------------------------------------------------------------
# transform and join step mapping
# ---------------------------------------------------------------------------


class TestTransformStepMapping:
    """A PyArrow source feeding the pandas aggregation group produces a transform step."""

    def test_transform_step_maps_from_and_to_group_and_framework(self) -> None:
        session = mloda.prepare(
            ["plan_info_arrow_sales__mean_aggr"],
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
        assert transform.join_type is None

        assert transform.source_feature_group_name == "PlanInfoArrowSource"
        assert transform.source_compute_framework_name == "PyArrowTable"

    def test_transform_step_sits_between_its_compute_steps(self) -> None:
        session = mloda.prepare(
            ["plan_info_arrow_sales__mean_aggr"],
            compute_frameworks={PandasDataFrame, PyArrowTable},
            plugin_collector=_TRANSFORM_PLUGINS,
        )

        kinds = [step.step_kind for step in session.resolved_plan()]

        assert kinds == ["compute", "transform", "compute"]


class TestJoinStepMapping:
    """Two linked pandas sources consumed by one feature group produce a join step."""

    def test_join_step_reports_the_linked_feature_groups_and_join_type(self) -> None:
        plan = _prepare_single_framework_join_session().resolved_plan()

        join_steps = [step for step in plan if step.step_kind == "join"]
        assert len(join_steps) == 1, f"expected one join step, got {[s.step_kind for s in plan]}"

        join = join_steps[0]

        # A join record is not blank: it names the link's declared sides and its join type.
        assert join.feature_names == ()
        assert join.feature_group is PlanInfoLeftSource
        assert join.source_feature_group is PlanInfoRightSource
        assert join.join_type == "inner"

        assert join.feature_group_name == "PlanInfoLeftSource"
        assert join.source_feature_group_name == "PlanInfoRightSource"

        # Both sides live on pandas here; the cross-framework test below is the one with teeth.
        assert join.compute_framework is PandasDataFrame
        assert join.source_compute_framework is PandasDataFrame

    def test_join_plan_contains_both_sources_and_the_consumer(self) -> None:
        plan = _prepare_single_framework_join_session().resolved_plan()

        compute_groups = {step.feature_group for step in plan if step.step_kind == "compute"}
        assert compute_groups == {PlanInfoLeftSource, PlanInfoRightSource, PlanInfoJoinConsumer}

        # The join must be planned before the consumer that depends on it.
        kinds = [step.step_kind for step in plan]
        last_compute = len(kinds) - 1 - kinds[::-1].index("compute")
        assert kinds.index("join") < last_compute


class TestCrossFrameworkJoinStepMapping:
    """The join sides live on different frameworks, so left/right cannot be confused.

    ``ExecutionPlan.run_link`` picks the merge destination, which may be the link's declared right
    side. The record's frameworks are therefore the destination and the merged-in source, while
    ``feature_group``/``source_feature_group`` stay the link's declared sides.
    """

    def test_join_step_frameworks_are_destination_and_source(self) -> None:
        plan = _prepare_cross_framework_join_session().resolved_plan()

        join_steps = [step for step in plan if step.step_kind == "join"]
        assert len(join_steps) == 1, f"expected one join step, got {[s.step_kind for s in plan]}"

        join = join_steps[0]

        # Distinct classes: swapping them would flip these assertions.
        assert join.compute_framework is PandasDataFrame
        assert join.source_compute_framework is PyArrowTable
        assert join.compute_framework_name == "PandasDataFrame"
        assert join.source_compute_framework_name == "PyArrowTable"

        # The link's declared sides, independent of any framework swap.
        assert join.feature_names == ()
        assert join.feature_group is PlanInfoCrossLeftPandas
        assert join.source_feature_group is PlanInfoCrossRightArrow
        assert join.join_type == "inner"

    def test_cross_framework_join_plan_has_a_transform_into_the_join_destination(self) -> None:
        plan = _prepare_cross_framework_join_session().resolved_plan()

        transform_steps = [step for step in plan if step.step_kind == "transform"]
        assert len(transform_steps) == 1, f"expected one transform step, got {[s.step_kind for s in plan]}"

        transform = transform_steps[0]

        assert transform.feature_names == ()
        assert transform.compute_framework is PandasDataFrame
        assert transform.source_compute_framework is PyArrowTable
        assert transform.feature_group is PlanInfoCrossLeftPandas
        assert transform.source_feature_group is PlanInfoCrossRightArrow
        assert transform.join_type is None

    def test_cross_framework_join_plan_shape(self) -> None:
        plan = _prepare_cross_framework_join_session().resolved_plan()

        kinds = [step.step_kind for step in plan]
        assert kinds.count("join") == 1
        assert kinds.count("transform") == 1
        assert kinds.count("compute") == 3

        # The two sources are computed on their own frameworks before the join merges them.
        left_step = next(step for step in plan if step.feature_group is PlanInfoCrossLeftPandas)
        assert left_step.step_kind == "compute"
        assert left_step.compute_framework is PandasDataFrame
        assert set(left_step.feature_names) == {"plan_info_xjid", "plan_info_xleft_val"}

        right_step = next(step for step in plan if step.feature_group is PlanInfoCrossRightArrow)
        assert right_step.step_kind == "compute"
        assert right_step.compute_framework is PyArrowTable
        assert set(right_step.feature_names) == {"plan_info_xjid", "plan_info_xright_val"}

        assert kinds.index("join") < kinds.index("compute", kinds.index("join"))
        consumer_index = [step.feature_group for step in plan].index(PlanInfoCrossConsumer)
        assert kinds.index("join") < consumer_index


class TestJoinOrientationOnResolvedPlans:
    """A join step reports which declared side holds the merge destination, and its completion token."""

    def test_join_step_reports_a_left_destination(self) -> None:
        session = _prepare_cross_framework_join_session()
        assert session.engine is not None
        records = session.engine.execution_planner.resolved_join_plan.records
        assert len(records) == 1, "the fixture must plan exactly one join for this to say anything"
        join_steps = session.engine.execution_planner.joinstep_collection.collection
        assert len(join_steps) == 1

        join = next(step for step in session.resolved_plan() if step.step_kind == "join")

        assert join.join_destination_side == "left"
        assert join.join_inverted is False
        assert join.join_token == records[0].token
        assert join.join_token == next(iter(join_steps)).uuid

    def test_join_step_reports_the_declared_frameworks_per_side(self) -> None:
        """The left side is pandas-only, the right side pyarrow-only, per the fixture's declarations."""
        plan = _prepare_cross_framework_join_session().resolved_plan()
        join = next(step for step in plan if step.step_kind == "join")

        assert join.declared_left_frameworks == (PandasDataFrame,)
        assert join.declared_right_frameworks == (PyArrowTable,)
        assert join.declared_left_framework_names == ("PandasDataFrame",)
        assert join.declared_right_framework_names == ("PyArrowTable",)
        assert join.compute_framework in join.declared_left_frameworks

    def test_compute_and_transform_steps_carry_no_join_orientation(self) -> None:
        plan = _prepare_cross_framework_join_session().resolved_plan()

        non_join_steps = [step for step in plan if step.step_kind != "join"]
        assert {step.step_kind for step in non_join_steps} == {"compute", "transform"}

        for step in non_join_steps:
            assert step.join_destination_side is None
            assert step.join_inverted is None
            assert step.join_token is None
            assert step.declared_left_frameworks == ()
            assert step.declared_right_frameworks == ()

    def test_join_orientation_is_none_without_a_resolved_join_plan(self) -> None:
        """build_plan_steps also takes a bare list of steps, which carries no ResolvedJoinPlan to read."""
        session = _prepare_cross_framework_join_session()
        assert session.engine is not None

        plan = build_plan_steps(list(session.engine.execution_planner))

        join = next(step for step in plan if step.step_kind == "join")
        assert join.join_destination_side is None
        assert join.join_inverted is None
        assert join.join_token is None
        assert join.declared_left_frameworks == ()
        assert join.declared_right_frameworks == ()

    def test_an_execution_plan_alone_carries_no_orientation_either(self) -> None:
        """A bare ExecutionPlan, with no explicit resolved_join_plan argument, behaves like a bare list."""
        session = _prepare_cross_framework_join_session()
        assert session.engine is not None

        plan = build_plan_steps(session.engine.execution_planner)

        join = next(step for step in plan if step.step_kind == "join")
        assert join.join_destination_side is None
        assert join.join_inverted is None
        assert join.join_token is None
        assert join.declared_left_frameworks == ()
        assert join.declared_right_frameworks == ()

    def test_a_given_resolved_join_plan_fills_the_orientation(self) -> None:
        session = _prepare_cross_framework_join_session()
        assert session.engine is not None
        planner = session.engine.execution_planner
        records = planner.resolved_join_plan.records
        assert len(records) == 1, "the fixture must plan exactly one join for this to say anything"

        plan = build_plan_steps(list(planner), planner.resolved_join_plan)

        join = next(step for step in plan if step.step_kind == "join")
        assert join.join_destination_side == "left"
        assert join.join_token == records[0].token
        assert join.declared_left_frameworks == (PandasDataFrame,)
        assert join.declared_right_frameworks == (PyArrowTable,)

    def test_a_join_step_missing_from_the_given_plan_raises(self) -> None:
        session = _prepare_cross_framework_join_session()
        assert session.engine is not None
        planner = session.engine.execution_planner

        with pytest.raises(ValueError, match="no resolved join record"):
            build_plan_steps(list(planner), ResolvedJoinPlan((), ()))


class TestInvertedCrossFrameworkJoin:
    """A PyArrow consumer puts the merge destination on the link's declared right side."""

    def test_join_step_reports_a_right_destination(self) -> None:
        session = _prepare_inverted_cross_framework_join_session()
        assert session.engine is not None
        records = session.engine.execution_planner.resolved_join_plan.records
        assert len(records) == 1, "the fixture must plan exactly one join for this to say anything"

        join = next(step for step in session.resolved_plan() if step.step_kind == "join")

        assert join.join_destination_side == "right"
        assert join.join_inverted is True
        assert join.join_token == records[0].token

        # The declared sides stay declared, whichever side the destination landed on.
        assert join.feature_group is PlanInfoCrossLeftPandas
        assert join.source_feature_group is PlanInfoCrossRightArrow
        assert join.compute_framework is PyArrowTable
        assert join.source_compute_framework is PandasDataFrame

        # The declared sides' candidate frameworks stay fixed by the Link, regardless of
        # which side the destination lands on.
        assert join.declared_left_frameworks == (PandasDataFrame,)
        assert join.declared_right_frameworks == (PyArrowTable,)
        assert join.compute_framework in join.declared_right_frameworks

    def test_the_hop_into_an_inverted_join_names_the_destination_side(self) -> None:
        """The hop moves the declared left side into the declared right one, which holds the destination."""
        plan = _prepare_inverted_cross_framework_join_session().resolved_plan()

        kinds = [step.step_kind for step in plan]
        assert kinds.count("join") == 1
        assert kinds.count("transform") == 1

        transform_steps = [step for step in plan if step.step_kind == "transform"]
        assert len(transform_steps) == 1, f"expected one transform step, got {kinds}"

        transform = transform_steps[0]

        assert transform.feature_group is PlanInfoCrossRightArrow
        assert transform.source_feature_group is PlanInfoCrossLeftPandas
        assert transform.compute_framework is PyArrowTable
        assert transform.source_compute_framework is PandasDataFrame


# ---------------------------------------------------------------------------
# requested vs engine-injected feature names
# ---------------------------------------------------------------------------


class TestRequestedAndInjectedForChainedFeature:
    """Only the chained ``*__mean_aggr`` name was requested: the source it chains off is injected."""

    def test_source_step_features_are_all_injected(self) -> None:
        plan = _prepare_chained_session().resolved_plan()

        source_step = next(step for step in plan if step.feature_group is PlanInfoPandasSource)

        assert source_step.requested_feature_names == ()
        assert source_step.injected_feature_names == source_step.feature_names
        assert source_step.injected_feature_names == ("plan_info_sales",)

    def test_aggregation_step_features_are_all_requested(self) -> None:
        plan = _prepare_chained_session().resolved_plan()

        aggregation_step = next(step for step in plan if step.feature_group is PandasAggregatedFeatureGroup)

        assert aggregation_step.requested_feature_names == aggregation_step.feature_names
        assert aggregation_step.requested_feature_names == ("plan_info_sales__mean_aggr",)
        assert aggregation_step.injected_feature_names == ()


class TestRequestedAndInjectedForJoinPlans:
    """Join and transform steps carry no names; compute steps partition feature_names exactly."""

    def test_join_and_transform_steps_keep_both_fields_empty(self) -> None:
        plan = _prepare_cross_framework_join_session().resolved_plan()

        non_compute_steps = [step for step in plan if step.step_kind != "compute"]
        assert {step.step_kind for step in non_compute_steps} == {"join", "transform"}

        for step in non_compute_steps:
            assert step.requested_feature_names == ()
            assert step.injected_feature_names == ()

    def test_compute_steps_partition_feature_names_disjointly(self) -> None:
        """requested and injected are disjoint, sorted, and their union is feature_names exactly."""
        plans = [
            _prepare_single_framework_join_session().resolved_plan(),
            _prepare_cross_framework_join_session().resolved_plan(),
        ]

        for plan in plans:
            for step in (step for step in plan if step.step_kind == "compute"):
                requested = set(step.requested_feature_names)
                injected = set(step.injected_feature_names)

                assert requested.isdisjoint(injected)
                assert tuple(sorted(requested | injected)) == step.feature_names
                assert list(step.requested_feature_names) == sorted(requested)
                assert list(step.injected_feature_names) == sorted(injected)

    def test_link_index_feature_is_injected_not_requested(self) -> None:
        """Only ``PlanInfoCrossConsumer`` was requested: the link's index feature on the source
        compute step is engine-injected, and the consumer's own name is the requested one."""
        plan = _prepare_cross_framework_join_session().resolved_plan()

        left_step = next(step for step in plan if step.feature_group is PlanInfoCrossLeftPandas)
        assert left_step.requested_feature_names == ()
        assert "plan_info_xjid" in left_step.injected_feature_names
        assert "plan_info_xjid" not in left_step.requested_feature_names
        assert left_step.injected_feature_names == left_step.feature_names

        consumer_step = next(step for step in plan if step.feature_group is PlanInfoCrossConsumer)
        assert consumer_step.requested_feature_names == ("PlanInfoCrossConsumer",)
        assert consumer_step.injected_feature_names == ()


# ---------------------------------------------------------------------------
# declared input feature names
# ---------------------------------------------------------------------------


class TestBuildPlanStepsInputFeatureNames:
    """build_plan_steps reads a compute step's FeatureSet.declared_input_feature_names, sorted."""

    @staticmethod
    def _source_compute_step() -> Any:
        """The root source's raw FeatureGroupStep, from a session nobody else shares."""
        compute_steps = _raw_steps_by_kind(_prepare_chained_session())["compute"]
        return next(step for step in compute_steps if step.feature_group is PlanInfoPandasSource)

    def test_declared_input_names_are_reported_sorted(self) -> None:
        step = self._source_compute_step()
        step.features.declared_input_feature_names = frozenset({"b", "a"})

        assert build_plan_steps([step])[0].input_feature_names == ("a", "b")

    def test_undeclared_input_names_yield_an_empty_tuple(self) -> None:
        step = self._source_compute_step()
        step.features.declared_input_feature_names = None

        assert build_plan_steps([step])[0].input_feature_names == ()

    def test_a_root_feature_group_step_is_planned_without_declared_inputs(self) -> None:
        """PlanInfoPandasSource is a root group, so the planner leaves its declared inputs at None."""
        step = self._source_compute_step()

        assert step.features.declared_input_feature_names is None
        assert build_plan_steps([step])[0].input_feature_names == ()

    def test_the_steps_own_feature_names_are_not_subtracted(self) -> None:
        step = self._source_compute_step()
        step.features.declared_input_feature_names = frozenset({"plan_info_sales", "plan_info_price"})

        record = build_plan_steps([step])[0]

        assert record.feature_names == ("plan_info_sales",)
        assert record.input_feature_names == ("plan_info_price", "plan_info_sales")

    def test_join_and_transform_steps_report_no_input_features(self) -> None:
        raw_steps = _raw_steps_by_kind(_prepare_cross_framework_join_session())
        assert raw_steps["join"], "the fixture must plan a join"
        assert raw_steps["transform"], "the fixture must plan a transform"

        for step in raw_steps["join"] + raw_steps["transform"]:
            assert build_plan_steps([step])[0].input_feature_names == ()


class TestInputFeatureNamesForChainedFeature:
    """The aggregation step names the source it chains off; the root source names nothing."""

    def test_derived_step_reports_the_source_feature_it_declares(self) -> None:
        plan = _prepare_chained_session().resolved_plan()

        aggregation_step = next(step for step in plan if step.feature_group is PandasAggregatedFeatureGroup)

        assert aggregation_step.input_feature_names == ("plan_info_sales",)

    def test_root_step_reports_no_input_features(self) -> None:
        plan = _prepare_chained_session().resolved_plan()

        source_step = next(step for step in plan if step.feature_group is PlanInfoPandasSource)

        assert source_step.input_feature_names == ()

    def test_input_feature_names_are_unchanged_by_run(self) -> None:
        session = _prepare_chained_session()

        before_run = [step.input_feature_names for step in session.resolved_plan()]
        session.run()
        after_run = [step.input_feature_names for step in session.resolved_plan()]

        assert after_run == before_run
        assert ("plan_info_sales",) in after_run


class TestInputFeatureNamesMatchTheRuntimeHookContext:
    """input_feature_names is the prepare-time twin of the run-time HookContext.input_features."""

    def test_every_compute_step_matches_its_calculate_hook_context(self) -> None:
        recorder = PlanInfoCalculateHookRecorder()
        session = _prepare_chained_session_with(recorder)

        session.run()

        compute_steps = [step for step in session.resolved_plan() if step.step_kind == "compute"]
        assert len(compute_steps) == 2
        assert any(step.input_feature_names for step in compute_steps), "an all-empty comparison proves nothing"

        hook_inputs = recorder.input_features_by_step()
        assert len(hook_inputs) == len(compute_steps), "every compute step must be hooked exactly once"

        for step in compute_steps:
            assert step.feature_group is not None
            key = (f"{step.feature_group.__module__}.{step.feature_group.__qualname__}", step.feature_names)
            assert key in hook_inputs, f"no calculate hook captured for {key}"
            assert frozenset(step.input_feature_names) == hook_inputs[key]


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
