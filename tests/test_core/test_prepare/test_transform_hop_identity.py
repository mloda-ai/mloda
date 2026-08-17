"""Pins that ``TransformFrameworkStep`` identity includes ``link_id``, so a join's transform hop
no longer collides with a same-shaped plain feature-group hop in ``ExecutionPlan.add_tfs``. Before
that, the dedup dropped one of the two hops, the JoinStep lost its transform-step dependency, and
``run()`` failed to find its source data.
"""

from typing import Any, Optional
from uuid import UUID, uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# ---------------------------------------------------------------------------
# Fixture feature groups
# ---------------------------------------------------------------------------


class HopIdA(FeatureGroup):
    """Pandas root: the join's declared left side."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"hop_id_jid", "hop_id_a_val", "hop_id_a_helper"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"hop_id_jid": [1, 2, 3], "hop_id_a_val": [1, 2, 3], "hop_id_a_helper": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("hop_id_jid",))]


class HopIdB(FeatureGroup):
    """PyArrow root (join's declared right side); its own index feature doubles as an option-gated derived one."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"hop_id_bjid", "hop_id_b_val"})

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        if options.get("hop_id_variant") == "other":
            return {Feature("hop_id_a_helper", options=Options({"hop_id_variant": "other"}))}
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if data is not None and "hop_id_a_helper" in data.column_names:
            product = pc.multiply(data["hop_id_a_helper"], pa.scalar(2))
            return data.append_column("hop_id_bjid", product)
        return pa.table({"hop_id_bjid": [1, 2, 3], "hop_id_b_val": [15, 25, 35]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("hop_id_bjid",))]

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        # is_root() is False for the options={"hop_id_variant": "other"} request (input_features
        # declares a parent for it), so the DataCreator root-match path does not apply; name it here.
        return {"hop_id_bjid"}


class HopIdC(FeatureGroup):
    """PyArrow consumer of both join sides: forces the join's own transform hop into the plan."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("hop_id_a_val"), Feature("hop_id_b_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        product = pc.multiply(data["hop_id_a_val"], data["hop_id_b_val"])
        return data.append_column("HopIdC", product)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_PLUGINS = PluginCollector.enabled_feature_groups({HopIdA, HopIdB, HopIdC})


def _link() -> Link:
    return Link.inner(JoinSpec(HopIdA, "hop_id_jid"), JoinSpec(HopIdB, "hop_id_bjid"))


def _prepare_session() -> mlodaAPI:
    return mloda.prepare(
        ["HopIdC", Feature("hop_id_bjid", options=Options({"hop_id_variant": "other"}))],
        compute_frameworks={PandasDataFrame, PyArrowTable},
        links={_link()},
        plugin_collector=_PLUGINS,
    )


def _transform_steps(session: mlodaAPI) -> list[TransformFrameworkStep]:
    assert session.engine is not None
    return [step for step in session.engine.execution_planner if isinstance(step, TransformFrameworkStep)]


def _join_step(session: mlodaAPI) -> JoinStep:
    assert session.engine is not None
    return next(step for step in session.engine.execution_planner if isinstance(step, JoinStep))


def _optioned_hop_id_bjid_step(session: mlodaAPI) -> FeatureGroupStep:
    """The HopIdB step computing the option-gated ``hop_id_bjid`` request, distinct from its root step."""
    assert session.engine is not None
    return next(
        step
        for step in session.engine.execution_planner
        if isinstance(step, FeatureGroupStep)
        and step.feature_group is HopIdB
        and step.features.options is not None
        and step.features.options.get("hop_id_variant") == "other"
    )


def _describe(step: TransformFrameworkStep) -> tuple[str, str, str, str, Optional[UUID]]:
    return (
        step.from_framework.get_class_name(),
        step.to_framework.get_class_name(),
        step.from_feature_group.get_class_name(),
        step.to_feature_group.get_class_name(),
        step.link_id,
    )


# ---------------------------------------------------------------------------
# End-to-end: the join's transform hop must survive planning
# ---------------------------------------------------------------------------


class TestTheJoinHopSurvivesASameShapedFeatureGroupHop:
    def test_the_join_hop_survives_a_same_shaped_feature_group_hop(self) -> None:
        session = _prepare_session()
        transform_steps = _transform_steps(session)
        described = [_describe(step) for step in transform_steps]

        assert len(transform_steps) == 2, f"expected two transform hops (join + plain FG), got: {described}"

        join_step = _join_step(session)
        join_hops = [step for step in transform_steps if step.link_id == join_step.link.uuid]
        assert len(join_hops) == 1, f"expected exactly one hop tied to the join, got: {described}"

        assert join_hops[0].uuid in join_step.required_uuids, (
            f"the join step must depend on its own transform hop; "
            f"required_uuids={join_step.required_uuids}, hops={described}"
        )

        optioned_step = _optioned_hop_id_bjid_step(session)
        feature_names = {str(feature.name) for feature in optioned_step.features.features}
        assert feature_names == {"hop_id_bjid"}, (
            f"the auto-injected index companion must merge into the requested feature itself, "
            f"leaving the step with exactly one member; got: {feature_names}"
        )


class TestTheJoinHopAndTheFeatureGroupHopShareAShapeButNotAnIdentity:
    """Both planned hops share one 4-tuple shape and differ only in link_id; that must not make them equal."""

    def test_the_join_hop_and_the_feature_group_hop_share_a_shape_but_not_an_identity(self) -> None:
        session = _prepare_session()
        transform_steps = _transform_steps(session)
        described = [_describe(step) for step in transform_steps]
        assert len(transform_steps) == 2, f"expected two transform hops (join + plain FG), got: {described}"

        join_link_uuid = _join_step(session).link.uuid
        join_hop = next(step for step in transform_steps if step.link_id == join_link_uuid)
        plain_hop = next(step for step in transform_steps if step.link_id is None)

        shape = (join_hop.from_framework, join_hop.to_framework, join_hop.from_feature_group, join_hop.to_feature_group)
        plain_shape = (
            plain_hop.from_framework,
            plain_hop.to_framework,
            plain_hop.from_feature_group,
            plain_hop.to_feature_group,
        )
        assert shape == plain_shape, f"the fixture must produce two same-shaped hops, got: {described}"

        assert join_hop.link_id != plain_hop.link_id
        assert join_hop != plain_hop, "a hop with a different link_id must not compare equal"
        assert hash(join_hop) != hash(plain_hop), "a hop with a different link_id must not hash equal"


class TestTheInvertedJoinRunsNextToASameShapedHop:
    def test_the_inverted_join_runs_next_to_a_same_shaped_hop(self) -> None:
        session = _prepare_session()
        results = session.run()

        hop_id_c_result = next(result for result in results if "HopIdC" in result.column_names)
        assert sorted(hop_id_c_result["HopIdC"].to_pylist()) == [15, 50, 105]


# ---------------------------------------------------------------------------
# TransformFrameworkStep identity must include link_id
# ---------------------------------------------------------------------------


class TestTransformFrameworkStepIdentityIncludesLinkId:
    """A join's transform hop and a plain feature-group hop of the same shape must not collide."""

    @staticmethod
    def _step(link_id: Optional[UUID]) -> TransformFrameworkStep:
        return TransformFrameworkStep(
            from_framework=PandasDataFrame,
            to_framework=PyArrowTable,
            required_uuids=set(),
            from_feature_group=HopIdA,
            to_feature_group=HopIdB,
            link_id=link_id,
        )

    def test_a_join_hop_and_a_plain_hop_of_the_same_shape_compare_unequal(self) -> None:
        join_hop = self._step(uuid4())
        plain_hop = self._step(None)

        assert join_hop != plain_hop
        assert hash(join_hop) != hash(plain_hop)

    def test_two_hops_of_the_same_link_compare_equal(self) -> None:
        link_id = uuid4()
        first = self._step(link_id)
        second = self._step(link_id)

        assert first == second
        assert hash(first) == hash(second)


class TestTransformFrameworkStepIdentityIncludesSourceStepUuid:
    """Two hops of the same shape must key on the owning FeatureGroupStep's uuid
    (``source_step_uuid``), not collide across different owning steps, and not collide with a
    join hop of the same shape either."""

    @staticmethod
    def _step(source_step_uuid: Optional[UUID]) -> TransformFrameworkStep:
        return TransformFrameworkStep(
            from_framework=PandasDataFrame,
            to_framework=PyArrowTable,
            required_uuids=set(),
            from_feature_group=HopIdA,
            to_feature_group=HopIdB,
            source_step_uuid=source_step_uuid,
        )

    def test_two_hops_with_different_source_step_uuids_compare_unequal(self) -> None:
        first = self._step(uuid4())
        second = self._step(uuid4())

        assert first != second
        assert hash(first) != hash(second)

    def test_two_hops_with_the_same_source_step_uuid_compare_equal(self) -> None:
        source_step_uuid = uuid4()
        first = self._step(source_step_uuid)
        second = self._step(source_step_uuid)

        assert first == second
        assert hash(first) == hash(second)

    def test_a_join_hop_and_a_plain_hop_of_the_same_shape_compare_unequal(self) -> None:
        join_hop = TransformFrameworkStep(
            from_framework=PandasDataFrame,
            to_framework=PyArrowTable,
            required_uuids=set(),
            from_feature_group=HopIdA,
            to_feature_group=HopIdB,
            link_id=uuid4(),
            source_step_uuid=None,
        )
        plain_hop = TransformFrameworkStep(
            from_framework=PandasDataFrame,
            to_framework=PyArrowTable,
            required_uuids=set(),
            from_feature_group=HopIdA,
            to_feature_group=HopIdB,
            link_id=None,
            source_step_uuid=uuid4(),
        )

        assert join_hop != plain_hop
        assert hash(join_hop) != hash(plain_hop)
