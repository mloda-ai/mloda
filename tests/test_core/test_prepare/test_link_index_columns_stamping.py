"""FeatureSet.link_index_columns must be stamped with the union of both sides' link index columns."""

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


LEFT_INDEX_COLUMN = "link_stamp_left_key"
RIGHT_INDEX_COLUMN = "link_stamp_right_key"
LEFT_INDEX = Index((LEFT_INDEX_COLUMN,))
RIGHT_INDEX = Index((RIGHT_INDEX_COLUMN,))


class LinkStampBaseFG(FeatureGroup):
    """Unmatchable base so a leaked subclass stays invisible to feature resolution."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class LinkStampLeftFG(LinkStampBaseFG):
    pass


class LinkStampRightFG(LinkStampBaseFG):
    pass


class LinkStampUnrelatedFG(LinkStampBaseFG):
    pass


def _feature(name: str, cfw: type[ComputeFramework], index: Index | None = None) -> Feature:
    feature = Feature(name, index=index)
    feature.compute_frameworks = {cfw}
    return feature


def _fg_step(fg: type[FeatureGroup], features: list[Feature], cfw: type[ComputeFramework]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, set(), cfw)


def test_step_read_by_a_link_gets_both_sides_index_columns_stamped() -> None:
    left_payload = _feature("link_stamp_left_payload", PyArrowTable)
    left_step = _fg_step(LinkStampLeftFG, [left_payload], PyArrowTable)

    right_payload = _feature("link_stamp_right_payload", PandasDataFrame)
    right_step = _fg_step(LinkStampRightFG, [right_payload], PandasDataFrame)

    link = Link.inner(JoinSpec(LinkStampLeftFG, LEFT_INDEX), JoinSpec(LinkStampRightFG, RIGHT_INDEX))
    join_step = JoinStep(
        link=link,
        destination_framework=PyArrowTable,
        source_framework=PandasDataFrame,
        required_uuids={left_payload.uuid, right_payload.uuid},
        destination_framework_uuids={left_payload.uuid},
        source_framework_uuids={right_payload.uuid},
    )

    steps: list[JoinStep | FeatureGroupStep] = [left_step, right_step, join_step]

    ExecutionPlan()._stamp_link_index_columns(steps)

    expected = frozenset({LEFT_INDEX_COLUMN, RIGHT_INDEX_COLUMN})
    assert left_step.features.link_index_columns == expected
    assert right_step.features.link_index_columns == expected


def test_step_not_read_by_any_link_keeps_empty_frozenset() -> None:
    left_payload = _feature("link_stamp_left_payload", PyArrowTable)
    left_step = _fg_step(LinkStampLeftFG, [left_payload], PyArrowTable)

    right_payload = _feature("link_stamp_right_payload", PandasDataFrame)
    right_step = _fg_step(LinkStampRightFG, [right_payload], PandasDataFrame)

    unrelated_payload = _feature("link_stamp_unrelated_payload", PyArrowTable)
    unrelated_step = _fg_step(LinkStampUnrelatedFG, [unrelated_payload], PyArrowTable)

    link = Link.inner(JoinSpec(LinkStampLeftFG, LEFT_INDEX), JoinSpec(LinkStampRightFG, RIGHT_INDEX))
    join_step = JoinStep(
        link=link,
        destination_framework=PyArrowTable,
        source_framework=PandasDataFrame,
        required_uuids={left_payload.uuid, right_payload.uuid},
        destination_framework_uuids={left_payload.uuid},
        source_framework_uuids={right_payload.uuid},
    )

    steps: list[JoinStep | FeatureGroupStep] = [left_step, right_step, unrelated_step, join_step]

    ExecutionPlan()._stamp_link_index_columns(steps)

    assert unrelated_step.features.link_index_columns == frozenset()
