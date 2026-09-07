"""End-to-end regression: add_tfs must collapse two Pandas->PyArrow transform hops that share a
from/to-framework + from/to-feature-group shape AND come from parents produced together by the
SAME owning FeatureGroupStep into ONE hop. Before the fix (keying the hop by the raw parent
feature uuid instead of the owning step's uuid), this builds a separate hop per feature even
though both physically live on the same source compute framework instance, leaking an orphaned
destination compute framework and, under multiprocessing, downloading the same table twice.
"""

from typing import Any

import pandas as pd
import pyarrow.compute as pc

from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# ---------------------------------------------------------------------------
# Fixture feature groups
# ---------------------------------------------------------------------------


class SharedSourceRootFG(FeatureGroup):
    """Pandas root exposing two columns that are always computed TOGETHER by one step."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"twocol_a", "twocol_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = {"twocol_a": [1, 2, 3], "twocol_b": [10, 20, 30]}
        return pd.DataFrame({name: values[name] for name in features.get_all_names()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SharedSourceConsumerFG(FeatureGroup):
    """PyArrow consumer pulling BOTH root columns in a single request, forcing one
    FeatureGroupStep on the root that produces both columns together."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("twocol_a"), Feature("twocol_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        total = pc.add(data["twocol_a"], data["twocol_b"])
        return data.append_column("shared_result", total)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        # is_root() is False (input_features declares parents), so the DataCreator root-match
        # path does not apply; name it here.
        return {"shared_result"}


_PLUGINS = PluginCollector.enabled_feature_groups({SharedSourceRootFG, SharedSourceConsumerFG})


def _prepare_session() -> mlodaAPI:
    return mloda.prepare(
        ["shared_result"],
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=_PLUGINS,
    )


def _transform_steps(session: mlodaAPI) -> list[TransformFrameworkStep]:
    assert session.engine is not None
    return [step for step in session.engine.execution_planner if isinstance(step, TransformFrameworkStep)]


# ---------------------------------------------------------------------------
# Planning-time: the two parents from ONE owning step must share ONE hop
# ---------------------------------------------------------------------------


def test_two_parents_from_the_same_owning_step_share_one_transform_hop() -> None:
    session = _prepare_session()

    hops = [
        step
        for step in _transform_steps(session)
        if step.from_feature_group is SharedSourceRootFG and step.to_feature_group is SharedSourceConsumerFG
    ]
    assert len(hops) == 1, (
        f"expected one shared transform hop for twocol_a and twocol_b, which are produced together "
        f"by the same owning FeatureGroupStep (same physical source compute framework instance), "
        f"got: {[(s.uuid, s.required_uuids) for s in hops]}"
    )


# ---------------------------------------------------------------------------
# Run-time: no crash, correct combined result
# ---------------------------------------------------------------------------


def test_running_the_plan_produces_the_correct_combined_result() -> None:
    session = _prepare_session()
    results = session.run()

    shared_results = [result for result in results if "shared_result" in result.column_names]
    assert len(shared_results) == 1, f"expected exactly one 'shared_result' result, got: {results}"

    assert shared_results[0]["shared_result"].to_pylist() == [11, 22, 33], (
        f"expected twocol_a + twocol_b element-wise, got: {shared_results[0]['shared_result'].to_pylist()}"
    )
