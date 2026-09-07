"""End-to-end regression: add_tfs must not collapse two Pandas->PyArrow transform hops that
share a from/to-framework + from/to-feature-group shape but pull from genuinely different parent
features. Before the fix this either crashes at runtime or silently serves one request the
other's data.
"""

from typing import Any

import pandas as pd

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# ---------------------------------------------------------------------------
# Fixture feature groups
# ---------------------------------------------------------------------------


class DedupParentsRootFG(FeatureGroup):
    """Pandas root exposing two distinct columns."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"dedup_col_x", "dedup_col_y"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Return only the requested column(s): a compute framework's data holds what was
        # asked for, not everything DataCreator could produce.
        values = {"dedup_col_x": [1, 2, 3], "dedup_col_y": [100, 200, 300]}
        return pd.DataFrame({name: values[name] for name in features.get_all_names()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class DedupParentsConsumerFG(FeatureGroup):
    """PyArrow consumer with two Options-gated requests, each pulling a DIFFERENT Pandas column
    as its own single parent, forcing two same-shaped Pandas->PyArrow transform hops."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        variant = options.get("dedup_variant")
        if variant == "x":
            return {Feature("dedup_col_x")}
        if variant == "y":
            return {Feature("dedup_col_y")}
        raise ValueError(f"DedupParentsConsumerFG requires options={{'dedup_variant': 'x' | 'y'}}, got {variant!r}")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "dedup_col_x" in data.column_names:
            return data.append_column("dedup_result", data["dedup_col_x"])
        if "dedup_col_y" in data.column_names:
            return data.append_column("dedup_result", data["dedup_col_y"])
        raise ValueError(f"Neither dedup_col_x nor dedup_col_y present in {data.column_names}")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        # is_root() is False for both variant requests (input_features declares a parent),
        # so the DataCreator root-match path does not apply; name it here.
        return {"dedup_result"}


_PLUGINS = PluginCollector.enabled_feature_groups({DedupParentsRootFG, DedupParentsConsumerFG})


def _prepare_session() -> mlodaAPI:
    return mloda.prepare(
        [
            Feature("dedup_result", options=Options({"dedup_variant": "x"})),
            Feature("dedup_result", options=Options({"dedup_variant": "y"})),
        ],
        compute_frameworks={PandasDataFrame, PyArrowTable},
        plugin_collector=_PLUGINS,
    )


def _transform_steps(session: mlodaAPI) -> list[TransformFrameworkStep]:
    assert session.engine is not None
    return [step for step in session.engine.execution_planner if isinstance(step, TransformFrameworkStep)]


# ---------------------------------------------------------------------------
# Planning-time: the two hops must not collapse into one
# ---------------------------------------------------------------------------


def test_two_same_shaped_hops_from_different_parents_stay_distinct_in_the_plan() -> None:
    session = _prepare_session()

    plain_hops = [
        step
        for step in _transform_steps(session)
        if step.from_feature_group is DedupParentsRootFG and step.to_feature_group is DedupParentsConsumerFG
    ]
    assert len(plain_hops) == 2, (
        f"expected two separate transform hops for the two genuinely different parents "
        f"(dedup_col_x vs dedup_col_y), got: {[(s.uuid, s.required_uuids) for s in plain_hops]}"
    )

    required_uuid_sets = [frozenset(step.required_uuids) for step in plain_hops]
    assert required_uuid_sets[0] != required_uuid_sets[1], (
        f"the two hops must not share required_uuids (i.e. must not have collapsed into one hop "
        f"that only ever moves one physical source's data); got: {required_uuid_sets}"
    )

    assert session.engine is not None
    consumer_steps = [
        step
        for step in session.engine.execution_planner
        if isinstance(step, FeatureGroupStep) and step.feature_group is DedupParentsConsumerFG
    ]
    assert len(consumer_steps) == 2
    for consumer_step in consumer_steps:
        assert len(consumer_step.tfs_ids) == 1, (
            f"each consumer step must depend on exactly its own hop, not a hop shared with the "
            f"other variant; got tfs_ids={consumer_step.tfs_ids}"
        )


# ---------------------------------------------------------------------------
# Run-time: no crash, no cross-contaminated data
# ---------------------------------------------------------------------------


def test_running_the_plan_does_not_crash_and_each_result_carries_its_own_source_column() -> None:
    session = _prepare_session()
    results = session.run()

    dedup_results = [result for result in results if "dedup_result" in result.column_names]
    assert len(dedup_results) == 2, f"expected two distinct 'dedup_result' results, got: {results}"

    value_tuples = sorted(tuple(result["dedup_result"].to_pylist()) for result in dedup_results)
    assert value_tuples == [(1, 2, 3), (100, 200, 300)], (
        f"each request must carry its OWN correctly-transformed source column's values, not "
        f"empty/None, and not the other request's data; got: {value_tuples}"
    )
