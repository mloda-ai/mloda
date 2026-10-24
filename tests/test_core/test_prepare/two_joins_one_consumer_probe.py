"""Prints one json line: the JoinStep order and the join-to-join ordering edges a fresh interpreter plans.
No test_ prefix, so pytest never collects it.
"""

import json
from typing import Any

from mloda.core.core.step.join_step import JoinStep
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, DataCreatorFeatureGroup
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

IDX = Index(("idx",))


class TjRootA(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tj_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"idx": ["x", "y", "z", "w"], "tj_a": [1, 2, 3, 4]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TjRootC(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tj_c"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"idx": ["y", "z", "w", "q"], "tj_c": [7, 8, 9, 10]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TjRootB(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tj_b"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"idx": ["x", "y", "z", "v", "u"], "tj_b": [10, 20, 30, 50, 60]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TjConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tj_a"), Feature("tj_c"), Feature("tj_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("TjConsumer", data.column("tj_a"))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _label(step: JoinStep) -> str:
    link = step.link
    return (
        f"{link.left_feature_group.__name__}-{link.right_feature_group.__name__}"
        f"|dest={step.destination_framework.get_class_name()}|src={step.source_framework.get_class_name()}"
    )


session = mloda.prepare(
    ["TjConsumer"],
    links={
        Link.inner(JoinSpec(TjRootA, IDX), JoinSpec(TjRootB, IDX)),
        Link.inner(JoinSpec(TjRootC, IDX), JoinSpec(TjRootB, IDX)),
    },
    compute_frameworks=[PyArrowTable],
    parallelization_modes={ParallelizationMode.SYNC},
    plugin_collector=PluginCollector.enabled_feature_groups({TjRootA, TjRootB, TjRootC, TjConsumer}),
)
assert session.engine is not None
joins = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
by_uuid = {step.uuid: step for step in joins}
edges = sorted(
    f"{_label(by_uuid[dependency])} => {_label(step)}"
    for step in joins
    for dependency in step.required_uuids
    if dependency in by_uuid
)
print(json.dumps({"order": json.dumps([_label(step) for step in joins]), "edges": json.dumps(edges)}))
