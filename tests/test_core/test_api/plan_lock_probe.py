"""Prints the plan-lock text a fresh interpreter produces, for the hash-seed stability test.
Also prints the explain order with framework names and reasons."""

import json
from typing import Any

import pandas as pd
import pyarrow as pa

from mloda.core.api.plan_lock import _lock_text
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class LockProbeLeftPandas(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"lock_probe_jid", "lock_probe_left_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"lock_probe_jid": [1, 2, 3], "lock_probe_left_val": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("lock_probe_jid",))]


class LockProbeRightArrow(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"lock_probe_jid", "lock_probe_right_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"lock_probe_jid": [1, 2, 3], "lock_probe_right_val": [1.5, 2.5, 3.5]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("lock_probe_jid",))]


class LockProbeConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("lock_probe_left_val"), Feature("lock_probe_right_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["LockProbeConsumer"] = data["lock_probe_left_val"] * data["lock_probe_right_val"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class LockProbeAnyFramework(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"lock_probe_any_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"lock_probe_any_value": [1, 2, 3]})


def collect() -> dict[str, str]:
    link = Link.inner(JoinSpec(LockProbeLeftPandas, "lock_probe_jid"), JoinSpec(LockProbeRightArrow, "lock_probe_jid"))
    plugins = PluginCollector.enabled_feature_groups(
        {LockProbeLeftPandas, LockProbeRightArrow, LockProbeConsumer, LockProbeAnyFramework}
    )
    plan = mloda.explain(
        [
            "LockProbeConsumer",
            Feature("lock_probe_any_value", compute_framework="PandasDataFrame"),
            Feature("lock_probe_any_value", compute_framework="PyArrowTable"),
        ],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        links={link},
        plugin_collector=plugins,
    )
    order = [
        [step.step_kind, step.feature_group_name, step.compute_framework_name, step.compute_framework_reason]
        for step in plan
    ]
    return {"lock": _lock_text(plan), "order": json.dumps(order)}


if __name__ == "__main__":
    print(json.dumps(collect()))
