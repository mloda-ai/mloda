"""Prints one json line: per planner shape, the outcome a fresh interpreter plans and runs.
No test_ prefix, so pytest never collects it; add a shape by extending SHAPES.
"""

import json
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer import *  # noqa: F403
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


def _sum_columns(data: Any, name: str, columns: list[str]) -> Any:
    if hasattr(data, "iloc"):
        data[name] = sum(data[column] for column in columns)
        return data
    total = data[columns[0]]
    for column in columns[1:]:
        total = pc.add(total, data[column])
    return data.append_column(name, total)


class LsRootX(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ls_x"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"ls_xid": [1, 2, 3], "ls_x": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("ls_xid",))]


class LsRootW(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ls_w"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"ls_xid": [1, 2, 3], "ls_w": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("ls_xid",))]


class LsD(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_x")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_d", ["ls_x"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_d"}


class LsE(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_x"), Feature("ls_w")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_e", ["ls_x", "ls_w"])

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_e"}


class LsYf(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_c1")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_yf", ["ls_c1"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_yf"}


class LsC(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if str(feature_name) == "ls_c1":
            return {Feature("ls_d")}
        return {Feature("ls_d"), Feature("ls_e"), Feature("ls_yf")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "ls_c1" in features.get_all_names():
            data = _sum_columns(data, "ls_c1", ["ls_d"])
        if "ls_c2" in features.get_all_names():
            data = _sum_columns(data, "ls_c2", ["ls_d", "ls_e", "ls_yf"])
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_c1", "ls_c2"}


class LsY(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_x")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_y", ["ls_x"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_y"}


class LsZ(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_x"), Feature("ls_w")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_z", ["ls_x", "ls_w"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_z"}


class LsYzC(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_y"), Feature("ls_z")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_yzc", ["ls_y", "ls_z"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_yzc"}


class LsP(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_x")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_p", ["ls_x"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_p"}


class LsQ(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_p"), Feature("ls_w")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_q", ["ls_p", "ls_w"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_q"}


class LsPqC(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ls_p"), Feature("ls_q")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _sum_columns(data, "ls_pqc", ["ls_p", "ls_q"])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ls_pqc"}


def _run(
    requested: list[str], groups: set[type[FeatureGroup]], links: set[Link], column: str | None = None
) -> dict[str, str]:
    """A plan-time reject shows up as the caught exception's message; column adds its sorted values."""
    try:
        results = mloda.run_all(
            [Feature(name) for name in requested],
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            links=links,
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
            parallelization_modes={ParallelizationMode.SYNC},
        )
    except ValueError as error:
        return {"outcome": "rejected", "error": str(error)}
    found = {"outcome": "accepted", "error": ""}
    if column is not None:
        found["values"] = json.dumps([sorted(list(result[column])) for result in results if column in result])
    return found


def hop_parent_shape() -> dict[str, str]:
    """A hop parent reaches a join side only through a framework change, so its join is missing."""
    link = Link.inner(JoinSpec(LsRootX, "ls_xid"), JoinSpec(LsRootW, "ls_xid"))
    return _run(["ls_c2"], {LsRootX, LsRootW, LsD, LsE, LsYf, LsC}, {link})


def twin_sibling_shape() -> dict[str, str]:
    """Y and Z both read join side X, Z also reads W, and a pandas-only C reads Y and Z."""
    link = Link.inner(JoinSpec(LsRootX, "ls_xid"), JoinSpec(LsRootW, "ls_xid"))
    return _run(["ls_yzc"], {LsRootX, LsRootW, LsY, LsZ, LsYzC}, {link}, "ls_yzc")


def twin_chain_shape() -> dict[str, str]:
    """P reads join side X, Q reads P and W, and a pandas-only C reads P and Q."""
    link = Link.inner(JoinSpec(LsRootX, "ls_xid"), JoinSpec(LsRootW, "ls_xid"))
    return _run(["ls_pqc"], {LsRootX, LsRootW, LsP, LsQ, LsPqC}, {link}, "ls_pqc")


SHAPES: dict[str, Callable[[], dict[str, str]]] = {
    "hop_parent": hop_parent_shape,
    "twin_sibling": twin_sibling_shape,
    "twin_chain": twin_chain_shape,
}


def collect() -> dict[str, str]:
    collected: dict[str, str] = {}
    for shape, build in SHAPES.items():
        for key, value in build().items():
            collected[f"{shape}_{key}"] = value
    return collected


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
