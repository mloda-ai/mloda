"""Prints one json line with the result a fresh interpreter plans and runs for a
consumer reading two plain hops out of one PythonDictFramework source, whose
from_feature_group classes are related only by subclassing (P3B(P3A)) for code
reuse, not by any shared parent data. No test_ prefix, so pytest never collects it; the
regression test runs it as a script under several PYTHONHASHSEED values.
"""

import json
from typing import Any

import pyarrow.compute as pc

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class P3Root(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"p3_r"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"p3_r": [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}


class P3GateA(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("p3_r")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "p3_ga": [v + 1 for v in data["p3_r"]]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"p3_ga"}


class P3GateB(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("p3_r")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "p3_gb": [v + 1 for v in data["p3_r"]]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"p3_gb"}


class P3A(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("p3_ga")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "p3_a": [v + 1 for v in data["p3_ga"]]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"p3_a"}


class P3B(P3A):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("p3_gb")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "p3_b": [v + 1 for v in data["p3_gb"]]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"p3_b"}


class P3Consumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("p3_a"), Feature("p3_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("p3_x", pc.add(data["p3_a"], data["p3_b"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"p3_x"}


def collect() -> dict[str, str]:
    """Plans and runs the repro shape; a plan-time reject shows up as the caught exception's message."""
    try:
        result = mloda.run_all(
            [Feature("p3_x")],
            compute_frameworks={PythonDictFramework, PyArrowTable},
            plugin_collector=PluginCollector.enabled_feature_groups({P3Root, P3GateA, P3GateB, P3A, P3B, P3Consumer}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
    except ValueError as error:
        return {"outcome": "rejected", "error": str(error)}

    return {"outcome": "accepted", "p3_x": str(result[0].column("p3_x").to_pylist())}


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
