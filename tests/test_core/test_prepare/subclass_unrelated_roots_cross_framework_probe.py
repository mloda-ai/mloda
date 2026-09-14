"""Prints one json line with the result a fresh interpreter plans and runs for a consumer
reading two ROOT DataCreator features whose feature groups are related only by subclassing
(ScRootB(ScRootA)) for code reuse, share NO physical lineage at all, and sit on two DIFFERENT
compute frameworks. No test_ prefix, so pytest never collects it; the regression test runs it
as a script under several PYTHONHASHSEED values.
"""

import json
from typing import Any

import pyarrow.compute as pc

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class ScRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sc_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sc_a": [1, 2, 3]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}


class ScRootB(ScRootA):
    """Subclass of ScRootA for code reuse only; independent DataCreator, different framework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sc_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"sc_b": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class ScConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sc_a"), Feature("sc_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("sc_x", pc.add(data["sc_a"], data["sc_b"]))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sc_x"}


def collect() -> dict[str, str]:
    """Plans and runs the repro shape.

    ``prepare_tfs_right_cfw`` (compute_framework_executor.py) and ``_drop_tfs_source_if_possible``
    (run.py) each grab an arbitrary ``next(iter(step.required_uuids))`` member. #1426's
    subclass-cluster widening (execution_plan.py) can put a SIBLING hop's parent, on a different
    framework, into that same set, so the arbitrary pick can raise
    "cfw_uuid should not be none in prepare_tfs" - a hard crash, classified separately below from
    the pre-existing, graceful "missing Links" rejection this shape produced before the #1426 fix.
    """
    try:
        result = mloda.run_all(
            [Feature("sc_x")],
            compute_frameworks={PythonDictFramework, PandasDataFrame, PyArrowTable},
            plugin_collector=PluginCollector.enabled_feature_groups({ScRootA, ScRootB, ScConsumer}),
            parallelization_modes={ParallelizationMode.SYNC},
        )
    except ValueError as error:
        if "cfw_uuid should not be none" in str(error):
            return {"outcome": "crashed", "error": str(error)}
        return {"outcome": "rejected", "error": str(error)}

    return {"outcome": "accepted", "sc_x": str(result[0].column("sc_x").to_pylist())}


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
