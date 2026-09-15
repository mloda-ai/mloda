"""Prints one json line with the result a fresh interpreter plans and runs for a
consumer with a subclass-linked plain hop (case override) plus a join-served parent.
No test_ prefix, so pytest never collects it; the regression test runs it as a script
under several PYTHONHASHSEED values.
"""

import json
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

_IDX = Index(("subclass_linked_key",))


class SubclassLinkedDerived(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("subclass_linked_s")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {**data, "subclass_linked_t": [v * 10 for v in data["subclass_linked_s"]]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subclass_linked_t"}


class SubclassLinkedSource(SubclassLinkedDerived):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"subclass_linked_key", "subclass_linked_s"})

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [_IDX]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"subclass_linked_key": [1, 2, 3], "subclass_linked_s": [1, 2, 3]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set()


class SubclassLinkedDest(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"subclass_linked_key", "subclass_linked_d"})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [_IDX]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"subclass_linked_key": [1, 2, 3], "subclass_linked_d": [100, 200, 300]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SubclassLinkedConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("subclass_linked_s"), Feature("subclass_linked_t"), Feature("subclass_linked_d")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(
            "subclass_linked_x",
            pc.add(pc.add(data["subclass_linked_s"], data["subclass_linked_t"]), data["subclass_linked_d"]),
        )

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subclass_linked_x"}


def collect() -> dict[str, str]:
    """Plans and runs the repro shape; a plan-time reject shows up as the caught exception's message."""
    try:
        result = mloda.run_all(
            [Feature("subclass_linked_x")],
            compute_frameworks={PythonDictFramework, PyArrowTable},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {SubclassLinkedSource, SubclassLinkedDerived, SubclassLinkedDest, SubclassLinkedConsumer}
            ),
            links={Link.inner(JoinSpec(SubclassLinkedDest, _IDX), JoinSpec(SubclassLinkedSource, _IDX))},
            parallelization_modes={ParallelizationMode.SYNC},
        )
    except ValueError as error:
        return {"outcome": "rejected", "error": str(error)}

    return {"outcome": "accepted", "subclass_linked_x": str(result[0].column("subclass_linked_x").to_pylist())}


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
