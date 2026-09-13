"""A link declared on base feature groups should inject keys into concrete subclasses, run end to end."""

from __future__ import annotations

from typing import Any

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


_PLKIR_A = {"plkir_a_key": [1, 2, 3, 4], "plkir_a_val": [10, 20, 30, 40]}
_PLKIR_B = {"plkir_b_key": [2, 3, 4, 5], "plkir_b_val": [200, 300, 400, 500]}


class PlkirBaseA(FeatureGroup):
    """Base side A: no input_data, so a link declared on it never resolves on its own."""


class PlkirBaseB(FeatureGroup):
    """Base side B: no input_data, so a link declared on it never resolves on its own."""


class PlkirSubA(PlkirBaseA):
    """Concrete side A: a missing plkir_a_key in the data would raise KeyError."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plkir_a_key", "plkir_a_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: _PLKIR_A[name] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlkirSubB(PlkirBaseB):
    """Concrete side B: a missing plkir_b_key in the data would raise KeyError."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plkir_b_key", "plkir_b_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: _PLKIR_B[name] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlkirJoinedFG(FeatureGroup):
    """Consumer needing one value column from each side, joined through the base-class link."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("plkir_a_val"), Feature("plkir_b_val")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        joined = data.to_pydict()
        return {
            cls.get_class_name(): [a + b for a, b in zip(joined["plkir_a_val"], joined["plkir_b_val"])],
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def test_base_class_link_joins_concrete_subclasses_end_to_end() -> None:
    """Without injection, no key reaches either subclass and the join raises KeyError."""
    link = Link.inner(JoinSpec(PlkirBaseA, "plkir_a_key"), JoinSpec(PlkirBaseB, "plkir_b_key"))

    results = mloda.run_all(
        [Feature(PlkirJoinedFG.get_class_name())],
        links={link},
        compute_frameworks=["PyArrowTable"],
        plugin_collector=PluginCollector.enabled_feature_groups({PlkirSubA, PlkirSubB, PlkirJoinedFG}),
        parallelization_modes={ParallelizationMode.SYNC},
    )

    values = sorted(results[0].to_pydict()[PlkirJoinedFG.get_class_name()])
    assert values == [220, 330, 440]
