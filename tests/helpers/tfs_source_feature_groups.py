"""Shared source feature group for the TFS connection end-to-end tests."""

from typing import Any

from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureSet
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class TfsRawValPyArrowSource(DataCreatorFeatureGroup):
    """Emits raw_val=[1,2,3] on PyArrowTable."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"raw_val"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"raw_val": [1, 2, 3]}
