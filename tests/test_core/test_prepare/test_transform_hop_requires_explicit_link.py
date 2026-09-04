"""A FeatureGroupStep whose parents live on two different, unlinked source-framework instances
must raise a "missing Links" ValueError at plan-build time, not silently bind only one hop's data.
"""

from typing import Any

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class UnlinkedRootA(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_root_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"unlinked_root_a": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class UnlinkedRootB(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"unlinked_root_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"unlinked_root_b": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class UnlinkedConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("unlinked_root_a"), Feature("unlinked_root_b")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "unlinked_root_a" in data.column_names:
            return data.append_column("unlinked_consumer_result", data["unlinked_root_a"])
        if "unlinked_root_b" in data.column_names:
            return data.append_column("unlinked_consumer_result", data["unlinked_root_b"])
        raise ValueError(f"neither present: {data.column_names}")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"unlinked_consumer_result"}


def test_two_unlinked_source_framework_instances_raise_missing_links_error_at_prepare_time() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            features=[Feature("unlinked_consumer_result")],
            links=set(),
            compute_frameworks={PandasDataFrame, PyArrowTable},
            plugin_collector=PluginCollector.enabled_feature_groups({UnlinkedRootA, UnlinkedRootB, UnlinkedConsumer}),
        )

    error_message = str(exc_info.value)

    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    assert "UnlinkedRootA" in error_message, "Error should name the first conflicting upstream feature group"
    assert "UnlinkedRootB" in error_message, "Error should name the second conflicting upstream feature group"
