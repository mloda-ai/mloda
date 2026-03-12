"""
Integration test: JoinSpec join columns must be injected even without index_columns().

When a Link is created with explicit JoinSpec objects specifying join columns,
and the referenced FeatureGroup does not define index_columns(), the engine
should still inject those join columns into the FeatureSet.
"""

from typing import Any, Optional, Set, Union

import pandas as pd

from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class JoinSpecNoIndexFGA(FeatureGroup):
    """Root FeatureGroup that produces a DataFrame with a join column and a value column.
    Does NOT define index_columns().
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name in ("join_key", "value_a")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"join_key": [1, 2, 3], "value_a": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set]:
        return {PandasDataFrame}


class JoinSpecNoIndexFGB(FeatureGroup):
    """Root FeatureGroup that produces a DataFrame with a join column and a value column.
    Defines index_columns() so its side of the join works via the standard path.
    """

    @classmethod
    def index_columns(cls) -> Optional[list]:
        return [Index(("join_key",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name in ("join_key", "value_b")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"join_key": [1, 2, 3], "value_b": [100, 200, 300]})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set]:
        return {PandasDataFrame}


class TestJoinSpecWithoutIndexColumns:
    def test_joinspec_injects_columns_without_index_columns(self) -> None:
        """JoinSpec should inject join columns even when the FeatureGroup has no index_columns()."""
        plugin_collector = PluginCollector.enabled_feature_groups({JoinSpecNoIndexFGA, JoinSpecNoIndexFGB})

        links = {
            Link.left(
                JoinSpec(JoinSpecNoIndexFGA, "join_key"),
                JoinSpec(JoinSpecNoIndexFGB, "join_key"),
            ),
        }

        result = mloda.run_all(
            [Feature("value_a"), Feature("value_b")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
            links=links,
        )

        all_columns: Set[str] = set()
        for df in result:
            all_columns.update(df.columns)

        assert "value_a" in all_columns
        assert "value_b" in all_columns
