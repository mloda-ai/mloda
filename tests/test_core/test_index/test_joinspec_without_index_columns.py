from __future__ import annotations

import os
from typing import Any, Set, Optional, Union, List

from mloda.provider import FeatureGroup, FeatureSet, HashableDict
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    Index,
    JoinSpec,
    Link,
    Options,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


class TestJoinSpecWithoutIndexColumns:
    """Test that JoinSpec join columns are injected even when a FeatureGroup does not define index_columns()."""

    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    def test_joinspec_injects_join_column_when_no_index_columns(self) -> None:
        """When FeatureGroupA has no index_columns(), the JoinSpec 'id' column must still be loaded for the merge."""

        class FeatureGroupA(ReadFileFeature):
            """Reads 'Amount' from CSV. Does NOT override index_columns(), so it returns None."""

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                if options.get("test_joinspec_no_index") is None:
                    return False

                if isinstance(feature_name, FeatureName):
                    feature_name = feature_name.name

                if feature_name != "Amount":
                    return False

                if cls().is_root(options, feature_name):
                    input_data_class = cls.input_data()
                    return input_data_class.matches(feature_name, options, data_access_collection)  # type: ignore
                return False

        class FeatureGroupB(ReadFileFeature):
            """Reads 'Class' from CSV. Defines index_columns() returning [Index(('id',))]."""

            @classmethod
            def index_columns(cls) -> Optional[list[Index]]:
                return [Index(("id",))]

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                if options.get("test_joinspec_no_index") is None:
                    return False

                if isinstance(feature_name, FeatureName):
                    feature_name = feature_name.name

                if feature_name != "Class":
                    return False

                if cls().is_root(options, feature_name):
                    input_data_class = cls.input_data()
                    return input_data_class.matches(feature_name, options, data_access_collection)  # type: ignore
                return False

        class FeatureGroupC(FeatureGroup):
            """Derived feature group that combines Amount from A and Class from B."""

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                if isinstance(feature_name, FeatureName):
                    feature_name = feature_name.name
                return feature_name == "JoinSpecNoIndexResult"

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
                return {Feature("Amount"), Feature("Class")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                import pyarrow.compute as pc

                return data.append_column(
                    "JoinSpecNoIndexResult",
                    pc.add(data["Amount"], data["Class"]),
                )

        link = Link.left(
            JoinSpec(FeatureGroupA, "id"),
            JoinSpec(FeatureGroupB, "id"),
        )

        f = Feature(
            name="JoinSpecNoIndexResult",
            options={
                CsvReader.__name__: self.file_path,
                "test_joinspec_no_index": True,
            },
        )

        result = mloda.run_all(
            [f],
            compute_frameworks=["PyArrowTable"],
            links={link},
            plugin_collector=PluginCollector.disabled_feature_groups(set()),
        )

        res = result[0].to_pydict()
        assert "JoinSpecNoIndexResult" in res
        assert len(res["JoinSpecNoIndexResult"]) > 0
