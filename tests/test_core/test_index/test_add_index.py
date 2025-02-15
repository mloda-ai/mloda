from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Any, Set, Optional, Union, List
import unittest

from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_plugins.feature_group.input_data.read_db_feature import ReadDBFeature
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.api.request import mlodaAPI
from tests.test_plugins.feature_group.input_data.test_classes.test_input_classes import (
    DBInputDataTestFeatureGroup,
)


class TestAddIndex(unittest.TestCase):
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    feature_names = "id,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class"
    feature_list = feature_names.split(",")

    def setUp(self) -> None:
        # Create a temporary file to act as the SQLite database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        # Initialize the SQLite database with a sample table
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, any_num INTEGER)")
        self.cursor.execute('INSERT INTO test_table (name, any_num) VALUES ("Alice", 3)')
        self.cursor.execute('INSERT INTO test_table (name, any_num) VALUES ("Bob", 4)')
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        os.close(self.db_fd)
        os.remove(self.db_path)

    def test_add_index_simple(
        self,
    ) -> None:
        class ReadFileFeatureWithIndex(ReadFileFeature):
            @classmethod
            def index_columns(cls) -> Optional[List[Index]]:
                return [Index(("id",))]

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Union[FeatureName, str],
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                # Feature is only valid for this test
                if options.get("test_add_index_simple") is None:
                    return False

                if isinstance(feature_name, FeatureName):
                    feature_name = feature_name.name

                if cls().is_root(options, feature_name):
                    input_data_class = cls.input_data()
                    return input_data_class.matches(feature_name, options, data_access_collection)  # type: ignore
                return False

        class DBInputDataTestFeatureGroupWithIndex(DBInputDataTestFeatureGroup):
            @classmethod
            def index_columns(cls) -> Optional[List[Index]]:
                return [Index(("id",))]

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                reader = cls.input_data()
                if reader is not None:
                    result = reader.load(features)

                    # As of date of writing this test, we did not handle the types automatically.
                    # Thus, we need to convert the columns to int64...
                    for column_name in features.get_all_names():
                        index = result.schema.get_field_index(column_name)
                        int64_column = result[column_name].cast(pa.int64())
                        result = result.set_column(index, column_name, int64_column)

                    return result

                raise ValueError(f"Reading file failed for feature {features.get_name_of_one_feature()}.")

        class AddIndexTest(AbstractFeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Union[FeatureName, str],
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                if "TestAddIndexFeature" in feature_name.name:  # type: ignore
                    return True
                return False

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature.int32_of("Amount"), Feature.int32_of("any_num")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data.append_column(
                    "TestAddIndexFeature",
                    pc.add(data["Amount"], data["any_num"]),  # Perform addition using pyarrow.compute
                )

        link = Link(
            jointype="inner",
            left=(ReadFileFeatureWithIndex, Index(("id",))),
            right=(DBInputDataTestFeatureGroupWithIndex, Index(("id",))),
        )
        f = Feature(
            name="TestAddIndexFeature",
            options={
                CsvReader.__name__: self.file_path,
                SQLITEReader.__name__: HashableDict({SQLITEReader.db_path(): self.db_path, "table_name": "test_table"}),
                "test_add_index_simple": True,
            },
        )

        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
            links={link},
            plugin_collector=PlugInCollector.disabled_feature_groups({ReadDBFeature}),
        )
        res = result[0].to_pydict()
        assert res == {"TestAddIndexFeature": [6534.37, 2517.54]}
