import os
from typing import Any, List, Optional, Union

import tempfile
import sqlite3
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
import pytest
import pyarrow as pa

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link, JoinSpec
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from tests.test_plugins.feature_group.input_data.test_classes.test_input_classes import (
    DBInputDataTestFeatureGroup,
)
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature


class TestTwoReader:
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    feature_names = "id"
    feature_list = feature_names.split(",")

    def setup_method(self) -> None:
        # Create a temporary file to act as the SQLite database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        # Initialize the SQLite database with a sample table
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, any_num INTEGER)")
        self.cursor.execute('INSERT INTO test_table (name, any_num) VALUES ("Alice", 3)')
        self.cursor.execute('INSERT INTO test_table (name, any_num) VALUES ("Bob", 4)')
        self.conn.commit()

    def teardown_method(self) -> None:
        self.conn.close()
        os.close(self.db_fd)
        os.remove(self.db_path)

    def test_load_local_feature_scope_data_double_reader_success(self) -> None:
        feature_list: List[Feature] = []

        for feature in self.feature_list:
            # add sqlite reader feature
            f = Feature(
                name=feature,
                options={
                    SQLITEReader.__name__: HashableDict(
                        {SQLITEReader.db_path(): self.db_path, "table_name": "test_table"}
                    )
                },
            )
            feature_list.append(f)
            # add csv reader feature
            f = Feature(name=feature, options={CsvReader.__name__: self.file_path})
            feature_list.append(f)

        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            compute_frameworks=["PyarrowTable"],
            plugin_collector=PlugInCollector.enabled_feature_groups({DBInputDataTestFeatureGroup, ReadFileFeature}),
        )
        assert result[0].to_pydict()["id"] != result[1].to_pydict()["id"]

    def test_load_multiple_local_data_for_one_feature_fail(self) -> None:
        feature_list: List[Feature] = []
        for feature in self.feature_list:
            f = Feature(
                name=feature,
                options={
                    SQLITEReader.__name__: HashableDict(
                        {SQLITEReader.db_path(): self.db_path, "table_name": "test_table"}
                    ),
                    CsvReader.__name__: self.file_path,
                },
            )
            feature_list.append(f)

        with pytest.raises(ValueError) as excinfo:
            mlodaAPI.run_all(
                feature_list,  # type: ignore
                compute_frameworks=["PyarrowTable"],
            )
        assert "BaseInputData already set with different values" in str(excinfo.value)

    def test_load_data_access_collection_feature_scope_data_double_reader_fail(self) -> None:
        feature_list: List[Feature] = []
        for feature in self.feature_list:
            f = Feature(
                name=feature,
                options={
                    SQLITEReader.__name__: HashableDict(
                        {SQLITEReader.db_path(): self.db_path, "table_name": "test_table"}
                    )
                },
            )
            feature_list.append(f)

        with pytest.raises(ValueError) as excinfo:
            mlodaAPI.run_all(
                feature_list,  # type: ignore
                compute_frameworks=["PyarrowTable"],
                data_access_collection=DataAccessCollection(files={self.file_path}),
            )
        assert "BaseInputData already set with different values" in str(excinfo.value)

    def test_agg_feature(self) -> None:
        index = Index(("id",))

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
                if options.get("test_agg_feature") is None:
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
                        if column_name == "any_num":
                            col = result[column_name].cast(pa.float64())
                        else:
                            col = result[column_name].cast(pa.int64())
                        result = result.set_column(index, column_name, col)

                    return result

                raise ValueError(f"Reading file failed for feature {features.get_name_of_one_feature()}.")

        link = Link(
            jointype="inner",
            left=JoinSpec(DBInputDataTestFeatureGroupWithIndex, index),
            right=JoinSpec(ReadFileFeatureWithIndex, index),
        )
        f = Feature(
            name="sum_of_",
            options={
                "sum": ("any_num", "Amount"),
                SQLITEReader.__name__: HashableDict({SQLITEReader.db_path(): self.db_path, "table_name": "test_table"}),
                CsvReader.__name__: self.file_path,
                "test_agg_feature": True,
            },
        )

        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
            links={link},
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {ReadFileFeatureWithIndex, DBInputDataTestFeatureGroupWithIndex, SumFeature}
            ),
        )
        assert result[0].to_pydict()["SumFeature_any_numAmount"] == [9051.91, 9051.91]

        with pytest.raises(ValueError):
            mlodaAPI.run_all(
                [f],
                compute_frameworks=["PyarrowTable"],
                links={link},
                plugin_collector=PlugInCollector.enabled_feature_groups(
                    {ReadFileFeature, DBInputDataTestFeatureGroupWithIndex}
                ),
            )
