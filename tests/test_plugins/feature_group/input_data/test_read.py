import gc
import os
from typing import Any, cast

import tempfile
import sqlite3
from mloda.user import FeatureName
from mloda.user import Options
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
import pytest
import pyarrow as pa

from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.user import Link, JoinSpec
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.feature_group.input_data.db_formats.sqlite_fg import SqliteFG
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
        feature_list: list[Feature] = []

        for feature in self.feature_list:
            # add sqlite reader feature
            f = Feature(
                name=feature,
                options={SqliteFG.__name__: {"sqlite": self.db_path, "table_name": "test_table"}},
            )
            feature_list.append(f)
            # add csv reader feature
            f = Feature(name=feature, options={CsvFG.get_class_name(): self.file_path})
            feature_list.append(f)

        result = mloda.run_all(
            feature_list,  # type: ignore
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG, CsvFG}),
        )
        assert result[0].to_pydict()["id"] != result[1].to_pydict()["id"]

    def test_load_multiple_local_data_for_one_feature_fail(self) -> None:
        def gated(cls: Any, feature_name: Any, options: Options, data_access_collection: Any = None) -> bool:
            if options.get("test_two_siblings") is None:
                return False
            return bool(
                getattr(super(cls, cls), "match_feature_group_criteria")(feature_name, options, data_access_collection)
            )

        sibling_a = type("SqliteFGSiblingA", (SqliteFG,), {"match_feature_group_criteria": classmethod(gated)})
        sibling_b = type("SqliteFGSiblingB", (SqliteFG,), {"match_feature_group_criteria": classmethod(gated)})
        feature_list = [
            Feature(
                name=feature,
                options={
                    SqliteFG.__name__: {"sqlite": self.db_path, "table_name": "test_table"},
                    "test_two_siblings": True,
                },
            )
            for feature in self.feature_list
        ]

        with pytest.raises(ValueError) as excinfo:
            mloda.run_all(
                feature_list,  # type: ignore
                compute_frameworks=["PyArrowTable"],
                plugin_collector=PluginCollector.enabled_feature_groups(cast(Any, {sibling_a, sibling_b})),
            )
        assert "Multiple feature groups found" in str(excinfo.value)
        assert "BaseInputData already set" not in str(excinfo.value)
        del sibling_a, sibling_b, excinfo
        gc.collect()

    def test_load_data_access_collection_feature_scope_data_double_reader_fail(self) -> None:
        feature_list: list[Feature] = []
        for feature in self.feature_list:
            f = Feature(
                name=feature,
                options={SqliteFG.__name__: {"sqlite": self.db_path, "table_name": "test_table"}},
            )
            feature_list.append(f)

        with pytest.raises(ValueError) as excinfo:
            mloda.run_all(
                feature_list,  # type: ignore
                compute_frameworks=["PyArrowTable"],
                data_access_collection=DataAccessCollection(files={self.file_path}),
                plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG, CsvFG}),
            )
        assert "Multiple feature groups found" in str(excinfo.value)
        assert "BaseInputData already set" not in str(excinfo.value)

    def _agg_groups(self) -> tuple[type[CsvFG], type[SqliteFG], Link]:
        index = Index(("id",))

        class CsvFGWithIndex(CsvFG):
            @classmethod
            def index_columns(cls) -> list[Index] | None:
                return [Index(("id",))]

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: DataAccessCollection | None = None,
            ) -> bool:
                # Feature is only valid for this test
                if options.get("test_agg_feature") is None:
                    return False

                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        class SqliteFGWithIndex(SqliteFG):
            @classmethod
            def index_columns(cls) -> list[Index] | None:
                return [Index(("id",))]

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: DataAccessCollection | None = None,
            ) -> bool:
                # Feature is only valid for this test
                if options.get("test_agg_feature") is None:
                    return False

                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                result = super().calculate_feature(data, features)

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

        link = Link(
            jointype="inner",
            left=JoinSpec(SqliteFGWithIndex, index),
            right=JoinSpec(CsvFGWithIndex, index),
        )
        return CsvFGWithIndex, SqliteFGWithIndex, link

    def _dac(self) -> DataAccessCollection:
        return DataAccessCollection(files={self.file_path}, credentials=[{"sqlite": self.db_path}])

    def test_agg_feature(self) -> None:
        CsvFGWithIndex, SqliteFGWithIndex, link = self._agg_groups()
        # The credential lives in the collection: a consumer pointer would be forwarded to the csv input too.
        f = Feature(
            name="sum_of_",
            options={
                "sum": ("any_num", "Amount"),
                "test_agg_feature": True,
            },
        )

        result = mloda.run_all(
            [f],
            compute_frameworks=["PyArrowTable"],
            links={link},
            data_access_collection=self._dac(),
            plugin_collector=PluginCollector.enabled_feature_groups({CsvFGWithIndex, SqliteFGWithIndex, SumFeature}),
        )
        assert result[0].to_pydict()["SumFeature_any_numAmount"] == [9051.91, 9051.91]

        with pytest.raises(ValueError):
            mloda.run_all(
                [f],
                compute_frameworks=["PyArrowTable"],
                links={link},
                data_access_collection=self._dac(),
                plugin_collector=PluginCollector.enabled_feature_groups({CsvFG, SqliteFGWithIndex}),
            )

    def test_pointer_on_the_consumer_aborts_naming_the_input_the_csv_lacks(self) -> None:
        csv_group, db_group, link = self._agg_groups()
        f = Feature(
            name="sum_of_",
            options={
                "sum": ("any_num", "Amount"),
                SqliteFG.__name__: {"sqlite": self.db_path, "table_name": "test_table"},
                "test_agg_feature": True,
                csv_group.get_class_name(): self.file_path,
            },
        )

        with pytest.raises(ValueError, match=r"column '(any_num|Amount)' is in none of the sources"):
            mloda.run_all(
                [f],
                compute_frameworks=["PyArrowTable"],
                links={link},
                plugin_collector=PluginCollector.enabled_feature_groups({csv_group, db_group, SumFeature}),
            )
