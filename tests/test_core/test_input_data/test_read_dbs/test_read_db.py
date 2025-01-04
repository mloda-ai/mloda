import os
from typing import Any, Dict

import tempfile
import unittest
import sqlite3

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.input_data.read_db import ReadDB
from mloda_plugins.input_data.read_dbs.sqlite import SQLITEReader
from tests.test_core.test_input_data.test_classes.test_input_classes import DBInputDataTestFeatureGroup
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature


class TestInputDataDB(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary file to act as the SQLite database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        # Initialize the SQLite database with a sample table
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        self.cursor.execute('INSERT INTO test_table (name) VALUES ("Alice")')
        self.cursor.execute('INSERT INTO test_table (name) VALUES ("Bob")')

        self.cursor.execute("CREATE TABLE test_table_2 (id INTEGER PRIMARY KEY, name TEXT)")

        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        os.close(self.db_fd)
        os.remove(self.db_path)

    def test_load_csv_local_feature_scope_data_access_with_a_concrete_file(self) -> Any:
        f = Feature(
            name="id",
            options={
                SQLITEReader.__name__: HashableDict({SQLITEReader.db_path(): self.db_path, "table_name": "test_table"})
            },
        )

        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
            plugin_collector=PlugInCollector.enabled_feature_groups({DBInputDataTestFeatureGroup}),
        )
        assert "id" in result[0].to_pydict()

    def test_load_sqlite_found_in_data_access_collection(self) -> Any:
        result = mlodaAPI.run_all(
            ["name", "id"],
            compute_frameworks=["PyarrowTable"],
            data_access_collection=DataAccessCollection(credential_dicts={SQLITEReader.db_path(): self.db_path}),
            plugin_collector=PlugInCollector.enabled_feature_groups({DBInputDataTestFeatureGroup}),
        )
        assert "name" in result[0].to_pydict()

    def test_aggr_load_sqlite_found_in_data_access_collection(self) -> Any:
        f = Feature(
            name="sum_of_",
            options={"sum": ("id", "id")},
        )

        result = mlodaAPI.run_all(
            [f],
            compute_frameworks=["PyarrowTable"],
            data_access_collection=DataAccessCollection(credential_dicts={SQLITEReader.db_path(): self.db_path}),
            plugin_collector=PlugInCollector.enabled_feature_groups({DBInputDataTestFeatureGroup, SumFeature}),
        )
        assert "SumFeature_idid" in result[0].to_pydict()


class TestReadDB(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary file to act as the SQLite database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        # Initialize the SQLite database with a sample table
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        self.cursor.execute('INSERT INTO test_table (name) VALUES ("Alice")')
        self.cursor.execute('INSERT INTO test_table (name) VALUES ("Bob")')
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        os.close(self.db_fd)
        os.remove(self.db_path)

    def test_load_data_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReadDB.load_data(None, None)  # type: ignore

    def test_connect_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReadDB.connect(None)

    def test_build_query_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReadDB.build_query(None)  # type: ignore

    def test_is_valid_credentials_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReadDB.is_valid_credentials({})

    def test_check_feature_in_data_access_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReadDB.check_feature_in_data_access("", None)

    def test_init_reader_no_options(self) -> None:
        read_db = ReadDB()
        with self.assertRaises(ValueError):
            read_db.init_reader(None)

    def test_init_reader_no_data_access(self) -> None:
        read_db = ReadDB()
        options = Options()
        with self.assertRaises(ValueError):
            read_db.init_reader(options)

    def test_load_no_data(self) -> None:
        read_db = ReadDB()
        features = FeatureSet()
        features.options = Options()
        with self.assertRaises(ValueError):
            read_db.load(features)

    def test_match_subclass_data_access(self) -> None:
        data_access = DataAccessCollection(credential_dicts={SQLITEReader.db_path(): self.db_path})
        feature_names = ["name"]
        result = ReadDB.match_subclass_data_access(data_access, feature_names)
        self.assertFalse(result)

    def test_get_connection_no_credentials(self) -> None:
        with self.assertRaises(NotImplementedError):
            ReadDB.get_connection(None)

    def test_read_db_success(self) -> None:
        class MockReadDB(ReadDB):
            @classmethod
            def connect(cls, credentials: Any) -> Any:
                return sqlite3.connect(credentials[SQLITEReader.db_path()])

            @classmethod
            def is_valid_credentials(cls, credentials: Dict[str, Any]) -> bool:
                return SQLITEReader.db_path() in credentials

            @classmethod
            def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
                return True

        credentials = {SQLITEReader.db_path(): self.db_path}
        query = "SELECT * FROM test_table"
        result, column_names = MockReadDB.read_db(credentials, query)
        self.assertEqual(len(result), 2)
        self.assertEqual(column_names, ["id", "name"])
