"""Regression coverage for SQLITEReader credentials in a DataAccessCollection under
ParallelizationMode.MULTIPROCESSING.

A plain dict credential must produce rows end-to-end through ``mloda.run_all`` under
MULTIPROCESSING. HashableDict is no longer accepted on the credentials path (issue
#519): passing it as a credential value is rejected at construction time with a
migration error pointing to ``Credential`` / plain dict.

Reference: https://github.com/mloda-ai/mloda/pull/449
"""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mloda.provider import HashableDict
from mloda.user import DataAccessCollection
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import PluginLoader
from mloda.user import mloda
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from tests.test_plugins.feature_group.input_data.test_classes.test_input_classes import DBInputDataTestFeatureGroup


class TestSqliteCredentialsUnderMultiprocessing:
    PluginLoader().all()

    @staticmethod
    def _seed_db(db_path: Path) -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute('INSERT INTO test_table (name) VALUES ("Alice")')
        conn.execute('INSERT INTO test_table (name) VALUES ("Bob")')
        conn.commit()
        conn.close()

    def test_hashable_dict_credentials_rejected(self, tmp_path: Path) -> None:
        """A HashableDict credential value is rejected at construction with a migration error."""
        db_path = tmp_path / "mp_hashable.sqlite"

        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials=[HashableDict({SQLITEReader.db_path(): str(db_path)})])
        assert "credential" in str(excinfo.value).lower()

    def test_plain_dict_credentials_under_multiprocessing(self, tmp_path: Path, flight_server: Any) -> None:
        """An unwrapped plain dict is now accepted and produces rows end-to-end under MULTIPROCESSING."""
        db_path = tmp_path / "mp_plain.sqlite"
        self._seed_db(db_path)

        result = mloda.run_all(
            ["name", "id"],
            compute_frameworks=["PyArrowTable"],
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
            data_access_collection=DataAccessCollection(credentials=[{SQLITEReader.db_path(): str(db_path)}]),
            plugin_collector=PluginCollector.enabled_feature_groups({DBInputDataTestFeatureGroup}),
        )

        assert "name" in result[0].to_pydict()
