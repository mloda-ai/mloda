"""Regression coverage probing whether HashableDict-wrapping of SQLITEReader credentials
in a DataAccessCollection still serves a purpose under ParallelizationMode.MULTIPROCESSING.

After PR #449 (named DataAccessCollection handles), DataAccessCollection.credentials accepts
plain dicts. SQLITEReader.is_valid_credentials and SQLITEReader.connect still isinstance-check
for HashableDict. These tests pin down whether that requirement is load-bearing for the
multiprocessing path through mloda.run_all (pickling subprocess args, hashable set membership
in subprocess, etc.), or whether it is solely enforced by the isinstance checks in SQLITEReader
itself and would be safe to loosen.

Observed today: the plain-dict failure surfaces during synchronous feature-group resolution,
before any multiprocessing worker is spawned -- which means the gate is the isinstance check
in SQLITEReader.is_valid_credentials, not anything inherent to the multiprocessing path.

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

    def test_hashable_dict_credentials_work_under_multiprocessing(self, tmp_path: Path, flight_server: Any) -> None:
        """Pins that the current HashableDict-wrapped pattern still produces rows under MULTIPROCESSING."""
        db_path = tmp_path / "mp_hashable.sqlite"
        self._seed_db(db_path)

        result = mloda.run_all(
            ["name", "id"],
            compute_frameworks=["PyArrowTable"],
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
            data_access_collection=DataAccessCollection(
                credentials=[HashableDict({SQLITEReader.db_path(): str(db_path)})]
            ),
            plugin_collector=PluginCollector.enabled_feature_groups({DBInputDataTestFeatureGroup}),
        )

        assert "name" in result[0].to_pydict()

    def test_plain_dict_credentials_under_multiprocessing(self, tmp_path: Path, flight_server: Any) -> None:
        """Pins that an unwrapped plain dict fails feature resolution because SQLITEReader.is_valid_credentials rejects it."""
        db_path = tmp_path / "mp_plain.sqlite"
        self._seed_db(db_path)

        with pytest.raises(ValueError, match="No feature groups found for feature name: 'name'"):
            mloda.run_all(
                ["name", "id"],
                compute_frameworks=["PyArrowTable"],
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                flight_server=flight_server,
                data_access_collection=DataAccessCollection(credentials=[{SQLITEReader.db_path(): str(db_path)}]),
                plugin_collector=PluginCollector.enabled_feature_groups({DBInputDataTestFeatureGroup}),
            )
