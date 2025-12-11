import sqlite3
from typing import Any, List
import pytest
from unittest.mock import MagicMock, patch

import pyarrow as pa

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader


class MockFeatureSet:
    def __init__(self, feature_names: List[str], options: Any = None) -> None:
        self._feature_names = feature_names
        self.options = options

    def get_all_names(self) -> List[str]:
        return self._feature_names


class MockOptions:
    def __init__(self, base_input_data: Any) -> None:
        self.base_input_data = base_input_data

    def get(self, key: Any, default: Any = None) -> Any:
        if key == "BaseInputData":
            return self.base_input_data
        return default


class TestSQLITEReader:
    @pytest.fixture(scope="class")
    def temp_sqlite_db(self, tmp_path_factory: Any) -> Any:
        db_path = tmp_path_factory.mktemp("data") / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER
            );
        """)
        cursor.executemany(
            "INSERT INTO test_table (name, age) VALUES (?, ?);",
            [
                ("Alice", 30),
                ("Bob", 25),
                ("Charlie", 35),
            ],
        )
        conn.commit()
        conn.close()
        return str(db_path)

    @pytest.fixture(scope="class")
    def valid_credentials(self, temp_sqlite_db: Any) -> Any:
        return HashableDict({"sqlite": temp_sqlite_db})

    @pytest.fixture(scope="class")
    def invalid_credentials(self) -> Any:
        return HashableDict({"sqlite": "non_existent.db"})

    @pytest.fixture
    def mock_read_db(self) -> Any:
        with patch.object(SQLITEReader, "read_db") as mock_method:
            yield mock_method

    @pytest.fixture
    def mock_read_as_pa_data(self) -> Any:
        with patch.object(SQLITEReader, "read_as_pa_data") as mock_method:
            yield mock_method

    def test_db_path(self) -> None:
        assert SQLITEReader.db_path() == "sqlite"

    def test_connect_valid(self, valid_credentials: Any) -> None:
        connection = SQLITEReader.connect(valid_credentials)
        assert isinstance(connection, sqlite3.Connection)
        connection.close()

    def test_connect_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Credentials must be an HashableDict."):
            SQLITEReader.connect({"sqlite": "path/to/db"})

    def test_is_valid_credentials_valid(self, valid_credentials: Any) -> None:
        assert SQLITEReader.is_valid_credentials(valid_credentials)

    def test_is_valid_credentials_nonexistent(self, tmp_path: Any) -> None:
        credentials = HashableDict({"sqlite": str(tmp_path / "nonexistent.db")})
        with pytest.raises(
            ValueError, match=f"Database file {credentials.data['sqlite']} does not exist, but key is given."
        ):
            SQLITEReader.is_valid_credentials(credentials)

    def test_load_data(self, valid_credentials: Any, mock_read_db: Any, mock_read_as_pa_data: Any) -> None:
        feature_set = FeatureSet()
        features = {Feature("id"), Feature("name"), Feature("age")}
        for feature in features:
            feature.options = MagicMock()
            feature_set.add(feature)
        # Mock the read_db to return dummy data
        mock_read_db.return_value = ([(1, "Alice", 30), (2, "Bob", 25)], ["id", "name", "age"])
        # Mock the read_as_pa_data to return a pyarrow table
        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"], "age": [30, 25]})
        mock_read_as_pa_data.return_value = table

        result = SQLITEReader.load_data(valid_credentials, feature_set)

        # Assert that read_db was called with the correct query
        mock_read_db.assert_called_once()
        # Assert that read_as_pa_data was called with the correct parameters
        mock_read_as_pa_data.assert_called_once_with(
            [(1, "Alice", 30), (2, "Bob", 25)], ["id", "name", "age"], feature_set
        )
        # Assert the result is the mocked table
        assert result == table

    def test_get_table_missing_options(self) -> None:
        with pytest.raises(ValueError, match="Options were not set."):
            SQLITEReader.get_table(None)

    def test_get_table_missing_table_name(self) -> None:
        options = MockOptions(("BaseInputData", HashableDict({})))
        with pytest.raises(KeyError, match="'table_name'"):
            SQLITEReader.get_table(options)  # type: ignore
