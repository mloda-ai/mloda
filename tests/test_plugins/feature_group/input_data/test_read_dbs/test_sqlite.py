import re
import sqlite3
from typing import Any
import pytest
from unittest.mock import MagicMock, patch

import pyarrow as pa

from mloda.user import DataType
from mloda.user import Feature
from mloda.user import Options
from mloda.provider import FeatureSet
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_affinity import sqlite_affinity_class
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import _sqlite_affinity_to_arrow_type
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader


class MockFeatureSet:
    def __init__(self, feature_names: list[str], options: Any = None) -> None:
        self._feature_names = feature_names
        self.options = options

    def get_all_names(self) -> list[str]:
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
    def affinity_test_table(self, temp_sqlite_db: Any) -> Any:
        """A second table covering describe_columns affinity edge cases: numeric/blob/text keywords, an
        undeclared column, and multi-keyword types that must resolve by affinity precedence, not check order."""
        conn = sqlite3.connect(temp_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE affinity_table (
                price REAL,
                weight FLOAT,
                distance DOUBLE,
                payload BLOB,
                label VARCHAR(255),
                notes CLOB,
                untyped_col,
                amount NUMERIC,
                precise DECIMAL(10,5),
                text_then_blob TEXT BLOB,
                blob_sub_type_text BLOB SUB_TYPE TEXT,
                char_then_double CHAR DOUBLE,
                double_then_blob DOUBLE BLOB
            );
        """)
        conn.commit()
        conn.close()
        return "affinity_table"

    @pytest.fixture(scope="class")
    def valid_credentials(self, temp_sqlite_db: Any) -> Any:
        # Credentials flow through SQLITEReader.connect / is_valid_credentials as plain dicts.
        return {"sqlite": temp_sqlite_db}

    @pytest.fixture(scope="class")
    def invalid_credentials(self) -> Any:
        return {"sqlite": "non_existent.db"}

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
        with pytest.raises(ValueError, match="must be a dict"):
            SQLITEReader.connect("not-a-dict")

    def test_is_valid_credentials_valid(self, valid_credentials: Any) -> None:
        assert SQLITEReader.is_valid_credentials(valid_credentials)

    def test_is_valid_credentials_nonexistent(self, tmp_path: Any) -> None:
        credentials = {"sqlite": str(tmp_path / "nonexistent.db")}
        with pytest.raises(
            ValueError,
            match=f"Database file {re.escape(credentials['sqlite'])} does not exist, but key is given.",
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

    def test_build_query_columns_sorted(self) -> None:
        """SELECT column order must be deterministic (alphabetically sorted), not
        dependent on FeatureSet.features set iteration order / PYTHONHASHSEED (#613)."""
        feature_set = FeatureSet()
        # Deliberately non-alphabetical insertion order; six names so at least one
        # PYTHONHASHSEED reorders the underlying set against the sorted expectation.
        names = ["c_col", "a_col", "f_col", "b_col", "e_col", "d_col"]
        for name in names:
            feature_set.add(
                Feature(
                    name,
                    options=Options(context={"BaseInputData": (SQLITEReader, {"table_name": "test_table"})}),
                )
            )

        query = SQLITEReader.build_query(feature_set)

        columns_part = query[len("select ") :].split(" from")[0]
        # Identifiers are double-quoted (quote_ident); strip the quotes to compare names.
        columns = [col.strip().strip('"') for col in columns_part.split(",")]

        assert columns == sorted(names), f"build_query emitted columns {columns}, expected {sorted(names)}"

    def test_build_query_quotes_identifiers_blocks_injection(self, temp_sqlite_db: Any) -> None:
        """A crafted feature name must not break out of the SELECT identifier position (CWE-89).

        Before quoting, a name like ``name FROM test_table UNION SELECT ...`` interpolated
        raw into ``select {name} from test_table`` and executed as attacker SQL. With
        quote_ident the whole string collapses into one double-quoted identifier, so the
        UNION never runs and the ``secrets`` table is never read.

        Note what SQLite actually does with that identifier: it does *not* reject it.
        SQLite resolves an unmatched double-quoted token to a string literal (the legacy
        double-quoted-string misfeature, disabled only by compiling with SQLITE_DQS=0), so
        the query succeeds and returns the crafted text itself as the column value on every
        row. That is harmless here (the payload is data, not SQL), and the assertion below
        pins the property that matters: the secret never comes back.
        """
        # Seed a secret table the injection would try to exfiltrate.
        conn = sqlite3.connect(temp_sqlite_db)
        conn.execute("CREATE TABLE IF NOT EXISTS secrets (api_key TEXT)")
        conn.execute("DELETE FROM secrets")
        conn.execute("INSERT INTO secrets VALUES ('SUPER_SECRET_KEY')")
        conn.commit()
        conn.close()

        malicious = "name FROM test_table UNION SELECT api_key FROM secrets --"
        feature_set = FeatureSet()
        feature_set.add(
            Feature(
                malicious,
                options=Options(context={"BaseInputData": (SQLITEReader, {"table_name": "test_table"})}),
            )
        )

        query = SQLITEReader.build_query(feature_set)

        # The whole crafted string is confined to one quoted identifier: no bare UNION/-- escapes.
        assert '"name FROM test_table UNION SELECT api_key FROM secrets --"' in query
        assert "UNION SELECT api_key" not in query.replace(
            '"name FROM test_table UNION SELECT api_key FROM secrets --"', ""
        )

        # Executing it must NOT exfiltrate the secret. The crafted name stays inside the
        # quoted identifier, so no UNION runs and 'SUPER_SECRET_KEY' never appears.
        result, _ = SQLITEReader.read_db({"sqlite": temp_sqlite_db}, query)
        leaked = {cell for row in result for cell in row}
        assert "SUPER_SECRET_KEY" not in leaked

    def test_get_table_missing_options(self) -> None:
        with pytest.raises(ValueError, match="Options were not set."):
            SQLITEReader.get_table(None)

    def test_get_table_missing_table_name(self) -> None:
        options = MockOptions(("BaseInputData", {}))
        with pytest.raises(KeyError, match="'table_name'"):
            SQLITEReader.get_table(options)  # type: ignore

    def test_describe_columns_happy_path(self, temp_sqlite_db: Any) -> None:
        result = SQLITEReader.describe_columns({"sqlite": temp_sqlite_db, "table_name": "test_table"})
        assert result == {"id": DataType.INT64, "name": DataType.STRING, "age": DataType.INT64}

    def test_describe_columns_missing_table_name(self, temp_sqlite_db: Any) -> None:
        with pytest.raises(ValueError, match="table_name"):
            SQLITEReader.describe_columns({"sqlite": temp_sqlite_db})

    def test_describe_columns_unknown_table(self, temp_sqlite_db: Any) -> None:
        with pytest.raises(ValueError, match="does_not_exist"):
            SQLITEReader.describe_columns({"sqlite": temp_sqlite_db, "table_name": "does_not_exist"})

    def test_describe_columns_affinity_edge_cases(self, temp_sqlite_db: Any, affinity_test_table: Any) -> None:
        result = SQLITEReader.describe_columns({"sqlite": temp_sqlite_db, "table_name": affinity_test_table})

        assert result["price"] == DataType.DOUBLE
        assert result["weight"] == DataType.DOUBLE
        assert result["distance"] == DataType.DOUBLE
        assert result["payload"] == DataType.BINARY
        # VARCHAR(255): substring match must still hit with a length modifier attached.
        assert result["label"] == DataType.STRING
        assert result["notes"] == DataType.STRING  # CLOB, not just plain TEXT
        # No declared type at all (PRAGMA table_info reports an empty string) must map to None, not raise.
        assert result["untyped_col"] is None
        # Diverges from _sqlite_affinity_to_arrow_type, which defaults an unmatched type to TEXT; here it's None.
        assert result["amount"] is None
        assert result["precise"] is None
        # A declared type with several keywords must resolve by SQLite's affinity precedence
        # (INTEGER, TEXT, BLOB, REAL/NUMERIC), not by whichever keyword is checked first.
        assert result["text_then_blob"] == DataType.STRING
        assert result["blob_sub_type_text"] == DataType.STRING
        assert result["char_then_double"] == DataType.STRING
        assert result["double_then_blob"] == DataType.BINARY

        # The same declared-type strings used above (per the affinity_table DDL) must resolve to the
        # same labels via the shared sqlite_affinity_class classifier, not just via describe_columns.
        assert sqlite_affinity_class("REAL") == "REAL"
        assert sqlite_affinity_class("FLOAT") == "REAL"
        assert sqlite_affinity_class("DOUBLE") == "REAL"
        assert sqlite_affinity_class("BLOB") == "BLOB"
        assert sqlite_affinity_class("VARCHAR(255)") == "TEXT"
        assert sqlite_affinity_class("CLOB") == "TEXT"
        assert sqlite_affinity_class("") == "NUMERIC"
        assert sqlite_affinity_class("NUMERIC") == "NUMERIC"
        assert sqlite_affinity_class("DECIMAL(10,5)") == "NUMERIC"
        assert sqlite_affinity_class("TEXT BLOB") == "TEXT"
        assert sqlite_affinity_class("BLOB SUB_TYPE TEXT") == "TEXT"
        assert sqlite_affinity_class("CHAR DOUBLE") == "TEXT"
        assert sqlite_affinity_class("DOUBLE BLOB") == "BLOB"

    def test_affinity_class_matches_relation_and_reader_call_sites(self) -> None:
        """Both call sites must agree with sqlite_affinity_class's label for the same declared type.

        Compared via labels rather than arrow-type equality, since TEXT and NUMERIC both map to pa.string().
        """
        label_to_arrow_type = {
            "INTEGER": pa.int64(),
            "TEXT": pa.string(),
            "BLOB": pa.large_binary(),
            "REAL": pa.float64(),
            "NUMERIC": pa.string(),
        }
        label_to_datatype: dict[str, DataType | None] = {
            "INTEGER": DataType.INT64,
            "TEXT": DataType.STRING,
            "BLOB": DataType.BINARY,
            "REAL": DataType.DOUBLE,
            "NUMERIC": None,
        }
        declared_types = [
            "INTEGER",
            "INT CHAR",
            "TEXT BLOB",
            "BLOB SUB_TYPE TEXT",
            "CHAR DOUBLE",
            "BLOB",
            "DOUBLE BLOB",
            "REAL",
            "FLOAT",
            "DOUBLE",
            "VARCHAR(255)",
            "CLOB",
            "NUMERIC",
            "DECIMAL(10,5)",
            "",
        ]

        for declared_type in declared_types:
            label = sqlite_affinity_class(declared_type)
            assert _sqlite_affinity_to_arrow_type(declared_type) == label_to_arrow_type[label], (
                f"{declared_type!r}: relation call site disagrees with sqlite_affinity_class label {label!r}"
            )
            assert SQLITEReader._affinity_to_datatype(declared_type) == label_to_datatype[label], (
                f"{declared_type!r}: reader call site disagrees with sqlite_affinity_class label {label!r}"
            )

    def test_describe_columns_nonexistent_db_does_not_create_file_and_error_mentions_path(self, tmp_path: Any) -> None:
        """sqlite3.connect would otherwise create the file; is_valid_credentials must fail fast, naming the path."""
        nonexistent_path = tmp_path / "nonexistent.db"
        assert not nonexistent_path.exists()

        with pytest.raises(ValueError, match=re.escape(str(nonexistent_path))):
            SQLITEReader.describe_columns({"sqlite": str(nonexistent_path), "table_name": "t"})

        assert not nonexistent_path.exists()

    def test_describe_columns_path_credential_rejected(self, tmp_path: Any) -> None:
        """A Path credential must fail is_valid_credentials's str check, not reach sqlite3.connect and create a file."""
        db_path = tmp_path / "missing.db"

        with pytest.raises(ValueError):
            SQLITEReader.describe_columns({"sqlite": db_path, "table_name": "t"})

        assert not db_path.exists()

    def test_describe_columns_missing_sqlite_key(self) -> None:
        """A data_access dict without the 'sqlite' key must raise ValueError, not KeyError."""
        with pytest.raises(ValueError):
            SQLITEReader.describe_columns({"table_name": "t"})

    def test_describe_columns_quotes_identifiers_blocks_injection(self, temp_sqlite_db: Any) -> None:
        """A crafted table_name must not break out of PRAGMA table_info(...)'s identifier position via quote_ident."""
        malicious_table_name = "test_table); DROP TABLE test_table; --"

        with pytest.raises(ValueError):
            SQLITEReader.describe_columns({"sqlite": temp_sqlite_db, "table_name": malicious_table_name})

        # test_table must still exist and be intact: the injected DROP never executed.
        result = SQLITEReader.describe_columns({"sqlite": temp_sqlite_db, "table_name": "test_table"})
        assert result == {"id": DataType.INT64, "name": DataType.STRING, "age": DataType.INT64}
