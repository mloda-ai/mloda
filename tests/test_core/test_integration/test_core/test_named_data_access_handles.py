"""End-to-end showcase of the named ``DataAccessCollection`` handle patterns.

Each test exercises one pattern from
``docs/docs/in_depth/named-data-access-handles.md`` through the public
``mloda.run_all`` API (with one method covering pure introspection and one
covering the CFW-level engine-binding surface). Read top-to-bottom to learn
every usage shape PR #449 enables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pytest

from mloda.provider import (
    ComputeFramework,
    FeatureGroup,
    FeatureSet,
    MatchData,
)
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import (
    SqliteFramework,
    _regexp,
)
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature  # noqa: F401
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader  # noqa: F401
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


# --- CSV fixtures ----------------------------------------------------------


def _write_csv(path: Path, header: str, rows: list[str]) -> str:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


@pytest.fixture
def single_csv(tmp_path: Path) -> str:
    """One CSV with a 'single_col' column for the bare-set pattern."""
    return _write_csv(tmp_path / "single.csv", "id,single_col", ["1,10", "2,20"])


@pytest.fixture
def overlapping_csvs(tmp_path: Path) -> tuple[str, str]:
    """Two CSVs sharing column name 'shared_col' to force ambiguity."""
    a = _write_csv(tmp_path / "a.csv", "id,shared_col", ["1,10", "2,20"])
    b = _write_csv(tmp_path / "b.csv", "id,shared_col", ["3,30", "4,40"])
    return a, b


@pytest.fixture
def pinning_csvs(tmp_path: Path) -> tuple[str, str]:
    """Two CSVs sharing column ``id`` with disjoint values, so column_to_file pinning is observable."""
    train = _write_csv(tmp_path / "train.csv", "id,train_target", ["1,A", "2,B"])
    bureau = _write_csv(tmp_path / "bureau.csv", "id,bureau_amt", ["10,100", "20,200"])
    return train, bureau


# --- SQLite seeding + transform feature groups -----------------------------


_sqlite_seed_dict: dict[str, Any] = {
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
}


class _SqliteSeedCreator(ATestDataCreator):
    compute_framework = SqliteFramework
    conversion = {**ATestDataCreator.conversion, SqliteFramework: lambda data: data}

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return _sqlite_seed_dict


class _DoubledValueFG(FeatureGroup, MatchData):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if feature_name not in cls.feature_names_supported():
            return None
        if isinstance(framework_connection_object, sqlite3.Connection):
            return framework_connection_object
        if data_access_collection is None:
            return None
        for conn in data_access_collection.connections.values():
            if isinstance(conn, sqlite3.Connection):
                return conn
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"doubled_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.select(_raw_sql="*, value * 2 AS doubled_value")


def _make_sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp, deterministic=True)
    return conn


# --- Showcase --------------------------------------------------------------


class TestNamedDataAccessHandlesIntegration:
    def test_single_file_bare_set_form(self, single_csv: str) -> None:
        """Pattern: single file in bare-set form needs no handle."""
        dac = DataAccessCollection(files={single_csv})
        result = mloda.run_all(
            ["single_col"],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=dac,
        )
        assert "single_col" in result[0].to_pydict()

    def test_multiple_files_no_hint_raises_value_error(self, overlapping_csvs: tuple[str, str]) -> None:
        """Pattern: two matching files without a hint surface a ValueError listing the handle candidates."""
        path_a, path_b = overlapping_csvs
        dac = DataAccessCollection(files={"a": path_a, "b": path_b})
        with pytest.raises(ValueError) as excinfo:
            mloda.run_all(
                ["shared_col"],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=dac,
            )
        msg = str(excinfo.value)
        assert "data_access_handle" in msg
        assert "'a'" in msg and "'b'" in msg

    def test_per_feature_data_access_handle_disambiguates(self, overlapping_csvs: tuple[str, str]) -> None:
        """Pattern: feature-level Options data_access_handle picks one of two same-shape files."""
        path_a, path_b = overlapping_csvs
        dac = DataAccessCollection(files={"a": path_a, "b": path_b})
        feature = Feature("shared_col", options=Options(context={"data_access_handle": "a"}))
        result = mloda.run_all(
            [feature],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=dac,
        )
        assert result[0].to_pydict()["shared_col"] == [10, 20]

    def test_column_to_file_by_handle(self, pinning_csvs: tuple[str, str]) -> None:
        """Pattern: column_to_file values may reference a file handle, overriding ambiguity for the listed columns."""
        train, bureau = pinning_csvs
        dac = DataAccessCollection(
            files={"train": train, "bureau": bureau},
            column_to_file={"id": "train", "train_target": "train"},
        )
        result = mloda.run_all(
            ["id", "train_target"],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=dac,
        )
        out = result[0].to_pydict()
        assert out["id"] == [1, 2]
        assert out["train_target"] == ["A", "B"]

    def test_column_to_file_by_path(self, pinning_csvs: tuple[str, str]) -> None:
        """Pattern: column_to_file values may also be a file path; PR #449 normalizes path values to handles."""
        train, bureau = pinning_csvs
        dac = DataAccessCollection(
            files={"train": train, "bureau": bureau},
            column_to_file={"id": bureau, "bureau_amt": bureau},
        )
        result = mloda.run_all(
            ["id", "bureau_amt"],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=dac,
        )
        out = result[0].to_pydict()
        assert out["id"] == [10, 20]
        assert out["bureau_amt"] == [100, 200]

    def test_single_connection_bare_set_form_via_run_all(self) -> None:
        """Pattern: single connection in bare-set form auto-binds at CFW engine setup."""
        conn = _make_sqlite_conn()
        dac = DataAccessCollection(connections={conn})
        plugin_collector = PluginCollector.enabled_feature_groups({_SqliteSeedCreator, _DoubledValueFG})
        result = mloda.run_all(
            [Feature(name="doubled_value", options={"_SqliteSeedCreator": conn})],
            compute_frameworks={SqliteFramework},
            data_access_collection=dac,
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        doubled = result[0].df()["doubled_value"].tolist()
        assert doubled == [20, 40, 60, 80, 100]
        conn.close()

    def test_two_connections_engine_binding_picks_named_handle(self) -> None:
        """Pattern: connections resolve per CFW engine; pass Options to pick_connection_from_dac to bind a named handle."""
        primary = _make_sqlite_conn()
        secondary = _make_sqlite_conn()
        dac = DataAccessCollection(connections={"primary": primary, "secondary": secondary})
        options = Options(context={"data_access_handle": "secondary"})
        resolved = SqliteFramework.pick_connection_from_dac(dac, options=options)
        assert resolved is secondary
        with pytest.raises(ValueError, match="data_access_handle"):
            SqliteFramework.pick_connection_from_dac(dac)
        primary.close()
        secondary.close()

    def test_handles_introspection_lists_all_kinds(self, tmp_path: Path) -> None:
        """Pattern: DataAccessCollection.handles() returns {handle: kind} across all four kinds."""
        conn = _make_sqlite_conn()
        file_path = _write_csv(tmp_path / "f.csv", "x", ["1"])
        folder_path = str(tmp_path)
        dac = DataAccessCollection(
            connections={"warehouse": conn},
            files={"transactions": file_path},
            folders={"raw": folder_path},
            credentials={"pg-prod": {"host": "h", "user": "u"}},
        )
        assert dac.handles() == {
            "warehouse": "connection",
            "transactions": "file",
            "raw": "folder",
            "pg-prod": "credentials",
        }
        conn.close()
