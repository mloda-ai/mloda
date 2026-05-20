"""Contract tests for the named-handle DataAccessCollection API.

See ``docs/docs/in_depth/named-data-access-handles.md`` for the user-facing guide.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection


class TestNewConstructorShape:
    """Constructor accepts keyed dicts for every registry field."""

    def test_accepts_keyed_connections(self) -> None:
        warehouse = object()
        analytics = object()
        dac = DataAccessCollection(connections={"warehouse": warehouse, "analytics": analytics})
        assert dac.connections == {"warehouse": warehouse, "analytics": analytics}

    def test_accepts_keyed_files(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/tx.parquet", "users": "/data/users.csv"})
        assert dac.files == {"tx": "/data/tx.parquet", "users": "/data/users.csv"}

    def test_accepts_keyed_folders(self) -> None:
        dac = DataAccessCollection(folders={"raw": "/data/raw", "stage": "/data/stage"})
        assert dac.folders == {"raw": "/data/raw", "stage": "/data/stage"}

    def test_accepts_keyed_credentials(self) -> None:
        dac = DataAccessCollection(
            credentials={"pg": {"host": "h"}, "snow": {"account": "a"}},
        )
        assert dac.credentials == {"pg": {"host": "h"}, "snow": {"account": "a"}}

    def test_empty_construction_yields_empty_dicts(self) -> None:
        dac = DataAccessCollection()
        assert dac.connections == {}
        assert dac.files == {}
        assert dac.folders == {}
        assert dac.credentials == {}

    def test_uninitialized_connection_objects_field_is_removed(self) -> None:
        """The dead ``uninitialized_connection_objects`` parameter is gone entirely."""
        with pytest.raises(TypeError):
            DataAccessCollection(uninitialized_connection_objects=[])  # type: ignore[call-arg]

    def test_column_to_file_value_must_reference_file_handle(self) -> None:
        """In the new API, column_to_file values are file handles, not paths."""
        with pytest.raises(ValueError):
            DataAccessCollection(
                files={"tx": "/data/tx.parquet"},
                column_to_file={"amount": "/data/tx.parquet"},
            )

    def test_column_to_file_with_valid_file_handle_succeeds(self) -> None:
        dac = DataAccessCollection(
            files={"tx": "/data/tx.parquet"},
            column_to_file={"amount": "tx"},
        )
        assert dac.column_to_file == {"amount": "tx"}


class TestMutatorsRaiseOnDuplicateHandle:
    """Each add_* mutator must refuse silent overwrite."""

    def test_add_connection_duplicate_handle_raises(self) -> None:
        dac = DataAccessCollection(connections={"warehouse": object()})
        with pytest.raises(ValueError, match="warehouse"):
            dac.add_connection("warehouse", object())

    def test_add_file_duplicate_handle_raises(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/a.csv"})
        with pytest.raises(ValueError, match="tx"):
            dac.add_file("tx", "/data/b.csv")

    def test_add_folder_duplicate_handle_raises(self) -> None:
        dac = DataAccessCollection(folders={"raw": "/data/raw"})
        with pytest.raises(ValueError, match="raw"):
            dac.add_folder("raw", "/data/other")

    def test_add_credentials_duplicate_handle_raises(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"host": "h"}})
        with pytest.raises(ValueError, match="pg"):
            dac.add_credentials("pg", {"host": "h2"})

    def test_add_connection_new_handle_registers(self) -> None:
        dac = DataAccessCollection(connections={"warehouse": object()})
        analytics = object()
        dac.add_connection("analytics", analytics)
        assert dac.connections["analytics"] is analytics


class TestHandles:
    """``handles()`` enumerates every registered handle and its kind."""

    def test_handles_maps_each_handle_to_its_kind(self) -> None:
        dac = DataAccessCollection(
            connections={"warehouse": object()},
            files={"tx": "/data/tx.parquet"},
            folders={"raw": "/data/raw"},
            credentials={"pg": {"host": "h"}},
        )
        assert dac.handles() == {
            "warehouse": "connection",
            "tx": "file",
            "raw": "folder",
            "pg": "credentials",
        }

    def test_handles_empty_for_empty_dac(self) -> None:
        assert DataAccessCollection().handles() == {}


class TestResolveContract:
    """Exhaustive coverage of the seven resolver contract cases.

    Each test method name carries the case number it covers (case_1 ... case_7) so the
    contract-to-test mapping is obvious.
    """

    # Case 1: explicit hint, found, correct kind, passes predicate (or no predicate).
    def test_case_1_hint_found_correct_kind_no_predicate(self) -> None:
        warehouse = object()
        analytics = object()
        dac = DataAccessCollection(connections={"warehouse": warehouse, "analytics": analytics})
        assert dac.resolve("connection", hint="analytics") is analytics

    def test_case_1_hint_found_correct_kind_passes_predicate(self) -> None:
        warehouse = object()
        analytics = object()
        dac = DataAccessCollection(connections={"warehouse": warehouse, "analytics": analytics})
        result = dac.resolve("connection", predicate=lambda c: c is analytics, hint="analytics")
        assert result is analytics

    # Case 2: explicit hint, handle missing -> ValueError naming kind + listing available handles.
    def test_case_2_hint_missing_raises_with_kind_and_available_handles(self) -> None:
        dac = DataAccessCollection(connections={"warehouse": object(), "analytics": object()})
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("connection", hint="nonexistent")
        msg = str(excinfo.value)
        assert "connection" in msg
        assert "nonexistent" in msg
        assert "warehouse" in msg
        assert "analytics" in msg

    # Case 3: explicit hint, handle exists but is the wrong kind -> ValueError.
    def test_case_3_hint_exists_but_wrong_kind_raises(self) -> None:
        dac = DataAccessCollection(
            connections={"warehouse": object()},
            files={"tx": "/data/tx.parquet"},
        )
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("connection", hint="tx")
        msg = str(excinfo.value)
        assert "tx" in msg

    # Case 4: explicit hint, handle correct kind, predicate fails -> ValueError with mismatch.
    def test_case_4_hint_found_correct_kind_predicate_fails_raises(self) -> None:
        analytics = object()
        dac = DataAccessCollection(connections={"analytics": analytics})
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("connection", predicate=lambda c: False, hint="analytics")
        msg = str(excinfo.value)
        assert "analytics" in msg

    # Case 5: no hint, zero matches -> None.
    def test_case_5_no_hint_zero_matches_returns_none(self) -> None:
        dac = DataAccessCollection()
        assert dac.resolve("connection") is None

    def test_case_5_no_hint_zero_matches_after_predicate_returns_none(self) -> None:
        dac = DataAccessCollection(connections={"warehouse": object()})
        assert dac.resolve("connection", predicate=lambda c: False) is None

    # Case 6: no hint, exactly one match -> returns the entry.
    def test_case_6_no_hint_one_match_returns_entry(self) -> None:
        warehouse = object()
        dac = DataAccessCollection(connections={"warehouse": warehouse})
        assert dac.resolve("connection") is warehouse

    def test_case_6_no_hint_one_match_after_predicate_returns_entry(self) -> None:
        warehouse = object()
        analytics = object()
        dac = DataAccessCollection(connections={"warehouse": warehouse, "analytics": analytics})
        assert dac.resolve("connection", predicate=lambda c: c is warehouse) is warehouse

    # Case 7: no hint, more than one match -> ValueError listing candidates.
    def test_case_7_no_hint_multi_match_raises_with_candidates(self) -> None:
        dac = DataAccessCollection(connections={"warehouse": object(), "analytics": object()})
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("connection")
        msg = str(excinfo.value)
        assert "warehouse" in msg
        assert "analytics" in msg
        # Same error shape as ComputeFramework.pick_connection_from_dac ships today.
        assert "data_access_handle" in msg

    def test_case_7_files_multi_match_raises_with_candidates(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/a.csv", "users": "/data/b.csv"})
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("file")
        msg = str(excinfo.value)
        assert "tx" in msg
        assert "users" in msg


class TestResolveAcrossKinds:
    """Verify that ``resolve`` does not cross kinds when filtering."""

    def test_resolve_file_kind_ignores_other_kinds(self) -> None:
        dac = DataAccessCollection(
            connections={"warehouse": object()},
            files={"tx": "/data/tx.parquet"},
        )
        assert dac.resolve("file") == "/data/tx.parquet"

    def test_resolve_folder_kind_returns_single_folder(self) -> None:
        dac = DataAccessCollection(folders={"raw": "/data/raw"})
        assert dac.resolve("folder") == "/data/raw"

    def test_resolve_credentials_kind_returns_single_credentials(self) -> None:
        creds: dict[str, Any] = {"host": "h"}
        dac = DataAccessCollection(credentials={"pg": creds})
        assert dac.resolve("credentials") == creds
