"""Contract tests for the named-handle DataAccessCollection API.

See ``docs/docs/in_depth/named-data-access-handles.md`` for the user-facing guide.
"""

import time
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.lazy_format_group import load_group


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

    def test_column_to_file_value_accepts_a_path_in_files_values(self) -> None:
        """Under the loosened API, ``column_to_file`` accepts either a handle or a path.

        Passing a path that appears in ``files.values()`` must succeed and be normalized
        to the matching file handle internally so a file format group can index
        ``files`` by the stored value.
        """
        dac = DataAccessCollection(
            files={"tx": "/data/tx.parquet"},
            column_to_file={"amount": "/data/tx.parquet"},
        )
        assert dac.column_to_file is not None
        stored = dac.column_to_file["amount"]
        # Stored value must be a valid file handle pointing to the same path.
        assert stored in dac.files
        assert dac.files[stored] == "/data/tx.parquet"

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
        # Error shape matches ComputeFramework.pick_connection_from_dac.
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


class TestConstructorAcceptsSetAndListShapes:
    """Bare set/list inputs no longer need user-invented handles."""

    def test_files_set_form_no_handles_needed(self) -> None:
        dac = DataAccessCollection(files={"/data/tx.csv"})
        assert isinstance(dac.files, dict)
        assert len(dac.files) == 1
        (only_handle,) = dac.files.keys()
        assert only_handle.startswith("_auto_file_")
        assert dac.files[only_handle] == "/data/tx.csv"

    def test_files_list_form_preserves_each_value(self) -> None:
        dac = DataAccessCollection(files=["/data/tx.csv", "/data/users.csv"])
        assert isinstance(dac.files, dict)
        assert len(dac.files) == 2
        for handle in dac.files:
            assert handle.startswith("_auto_file_")
        assert set(dac.files.values()) == {"/data/tx.csv", "/data/users.csv"}

    def test_folders_set_form(self) -> None:
        dac = DataAccessCollection(folders={"/data/raw"})
        (only_handle,) = dac.folders.keys()
        assert only_handle.startswith("_auto_folder_")
        assert dac.folders[only_handle] == "/data/raw"

    def test_connections_set_form_of_hashable_objects(self) -> None:
        obj1 = object()
        obj2 = object()
        dac = DataAccessCollection(connections={obj1, obj2})
        assert len(dac.connections) == 2
        for handle in dac.connections:
            assert handle.startswith("_auto_connection_")
        assert set(dac.connections.values()) == {obj1, obj2}

    def test_credentials_list_form_of_dicts(self) -> None:
        dac = DataAccessCollection(credentials=[{"host": "h1"}, {"host": "h2"}])
        assert len(dac.credentials) == 2
        for handle in dac.credentials:
            assert handle.startswith("_auto_credentials_")
        assert list(dac.credentials.values()) == [{"host": "h1"}, {"host": "h2"}]

    def test_dict_form_still_works(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/tx.csv"})
        assert dac.files == {"tx": "/data/tx.csv"}

    def test_mixed_kinds_one_set_one_dict(self) -> None:
        obj = object()
        dac = DataAccessCollection(files={"/data/tx.csv"}, connections={"warehouse": obj})
        assert dac.connections == {"warehouse": obj}
        assert len(dac.files) == 1
        (only_handle,) = dac.files.keys()
        assert only_handle.startswith("_auto_file_")
        assert dac.files[only_handle] == "/data/tx.csv"


class TestAutoHandleAssignment:
    """Auto-handles are surfaced through ``handles()`` and never clobber user handles."""

    def test_handles_includes_auto_handles_with_kind(self) -> None:
        dac = DataAccessCollection(files={"/data/tx.csv"})
        registered = dac.handles()
        assert len(registered) == 1
        (handle, kind) = next(iter(registered.items()))
        assert handle.startswith("_auto_file_")
        assert kind == "file"

    def test_auto_handles_paired_with_their_kinds(self) -> None:
        dac = DataAccessCollection(
            files=["/data/tx.csv"],
            folders={"/data/raw"},
            credentials=[{"host": "h"}],
        )
        registered = dac.handles()
        kinds = sorted(registered.values())
        assert kinds == ["credentials", "file", "folder"]
        for handle, kind in registered.items():
            assert handle.startswith(f"_auto_{kind}_")

    def test_auto_handles_skip_user_supplied_collisions(self) -> None:
        """User-supplied handles always win; auto-numbering must dodge them."""
        obj = object()
        dac = DataAccessCollection(
            connections={"_auto_file_0": obj},
            files={"/data/tx.csv"},
        )
        # The user's hand-named handle is preserved as a connection.
        assert dac.connections == {"_auto_file_0": obj}
        # The file got an auto handle that does NOT collide with the user's name.
        (file_handle,) = dac.files.keys()
        assert file_handle.startswith("_auto_file_")
        assert file_handle != "_auto_file_0"


class TestColumnToFileAcceptsPathOrHandle:
    """``column_to_file`` may reference a file by handle or by path; mixed is OK."""

    def test_path_value_resolves_through_a_file_format_group(self, tmp_path: Path) -> None:
        """End-to-end: a path value is normalized so a file format group finds the pinned file."""
        path = tmp_path / "tx.csv"
        path.write_text("amount\n1\n", encoding="utf-8")
        dac = DataAccessCollection(files={str(path)}, column_to_file={"amount": str(path)})
        feature = Feature("amount")

        resolve_or_raise(feature, {load_group("csv_fg", "CsvFG"): {PyArrowTable}}, None, dac)

        assert feature.input_data_match is not None
        assert feature.input_data_match[1].source == str(path)

    def test_mixed_handle_and_path_both_resolve(self) -> None:
        dac = DataAccessCollection(
            files={"tx": "/data/tx.csv", "users": "/data/users.csv"},
            column_to_file={"amount": "tx", "email": "/data/users.csv"},
        )
        # Both column values must end up as valid keys into dac.files.
        assert dac.column_to_file is not None
        assert dac.column_to_file["amount"] in dac.files
        assert dac.column_to_file["email"] in dac.files
        assert dac.files[dac.column_to_file["amount"]] == "/data/tx.csv"
        assert dac.files[dac.column_to_file["email"]] == "/data/users.csv"

    def test_unknown_value_lists_handles_and_paths(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(
                files={"/data/tx.csv"},
                column_to_file={"col": "/missing.csv"},
            )
        msg = str(excinfo.value)
        assert "/missing.csv" in msg
        assert "col" in msg
        # New contract: error must surface both available handles and available paths
        # so the user can correct the typo without spelunking through .files.
        assert "/data/tx.csv" in msg


class TestAmbiguityMessageShape:
    """When the resolver raises on ambiguity, the message guidance depends on whether
    the colliding handles are auto-generated or user-named."""

    def test_files_all_auto_message_lists_paths_and_guides_naming(self) -> None:
        dac = DataAccessCollection(files=["/data/tx.csv", "/data/users.csv"])
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("file")
        msg = str(excinfo.value)
        # When every candidate is anonymous, listing _auto_file_0 / _1 helps nobody.
        # The message must surface the underlying paths and tell the user to name them.
        assert "/data/tx.csv" in msg
        assert "/data/users.csv" in msg
        assert "data_access_handle" in msg
        assert "Name them" in msg

    def test_files_named_message_lists_handles(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/tx.csv", "users": "/data/users.csv"})
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("file")
        msg = str(excinfo.value)
        assert "tx" in msg
        assert "users" in msg
        assert "data_access_handle" in msg

    def test_files_mixed_named_and_auto_message_lists_named_handles(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/tx.csv"})
        # Add an unnamed sibling so two distinct files exist.
        dac.add_file("_auto_file_99", "/data/users.csv")
        # User-named handle present -> stick to the handle-listing shape.
        # We at least require the named handle to appear in the message.
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("file")
        msg = str(excinfo.value)
        assert "tx" in msg
        assert "data_access_handle" in msg

    def test_connections_all_auto_message_lists_values_and_guides_naming(self) -> None:
        obj1 = object()
        obj2 = object()
        dac = DataAccessCollection(connections={obj1, obj2})
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("connection")
        msg = str(excinfo.value)
        # Auto-only: guide the user to name them.
        assert "data_access_handle" in msg
        assert "Name them" in msg


class TestMutatorsAcceptOmittedHandle:
    """Mutators may be called with the handle omitted; arity decides the shape."""

    # --- Requirement 1: single-arg form auto-names and registers --------------

    def test_add_file_single_arg_auto_names(self) -> None:
        dac = DataAccessCollection()
        dac.add_file("/data/tx.csv")
        assert len(dac.files) == 1
        (only_handle,) = dac.files.keys()
        assert only_handle.startswith("_auto_file_")
        assert dac.files[only_handle] == "/data/tx.csv"
        assert dac.handles()[only_handle] == "file"
        assert only_handle in dac._auto_handles

    def test_add_folder_single_arg_auto_names(self) -> None:
        dac = DataAccessCollection()
        dac.add_folder("/data/raw")
        assert len(dac.folders) == 1
        (only_handle,) = dac.folders.keys()
        assert only_handle.startswith("_auto_folder_")
        assert dac.folders[only_handle] == "/data/raw"
        assert dac.handles()[only_handle] == "folder"
        assert only_handle in dac._auto_handles

    def test_add_connection_single_arg_auto_names(self) -> None:
        dac = DataAccessCollection()
        conn = object()
        dac.add_connection(conn)
        assert len(dac.connections) == 1
        (only_handle,) = dac.connections.keys()
        assert only_handle.startswith("_auto_connection_")
        assert dac.connections[only_handle] is conn
        assert dac.handles()[only_handle] == "connection"
        assert only_handle in dac._auto_handles

    def test_add_credentials_single_arg_auto_names(self) -> None:
        dac = DataAccessCollection()
        creds: dict[str, Any] = {"host": "h"}
        dac.add_credentials(creds)
        assert len(dac.credentials) == 1
        (only_handle,) = dac.credentials.keys()
        assert only_handle.startswith("_auto_credentials_")
        assert dac.credentials[only_handle] == creds
        assert dac.handles()[only_handle] == "credentials"
        assert only_handle in dac._auto_handles

    # --- Requirement 2: two-arg form preserves the named handle ---------------

    def test_add_file_two_arg_preserves_named_handle(self) -> None:
        dac = DataAccessCollection()
        dac.add_file("tx", "/data/tx.csv")
        assert dac.files == {"tx": "/data/tx.csv"}
        assert "tx" not in dac._auto_handles

    def test_add_folder_two_arg_preserves_named_handle(self) -> None:
        dac = DataAccessCollection()
        dac.add_folder("raw", "/data/raw")
        assert dac.folders == {"raw": "/data/raw"}
        assert "raw" not in dac._auto_handles

    def test_add_connection_two_arg_preserves_named_handle(self) -> None:
        dac = DataAccessCollection()
        conn = object()
        dac.add_connection("warehouse", conn)
        assert dac.connections == {"warehouse": conn}
        assert "warehouse" not in dac._auto_handles

    def test_add_credentials_two_arg_preserves_named_handle(self) -> None:
        dac = DataAccessCollection()
        dac.add_credentials("pg", {"host": "h"})
        assert dac.credentials == {"pg": {"host": "h"}}
        assert "pg" not in dac._auto_handles

    # --- Requirement 3: mix of named-from-constructor and unnamed-from-mutator -

    def test_named_from_constructor_then_unnamed_mutator_both_registered(self) -> None:
        dac = DataAccessCollection(files={"tx": "/data/tx.csv"})
        dac.add_file("/data/users.csv")
        assert "tx" in dac.files
        assert dac.files["tx"] == "/data/tx.csv"
        assert len(dac.files) == 2
        # The auto-added one resolves via predicate.
        assert dac.resolve("file", predicate=lambda p: p.endswith("users.csv")) == "/data/users.csv"

    # --- Requirement 4: ambiguity message in pure-auto state ------------------

    def test_resolver_ambiguity_after_two_auto_mutator_adds_uses_auto_shape(self) -> None:
        dac = DataAccessCollection()
        dac.add_file("/a.csv")
        dac.add_file("/b.csv")
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("file")
        msg = str(excinfo.value)
        assert "/a.csv" in msg
        assert "/b.csv" in msg
        assert "data_access_handle" in msg
        assert "Name them" in msg

    # --- Requirement 5: auto numbering survives constructor + mutator ---------

    def test_auto_numbering_does_not_collide_across_constructor_and_mutator(self) -> None:
        dac = DataAccessCollection(files=["/a.csv"])
        dac.add_file("/b.csv")
        assert len(dac.files) == 2
        for handle in dac.files:
            assert handle.startswith("_auto_file_")
        handles = list(dac.files.keys())
        assert handles[0] != handles[1]
        assert set(dac.files.values()) == {"/a.csv", "/b.csv"}

    # --- Requirement 6: wrong arity raises a clear error ----------------------

    def test_add_file_zero_args_raises(self) -> None:
        dac = DataAccessCollection()
        with pytest.raises((TypeError, ValueError)):
            dac.add_file()

    def test_add_file_three_args_raises(self) -> None:
        dac = DataAccessCollection()
        with pytest.raises((TypeError, ValueError)):
            dac.add_file("h", "/p", "extra")

    # --- Requirement 7: duplicate detection still works on two-arg form -------

    def test_two_arg_duplicate_handle_still_raises(self) -> None:
        dac = DataAccessCollection(files={"tx": "/a.csv"})
        with pytest.raises(ValueError, match="tx"):
            dac.add_file("tx", "/b.csv")


class TestRedact:
    """``dac.redact(text)`` masks registered credential values (8+ chars) after scrub_credentials."""

    _REGISTERED_VALUE = "hunter2plus"  # nosec B105
    _SHORT_VALUE = "abc123"  # nosec B105
    _REGISTERED_HOST = "db.internal.example"

    def test_registered_password_in_prose_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": self._REGISTERED_VALUE}})
        redacted = dac.redact(f"connection failed, saw {self._REGISTERED_VALUE} in the logs")
        assert self._REGISTERED_VALUE not in redacted
        assert "***" in redacted

    def test_value_added_later_via_add_credentials_is_masked(self) -> None:
        dac = DataAccessCollection()
        dac.add_credentials("pg", {"password": self._REGISTERED_VALUE})
        redacted = dac.redact(f"echoed {self._REGISTERED_VALUE} back")
        assert self._REGISTERED_VALUE not in redacted

    def test_nested_named_credential_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"auth": {"password": self._REGISTERED_VALUE}}})
        redacted = dac.redact(f"nested {self._REGISTERED_VALUE} value")
        assert self._REGISTERED_VALUE not in redacted

    def test_list_form_entry_with_nested_dict_is_masked(self) -> None:
        dac = DataAccessCollection(credentials=[{"pg": {"password": self._REGISTERED_VALUE}}])
        redacted = dac.redact(f"nested list {self._REGISTERED_VALUE} value")
        assert self._REGISTERED_VALUE not in redacted

    def test_short_value_not_literal_masked_but_password_form_is_masked_by_pattern(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": self._SHORT_VALUE}})
        prose = dac.redact(f"the code is {self._SHORT_VALUE} today")
        assert self._SHORT_VALUE in prose
        keyword_form = dac.redact(f"password={self._SHORT_VALUE}")
        assert self._SHORT_VALUE not in keyword_form

    def test_non_str_values_are_ignored_without_error(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"port": 5432}})
        redacted = dac.redact("connecting on port 5432")
        assert redacted == "connecting on port 5432"

    def test_overlapping_values_fully_masked_no_leftover_tail(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "a": {"password": self._REGISTERED_VALUE},
                "b": {"password": f"{self._REGISTERED_VALUE}extra"},
            }
        )
        redacted = dac.redact(f"saw {self._REGISTERED_VALUE}extra in the logs")
        assert self._REGISTERED_VALUE not in redacted
        assert "extra" not in redacted

    def test_registered_host_masked_alongside_scrubbed_token(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"host": self._REGISTERED_HOST}})
        redacted = dac.redact(f"GET https://{self._REGISTERED_HOST}/api?token=hunter2plus")
        assert self._REGISTERED_HOST not in redacted
        assert "hunter2plus" not in redacted

    def test_url_without_registered_values_still_scrubbed_by_delegation(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": self._REGISTERED_VALUE}})
        redacted = dac.redact("postgres://user:pw@host:5432/db")
        assert "user:pw" not in redacted

    def test_text_without_secrets_unchanged(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": self._REGISTERED_VALUE}})
        text = "plain error text with no credentials at all"
        assert dac.redact(text) == text

    def test_credentials_unchanged_after_redact(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": self._REGISTERED_VALUE}})
        dac.redact(f"echo {self._REGISTERED_VALUE}")
        assert dac.credentials == {"pg": {"password": self._REGISTERED_VALUE}}

    def test_two_collections_mask_only_their_own_values(self) -> None:
        dac1 = DataAccessCollection(credentials={"pg": {"password": self._REGISTERED_VALUE}})
        dac2 = DataAccessCollection(credentials={"pg": {"password": "swordfishlong"}})  # nosec B105 B106
        text = f"{self._REGISTERED_VALUE} and swordfishlong side by side"
        redacted1 = dac1.redact(text)
        assert self._REGISTERED_VALUE not in redacted1
        assert "swordfishlong" in redacted1
        redacted2 = dac2.redact(text)
        assert "swordfishlong" not in redacted2
        assert self._REGISTERED_VALUE in redacted2

    def test_large_text_with_secrets_redacts_well_under_one_second(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "a": {"password": self._REGISTERED_VALUE},
                "b": {"password": "swordfishlong"},  # nosec B105 B106
            }
        )
        text = ("filler text " * 20_000) + f"{self._REGISTERED_VALUE} swordfishlong"
        start = time.perf_counter()
        redacted = dac.redact(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"redact took {elapsed:.3f}s on 200k chars"
        assert self._REGISTERED_VALUE not in redacted
        assert "swordfishlong" not in redacted

    def test_password_with_semicolon_is_masked(self) -> None:
        value = "abc;defghijk"  # nosec B105
        dac = DataAccessCollection(credentials={"pg": {"password": value}})
        keyword_form = dac.redact(f"password={value}")
        prose = dac.redact(f"leaked {value} here")
        assert "defghijk" not in keyword_form
        assert "defghijk" not in prose

    def test_password_with_space_is_masked(self) -> None:
        value = "abc defghijk"  # nosec B105
        dac = DataAccessCollection(credentials={"pg": {"password": value}})
        keyword_form = dac.redact(f"password={value}")
        prose = dac.redact(f"leaked {value} here")
        assert "defghijk" not in keyword_form
        assert "defghijk" not in prose

    def test_password_with_ampersand_is_masked(self) -> None:
        value = "abc&defghijk"  # nosec B105
        dac = DataAccessCollection(credentials={"pg": {"password": value}})
        keyword_form = dac.redact(f"password={value}")
        prose = dac.redact(f"leaked {value} here")
        assert "defghijk" not in keyword_form
        assert "defghijk" not in prose

    def test_mid_string_overlapping_values_leave_no_fragment(self) -> None:
        first = "abcdefgh12"  # nosec B105
        second = "gh12345678"  # nosec B105
        dac = DataAccessCollection(credentials={"a": {"password": first}, "b": {"password": second}})
        redacted = dac.redact("abcdefgh12345678")
        assert "345678" not in redacted
        assert "abcdef" not in redacted

    def test_regex_metacharacter_value_masked_exactly_without_overmatching(self) -> None:
        value = "a.b*c+d?e(f)"  # nosec B105
        dac = DataAccessCollection(credentials={"pg": {"password": value}})
        redacted = dac.redact(f"leaked {value} here")
        assert value not in redacted
        unrelated = dac.redact("aXb stays untouched")
        assert unrelated == "aXb stays untouched"

    def test_nested_credential_object_inside_named_dict_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"inner": Credential(password=self._REGISTERED_VALUE)}})
        redacted = dac.redact(f"leaked {self._REGISTERED_VALUE} here")
        assert self._REGISTERED_VALUE not in redacted

    def test_value_inside_set_leaf_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"tags": {self._REGISTERED_VALUE}}})
        redacted = dac.redact(f"leaked {self._REGISTERED_VALUE} here")
        assert self._REGISTERED_VALUE not in redacted

    def test_value_inside_frozenset_leaf_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"tags": frozenset({self._REGISTERED_VALUE})}})
        redacted = dac.redact(f"leaked {self._REGISTERED_VALUE} here")
        assert self._REGISTERED_VALUE not in redacted

    def test_cyclic_container_does_not_raise_and_still_masks_other_values(self) -> None:
        cyclic: list[Any] = []
        cyclic.append(cyclic)
        dac = DataAccessCollection(credentials={"pg": {"password": self._REGISTERED_VALUE, "loop": cyclic}})
        redacted = dac.redact(f"leaked {self._REGISTERED_VALUE} here")
        assert self._REGISTERED_VALUE not in redacted

    def test_whitespace_only_value_does_not_mask_ordinary_whitespace(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": " " * 8}})
        text = "a line" + " " * 8 + "with padding"
        assert dac.redact(text) == text
