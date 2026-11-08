"""Contract tests for the named-handle DataAccessCollection API.

See ``docs/docs/in_depth/named-data-access-handles.md`` for the user-facing guide.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.read_db_fg import ReadDBFG
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.helpers.cpu_scaling import assert_scales_linearly
from tests.mixins.reader_feature_groups.lazy_source_group import load_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import TOY_TOKEN


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
        to the matching file handle internally so a file source group can index
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


class _Conn:
    """Stand-in connection type for the resolver tests."""


class _OtherConn:
    """A connection of a type the stub framework does not accept."""


class _StubFramework:
    @classmethod
    def _connection_matches(cls, conn: Any) -> bool:
        return isinstance(conn, _Conn)


class TestConnectionHandleResolver:
    """The single resolver behind ``connection_handle``: returns (handle, explicit, reason)."""

    def _resolve(self, dac: DataAccessCollection, **group: Any) -> tuple[str | None, bool, str | None]:
        from mloda.core.abstract_plugins.components.data_access_collection import resolve_connection_handle
        from mloda.user import Options

        return resolve_connection_handle(_StubFramework, dac, Options(group=group))

    def test_option_name_constant(self) -> None:
        from mloda.core.abstract_plugins.components.data_access_collection import CONNECTION_HANDLE_OPTION

        assert CONNECTION_HANDLE_OPTION == "connection_handle"

    def test_connection_handle_is_explicit_and_picks_the_named_one(self) -> None:
        dac = DataAccessCollection(connections={"a": _Conn(), "b": _Conn()})
        assert self._resolve(dac, connection_handle="b") == ("b", True, None)

    def test_data_access_handle_naming_a_connection_is_explicit(self) -> None:
        dac = DataAccessCollection(connections={"a": _Conn(), "b": _Conn()})
        assert self._resolve(dac, data_access_handle="a") == ("a", True, None)

    def test_connection_handle_wins_over_data_access_handle(self) -> None:
        dac = DataAccessCollection(connections={"a": _Conn(), "b": _Conn()})
        assert self._resolve(dac, connection_handle="b", data_access_handle="a")[0] == "b"

    def test_unknown_connection_handle_raises_naming_it(self) -> None:
        dac = DataAccessCollection(connections={"a": _Conn()})
        with pytest.raises(ValueError, match="nope"):
            self._resolve(dac, connection_handle="nope")

    def test_wrong_type_connection_handle_gives_no_handle_and_a_reason(self) -> None:
        dac = DataAccessCollection(connections={"a": _Conn(), "b": _Conn(), "other": _OtherConn()})
        handle, _explicit, reason = self._resolve(dac, connection_handle="other")
        assert handle is None
        assert reason is not None
        assert "other" in reason

    def test_wrong_type_reason_renders_an_auto_handle_as_unnamed_and_a_named_one_by_name(self) -> None:
        dac = DataAccessCollection()
        dac.add_connection(_OtherConn())
        (auto_handle,) = dac.connections
        _handle, _explicit, reason = self._resolve(dac, connection_handle=auto_handle)
        assert reason is not None
        assert "<unnamed>" in reason
        assert "_auto_connection_" not in reason

        named = DataAccessCollection(connections={"other": _OtherConn()})
        _handle, _explicit, named_reason = self._resolve(named, connection_handle="other")
        assert named_reason is not None
        assert "other" in named_reason
        assert "<unnamed>" not in named_reason

    def test_handle_naming_another_frameworks_connection_falls_through_to_the_single_match(self) -> None:
        dac = DataAccessCollection(connections={"s": _OtherConn(), "d": _Conn()})
        assert self._resolve(dac, connection_handle="s") == ("d", False, None)

    def test_connection_handle_without_a_data_access_collection_raises_naming_it(self) -> None:
        from mloda.core.abstract_plugins.components.data_access_collection import resolve_connection_handle
        from mloda.user import Options

        with pytest.raises(ValueError, match="orphan"):
            resolve_connection_handle(_StubFramework, None, Options(group={"connection_handle": "orphan"}))

    def test_ambiguous_reason_hides_auto_handles_counts_them_and_keeps_given_names(self) -> None:
        dac = DataAccessCollection()
        dac.add_connection("given", _Conn())
        dac.add_connection(_Conn())
        dac.add_connection(_Conn())
        _handle, _explicit, reason = self._resolve(dac)
        assert reason is not None
        assert "_auto_connection" not in reason
        assert "given" in reason
        assert "2" in reason
        assert "connection_handle" in reason

    def test_single_unnamed_match_is_implicit(self) -> None:
        conn = _Conn()
        dac = DataAccessCollection(connections={conn})
        handle, explicit, reason = self._resolve(dac)
        assert handle is not None
        assert dac.connections[handle] is conn
        assert explicit is False
        assert reason is None

    def test_two_unnamed_matches_return_a_reason_naming_both_handles_and_the_option(self) -> None:
        dac = DataAccessCollection(connections={"a": _Conn(), "b": _Conn()})
        handle, explicit, reason = self._resolve(dac)
        assert (handle, explicit) == (None, False)
        assert reason is not None
        for expected in ("a", "b", "connection_handle"):
            assert expected in reason

    def test_no_connection_returns_nothing(self) -> None:
        assert self._resolve(DataAccessCollection())[0] is None


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

    def test_path_value_resolves_through_a_file_source_group(self, tmp_path: Path) -> None:
        """End-to-end: a path value is normalized so a file source group finds the pinned file."""
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

    def test_unknown_value_message_hides_uri_password(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(
                files={"s3://alice:hunter2@bucket/x.csv"},  # nosec B105
                column_to_file={"col": "/missing.csv"},
            )
        msg = str(excinfo.value)
        assert "hunter2" not in msg
        assert "s3://bucket/x.csv" in msg


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

    def test_percent_encoded_registered_value_in_a_uri_path_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": "hunter2!xyz"}})  # nosec B105
        redacted = dac.redact("https://host/t/hunter2%21xyz")
        assert "hunter2" not in redacted

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

    def test_large_text_with_secrets_redacts_linearly(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "a": {"password": self._REGISTERED_VALUE},
                "b": {"password": "swordfishlong"},  # nosec B105 B106
            }
        )
        redacted = assert_scales_linearly(
            dac.redact,
            lambda n: ("filler text " * n) + f"{self._REGISTERED_VALUE} swordfishlong",
            20_000,
            "dac.redact with registered secrets",
        )
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


class TestCopy:
    """copy() returns an independent collection that shares the connection objects."""

    def test_mutating_the_copy_leaves_the_original_untouched(self) -> None:
        import sqlite3

        kept = sqlite3.connect(":memory:")
        extra = sqlite3.connect(":memory:")
        original = DataAccessCollection(connections={"kept": kept}, files=["/a.csv"], folders={"raw": "/raw"})
        original.add_connection(sqlite3.connect(":memory:"))
        before_connections = dict(original.connections)
        before_files = dict(original.files)
        before_auto = set(original._auto_handles)

        duplicate = original.copy()
        assert duplicate.connections == before_connections
        assert duplicate.connections["kept"] is kept
        assert duplicate._auto_handles == before_auto

        duplicate.add_connection(extra)
        duplicate.add_connection("named", sqlite3.connect(":memory:"))
        duplicate.add_file("/b.csv")
        duplicate.add_folder("more", "/more")
        duplicate.credentials["c"] = {"k": "v"}

        assert original.connections == before_connections
        assert original.files == before_files
        assert original.folders == {"raw": "/raw"}
        assert original.credentials == {}
        assert original._auto_handles == before_auto
        assert duplicate._auto_handles is not original._auto_handles
        assert len(duplicate._auto_handles) == len(before_auto) + 2


class TestSecretLiterals:
    """secret_literals() yields every 8+ char str leaf except under known-safe connection keys."""

    def test_every_leaf_except_safe_keys_is_registered(self, tmp_path: Path) -> None:
        dac = DataAccessCollection(
            credentials={
                "db": {
                    "sqlite": str(tmp_path / "long_database_name.db"),
                    "host": "db.internal.example",
                    "user": "administrator",
                },
                "api": {
                    "token": TOY_TOKEN,
                    "password": "correct-horse-battery",  # nosec B105
                    "key": "key-value-abcdef",
                    "hmac": "hmac-value-abcdef",
                    "toyfmt_custom_field": "custom-value-abcdef",
                    "unknown_thing": "unknown-value-abcdef",
                },
            }
        )

        assert dac.secret_literals() == frozenset(
            {
                TOY_TOKEN,
                "correct-horse-battery",
                "key-value-abcdef",
                "hmac-value-abcdef",
                "custom-value-abcdef",
                "unknown-value-abcdef",
            }
        )
        assert dac.identity_literals() == frozenset({TOY_TOKEN, "correct-horse-battery"})

    @pytest.mark.parametrize(
        "key",
        ["server", "account", "role", "authenticator", "db_name", "DB-Name", "dbname", "HOST", ReadDBFG.TABLE_KEY],
    )
    def test_safe_connection_keys_stay_out(self, key: str) -> None:
        dac = DataAccessCollection(credentials={"db": {key: "safe-value-12345", "password": "pw-value-123456"}})  # nosec B105

        assert dac.secret_literals() == frozenset({"pw-value-123456"})

    @pytest.mark.parametrize(
        ("key", "expected_identity"),
        [
            ("db_password", frozenset({"masked-value-12345"})),
            ("db-password", frozenset({"masked-value-12345"})),
            ("client_secret", frozenset({"masked-value-12345"})),
            ("dsn", frozenset()),
            ("url", frozenset()),
            ("conn", frozenset()),
            ("uri", frozenset()),
            ("connection_string", frozenset()),
        ],
    )
    def test_masked_keys_are_registered(self, key: str, expected_identity: frozenset[str]) -> None:
        dac = DataAccessCollection(credentials={"db": {key: "masked-value-12345", "client_id": "client-id-12345"}})

        assert dac.secret_literals() == frozenset({"masked-value-12345"})
        assert dac.identity_literals() == expected_identity

    def test_nested_mapping_under_a_safe_key_masks_its_shaped_child_only(self) -> None:
        dac = DataAccessCollection(
            credentials={"db": {"host": {"password": "nested-pw-123456", "dbname": "nested-db-12345"}}}
        )

        assert dac.secret_literals() == frozenset({"nested-pw-123456"})

    def test_shaped_ancestor_above_a_safe_child_is_masked(self) -> None:
        dac = DataAccessCollection(credentials={"db": {"password": {"host": "shaped-host-12345"}}})  # nosec B105

        assert dac.secret_literals() == frozenset({"shaped-host-12345"})

    def test_container_shared_under_a_safe_and_a_non_safe_key_is_masked(self) -> None:
        shared = ["shared-value-12345"]
        dac = DataAccessCollection(credentials={"db": {"host": shared, "other": shared}})

        assert dac.secret_literals() == frozenset({"shared-value-12345"})

    def test_cyclic_credential_does_not_hang(self) -> None:
        cyclic: dict[str, Any] = {"token": "cyclic-token-12345"}
        cyclic["self"] = cyclic

        dac = DataAccessCollection(credentials={"db": cyclic})

        assert dac.secret_literals() == frozenset({"cyclic-token-12345"})

    def test_credential_nested_in_a_list_that_contains_itself_does_not_hang(self) -> None:
        loop: list[Any] = []
        loop.append(loop)
        inner = Credential(items=loop, token="inner-token-12345")  # nosec B106
        loop.append(inner)

        dac = DataAccessCollection(credentials={"db": {"wrapped": loop}})

        assert "inner-token-12345" in dac.secret_literals()

    def test_nested_and_credential_wrapped_secrets_are_found(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "a": {"cfg": {"api_key": "key-value-123456"}},
                "b": {"items": [{"client_secret": "secret-value-9876"}]},  # nosec B105
                "c": Credential({"token": "wrapped-token-abcdef"}),  # nosec B105
            }
        )

        assert dac.secret_literals() == frozenset({"key-value-123456", "secret-value-9876", "wrapped-token-abcdef"})

    def test_short_and_non_str_secret_values_are_skipped(self) -> None:
        dac = DataAccessCollection(credentials={"a": {"token": "short", "password": 123456789012, "secret": "   "}})  # nosec B105

        assert dac.secret_literals() == frozenset()

    def test_no_credentials_gives_an_empty_set(self) -> None:
        assert DataAccessCollection().secret_literals() == frozenset()


class TestShortSecretWarning:
    """Short credential-shaped str values warn at registration, naming handle and key path."""

    _LOGGER = "mloda.core.abstract_plugins.components.data_access_collection"

    def test_short_password_warns_naming_handle_and_key_without_the_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            dac = DataAccessCollection(credentials={"pg": {"auth": {"password": "hunter"}}})  # nosec B105

        assert "pg" in caplog.text
        assert "auth.password" in caplog.text
        assert "hunter" not in caplog.text
        assert "hunter" not in dac.secret_literals()

    def test_long_value_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            DataAccessCollection(credentials={"pg": {"password": "hunter-long"}})  # nosec B105

        assert caplog.records == []

    def test_blank_value_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            DataAccessCollection(credentials={"pg": {"password": "   "}})  # nosec B105

        assert caplog.records == []

    def test_add_credentials_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        dac = DataAccessCollection()
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            dac.add_credentials("late", {"password": "hunter"})  # nosec B105

        assert "late" in caplog.text
        assert "password" in caplog.text
        assert "hunter" not in caplog.text

    def test_auto_handle_shows_unnamed(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            DataAccessCollection(credentials=[{"password": "hunter"}])  # nosec B105

        assert "<unnamed>" in caplog.text
        assert "_auto_credentials_" not in caplog.text
        assert "hunter" not in caplog.text

    def test_single_credential_form_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            DataAccessCollection(credentials=Credential(password="hunter"))  # nosec B106

        assert "password" in caplog.text
        assert "hunter" not in caplog.text

    def test_safe_key_short_value_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            DataAccessCollection(credentials={"pg": {"host": "db1", "user": "bob"}})

        assert caplog.records == []

    def test_copy_does_not_warn_again(self, caplog: pytest.LogCaptureFixture) -> None:
        dac = DataAccessCollection(credentials={"pg": {"password": "hunter"}})  # nosec B105
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            dac.copy()

        assert caplog.records == []


class TestApiDataColumnsAccessor:
    """DataAccessCollection exposes api_data column specs through a public read-only accessor."""

    def test_default_is_empty(self) -> None:
        assert dict(DataAccessCollection().api_data_columns) == {}

    def test_with_api_data_columns_returns_a_copy_carrying_the_spec(self) -> None:
        base = DataAccessCollection(files={"tx": "/data/tx.csv"})
        spec = {"Cols": ("dac_api_a", "dac_api_b")}

        copy = base.with_api_data_columns(spec)

        assert copy is not base
        assert dict(copy.api_data_columns) == spec
        assert dict(base.api_data_columns) == {}
        assert copy.files == {"tx": "/data/tx.csv"}

    def test_with_api_data_columns_carries_types_through_copy(self) -> None:
        assert dict(DataAccessCollection().api_data_types) == {}

        dac = DataAccessCollection().with_api_data_columns({"Cols": ("dac_api_a",)}, types={"Cols": dict})

        assert dict(dac.api_data_types) == {"Cols": dict}
        assert dict(dac.copy().api_data_types) == {"Cols": dict}
        assert dict(dac.copy().api_data_columns) == {"Cols": ("dac_api_a",)}

    @pytest.mark.parametrize("accessor", ["api_data_columns", "api_data_types"])
    def test_accessor_is_read_only(self, accessor: str) -> None:
        dac = DataAccessCollection().with_api_data_columns({"Cols": ("dac_api_a",)}, types={"Cols": dict})

        with pytest.raises(AttributeError):
            setattr(dac, accessor, {})


_PER_CREDENTIAL_CASES: list[dict[str, Any]] = [
    {"db": {"host": "db.internal.example", "password": "pw-value-123456"}, "api": {"token": TOY_TOKEN}},  # nosec B105
    {"db": {"host": ["shared-value-12345"], "other": ["shared-value-12345"]}},
    {"c": Credential({"token": "wrapped-token-abcdef"}), "d": {"password": {"host": "shaped-host-12345"}}},  # nosec B105
    {"a": {"token": "short", "password": 123456789012, "secret": "   "}},  # nosec B105
    {"same": {"key": "dup-value-123456"}, "also": {"key": "dup-value-123456"}},
    {},
]


class TestCredentialSecretLiteralsPerCredential:
    """credential_secret_literals() over each credential unions to secret_literals()."""

    @pytest.mark.parametrize("credentials", _PER_CREDENTIAL_CASES)
    def test_per_credential_union_equals_secret_literals(self, credentials: dict[str, Any]) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import credential_secret_literals

        dac = DataAccessCollection(credentials=credentials)

        union: frozenset[str] = frozenset().union(*(credential_secret_literals(v) for v in dac.credentials.values()))
        assert union == dac.secret_literals()

    @pytest.mark.parametrize("credentials", _PER_CREDENTIAL_CASES)
    def test_per_credential_union_equals_identity_literals(self, credentials: dict[str, Any]) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import credential_identity_literals

        dac = DataAccessCollection(credentials=credentials)

        union: frozenset[str] = frozenset().union(*(credential_identity_literals(v) for v in dac.credentials.values()))
        assert union == dac.identity_literals()

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param("bare-secret-value-77", frozenset({"bare-secret-value-77"}), id="bare_str_is_a_leaf"),
            pytest.param("short", frozenset(), id="short_bare_str"),
            pytest.param("        ", frozenset(), id="blank_bare_str"),
            pytest.param({"token": "x" * 7}, frozenset(), id="below_the_floor"),
            pytest.param({"token": "x" * 8}, frozenset({"x" * 8}), id="at_the_floor"),
            pytest.param(
                {"host": "db.internal.example", "password": "pw-value-123456"},  # nosec B105
                frozenset({"pw-value-123456"}),
                id="safe_key_leaf_is_not_a_secret",
            ),
        ],
    )
    def test_credential_secret_literals_of_a_value(self, value: Any, expected: frozenset[str]) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import credential_secret_literals

        assert credential_secret_literals(value) == expected

    def test_the_floor_constant_is_eight(self) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import MIN_REDACT_LENGTH

        assert MIN_REDACT_LENGTH == 8


_PW = "pw-value-123456"  # nosec B105
_VAL = "declared-value-12345"


class TestCredentialKeyPolicy:
    """Declared public and secret credential keys widen or narrow what secret_literals() masks."""

    def test_a_key_a_loaded_group_declares_public_is_not_masked(self) -> None:
        from mloda.provider import FeatureGroup

        class _PublicKeyFG(FeatureGroup):
            PUBLIC_CREDENTIAL_KEYS = ("zz_pubkey_alpha",)

        dac = DataAccessCollection(credentials={"a": {"zz_pubkey_alpha": _VAL, "password": _PW}})

        assert dac.secret_literals() == frozenset({_PW})

    def test_a_group_declared_key_is_masked_without_the_group_declaring_it(self) -> None:
        dac = DataAccessCollection(credentials={"a": {"zz_pubkey_unseen": _VAL, "password": _PW}})

        assert dac.secret_literals() == frozenset({_VAL, _PW})

    def test_a_public_field_of_a_handle_spec_is_not_masked(self) -> None:
        from mloda.provider import FeatureGroup, credential_handle_spec

        class _PublicFieldFG(FeatureGroup):
            PROPERTY_MAPPING = {"zz_pubfield_handle": credential_handle_spec("H.", public_fields=("zz_pubfield_beta",))}

        dac = DataAccessCollection(credentials={"a": {"zz_pubfield_beta": _VAL, "password": _PW}})

        assert dac.secret_literals() == frozenset({_PW})

    def test_the_declared_public_keys_are_the_union_over_loaded_groups_normalized(self) -> None:
        from mloda.core.abstract_plugins.components.credential_handle import public_credential_keys
        from mloda.provider import FeatureGroup, credential_handle_spec

        class _OneFG(FeatureGroup):
            PUBLIC_CREDENTIAL_KEYS = ("ZZ_Pub-Gamma",)

        class _TwoFG(FeatureGroup):
            PROPERTY_MAPPING = {"zz_pubunion_handle": credential_handle_spec("H.", public_fields=("ZZ_Pub_Delta",))}

        keys = public_credential_keys()

        assert {"zzpubgamma", "zzpubdelta"} <= keys

    def test_secret_credential_keys_mask_a_base_safe_key(self) -> None:
        dac = DataAccessCollection(
            credentials={"db": {"host": "db.internal.example", "password": _PW}},
            secret_credential_keys={"host"},
        )

        assert dac.secret_literals() == frozenset({"db.internal.example", _PW})

    def test_secret_credential_keys_mask_a_new_base_key_too(self) -> None:
        creds = {"db": {"table": "orders_table_1", "password": _PW}}

        assert DataAccessCollection(credentials=creds).secret_literals() == frozenset({_PW})
        masked = DataAccessCollection(credentials=creds, secret_credential_keys={"table"})
        assert masked.secret_literals() == frozenset({"orders_table_1", _PW})

    def test_secret_wins_over_an_own_public_key(self) -> None:
        dac = DataAccessCollection(
            credentials={"a": {"zz_both_key": _VAL}},
            public_credential_keys={"zz_both_key"},
            secret_credential_keys={"zz_both_key"},
        )

        assert dac.secret_literals() == frozenset({_VAL})

    def test_own_public_keys_unmask_a_non_shaped_key(self) -> None:
        dac = DataAccessCollection(
            credentials={"a": {"zz_own_public": _VAL, "password": _PW}}, public_credential_keys={"zz_own_public"}
        )

        assert dac.secret_literals() == frozenset({_PW})

    @pytest.mark.parametrize("shaped", ["password", "token", "client_secret", "api_key"])
    def test_no_layer_unmasks_a_credential_shaped_key(self, shaped: str) -> None:
        from mloda.provider import FeatureGroup

        class _ShapedPublicFG(FeatureGroup):
            PUBLIC_CREDENTIAL_KEYS = (shaped,)

        dac = DataAccessCollection(credentials={"a": {shaped: _VAL}}, public_credential_keys={shaped})

        assert dac.secret_literals() == frozenset({_VAL})
        assert shaped not in dac.credential_key_policy().secret

    @pytest.mark.parametrize("spelling", ["Key_File", "keyfile", "KEY-FILE", "key_file"])
    def test_keys_are_normalized_when_matching(self, spelling: str) -> None:
        creds = {"a": {"key_file": _VAL, "password": _PW}}

        public = DataAccessCollection(credentials=creds, public_credential_keys={spelling})
        secret = DataAccessCollection(credentials={"a": {"keyfile": _VAL}}, secret_credential_keys={spelling})

        assert public.secret_literals() == frozenset({_PW})
        assert secret.secret_literals() == frozenset({_VAL})

    def test_the_policy_holds_normalized_plugin_and_own_keys(self) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import CredentialKeyPolicy

        dac = DataAccessCollection(public_credential_keys=["Zz_Own-Pub"], secret_credential_keys=("ZZ_own_Sec",))

        policy = dac.credential_key_policy()

        assert isinstance(policy, CredentialKeyPolicy)
        assert "zzownpub" in policy.public
        assert policy.secret == frozenset({"zzownsec"})

    @pytest.mark.parametrize(
        "key", ["table", "database_name", "schema_name", "bucket_name", "project_id", "dataset_id", "file_path"]
    )
    def test_new_base_keys_are_not_masked(self, key: str) -> None:
        dac = DataAccessCollection(credentials={"db": {key: "safe-value-12345", "password": _PW}})

        assert dac.secret_literals() == frozenset({_PW})

    def test_engine_stays_masked(self) -> None:
        dac = DataAccessCollection(credentials={"db": {"engine": "postgresql://h/db1"}})

        assert dac.secret_literals() == frozenset({"postgresql://h/db1"})

    @pytest.mark.parametrize("name", ["public_credential_keys", "secret_credential_keys"])
    @pytest.mark.parametrize("bad", ["host", ["host", ""], [""]])
    def test_a_bare_str_or_empty_item_is_refused(self, name: str, bad: Any) -> None:
        with pytest.raises(ValueError):
            DataAccessCollection(**{name: bad})

    def test_copy_carries_the_keys(self) -> None:
        dac = DataAccessCollection(
            credentials={"a": {"zz_copy_pub": _VAL, "host": "db.internal.example"}},
            public_credential_keys={"zz_copy_pub"},
            secret_credential_keys={"host"},
        )

        duplicate = dac.copy()

        assert duplicate.credential_key_policy() == dac.credential_key_policy()
        assert duplicate.secret_literals() == frozenset({"db.internal.example"})

    def test_identity_literals_and_redact_ignore_the_keys(self) -> None:
        creds = {"a": {"password": _PW, "host": "db.internal.example", "zz_idn_key": _VAL}}
        plain = DataAccessCollection(credentials=creds)
        keyed = DataAccessCollection(
            credentials=creds, public_credential_keys={"zz_idn_key"}, secret_credential_keys={"host"}
        )

        assert keyed.identity_literals() == plain.identity_literals() == frozenset({_PW})
        text = f"saw {_PW} {_VAL} db.internal.example"
        assert keyed.redact(text) == plain.redact(text)

    @pytest.mark.parametrize("bad", ["zz_bare", ("",), (1,)], ids=["bare_str", "empty_item", "non_str_item"])
    def test_malformed_public_credential_keys_raise_at_class_definition(self, bad: Any) -> None:
        from mloda.provider import FeatureGroup

        with pytest.raises(ValueError, match="ZzMalformedKeysFG"):

            class ZzMalformedKeysFG(FeatureGroup):
                PUBLIC_CREDENTIAL_KEYS = bad

    @pytest.mark.parametrize("key", ["dsn", "url", "uri", "conn", "connection_string", "engine", "Connection-String"])
    def test_connection_keys_cannot_be_declared_public(self, key: str) -> None:
        from mloda.provider import FeatureGroup, credential_handle_spec

        with pytest.raises(ValueError):

            class ZzPublicConnKeyFG(FeatureGroup):
                PUBLIC_CREDENTIAL_KEYS = (key,)

        with pytest.raises(ValueError):
            credential_handle_spec("H.", public_fields=(key,))
        with pytest.raises(ValueError):
            DataAccessCollection(public_credential_keys={key})

    def test_connection_keys_can_be_declared_secret(self) -> None:
        dac = DataAccessCollection(secret_credential_keys={"dsn", "Connection-String"})

        assert {"dsn", "connectionstring"} <= dac.credential_key_policy().secret

    @pytest.mark.parametrize("make", [lambda: {"a": 1}, lambda: (k for k in ("a",))], ids=["dict", "generator"])
    def test_key_names_refuse_dict_and_generator(self, make: Any) -> None:
        from mloda.provider import credential_handle_spec

        with pytest.raises(ValueError):
            credential_handle_spec("H.", required_fields=make())
        with pytest.raises(ValueError):
            credential_handle_spec("H.", public_fields=make())
        with pytest.raises(ValueError):
            DataAccessCollection(public_credential_keys=make())
        with pytest.raises(ValueError):
            DataAccessCollection(secret_credential_keys=make())

    @pytest.mark.parametrize("wrap", [tuple, list, set, frozenset])
    def test_key_names_accept_tuple_list_set_frozenset(self, wrap: Any) -> None:
        from mloda.provider import credential_handle_spec

        spec = credential_handle_spec("H.", required_fields=wrap(["zz_ok_key"]))
        dac = DataAccessCollection(public_credential_keys=wrap(["zz_ok_key"]))

        assert spec.required_fields == ("zz_ok_key",)
        assert "zzokkey" in dac.credential_key_policy().public


class TestPolicyAwareLeafHelpers:
    """iter_credential_leaves and credential_secret_literals take an optional CredentialKeyPolicy."""

    def _policy(self, public: set[str], secret: set[str]) -> Any:
        from mloda.core.abstract_plugins.components.credential_leaves import CredentialKeyPolicy

        return CredentialKeyPolicy(public=frozenset(public), secret=frozenset(secret))

    def test_public_key_is_not_secret_and_secret_key_wins(self) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import credential_secret_literals

        value = {"zz_pol_pub": _VAL, "host": "db.internal.example", "password": _PW}
        policy = self._policy({"zzpolpub", "password"}, {"host"})

        assert credential_secret_literals(value, policy) == frozenset({"db.internal.example", _PW})
        assert credential_secret_literals(value, None) == credential_secret_literals(value)

    def test_iter_leaves_marks_shaped_under_a_public_shaped_key(self) -> None:
        from mloda.core.abstract_plugins.components.credential_leaves import iter_credential_leaves

        (leaf,) = list(iter_credential_leaves({"token": _VAL}, self._policy({"token"}, set())))

        assert leaf.secret and leaf.shaped
