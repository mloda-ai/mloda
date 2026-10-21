"""MatchData must mark the class-name key it stashes a live framework connection under as
non-forwarded, both when matched through global scope (add_base_input_data_to_options) and
through feature scope (feature_scope_data_access), so the connection never leaks to upstream
input features via Options.inherit_from or Features.build_feature_collection."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import (
    CONNECTION_HANDLE_OPTION,
    DataAccessCollection,
)
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.options import Options


class _StubConnection:
    """Stands in for a live framework connection object (e.g. duckdb.DuckDBPyConnection)."""


class _StubMatchDataFG(MatchData):
    """Minimal MatchData subclass: matches only a _StubConnection instance, either scope (mirrors
    the real isinstance-gated matchers so a call that's invoked but declines is reachable)."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if isinstance(framework_connection_object, _StubConnection):
            return framework_connection_object
        if data_access_collection is not None:
            for conn in data_access_collection.connections.values():
                if isinstance(conn, _StubConnection):
                    return conn
        return None


class TestGlobalScopeDataAccessWritesTheConnectionHandle:
    """A DAC connection match records its DAC handle under connection_handle, never the object."""

    def test_match_sets_connection_handle_to_the_dac_handle(self) -> None:
        options = Options()
        conn = _StubConnection()
        # The stub returns the first _StubConnection, so "handle" is the match; "first" and "last" must not win.
        dac = DataAccessCollection(connections={"first": "not_a_connection", "handle": conn, "last": _StubConnection()})

        matched = _StubMatchDataFG.global_scope_data_access("some_feature", options, dac)

        assert matched is True
        assert options.get(CONNECTION_HANDLE_OPTION) == "handle"

    def test_match_writes_nothing_under_the_class_name_key(self) -> None:
        options = Options()
        dac = DataAccessCollection(connections={"handle": _StubConnection()})

        _StubMatchDataFG.global_scope_data_access("some_feature", options, dac)

        assert _StubMatchDataFG.get_class_name() not in options
        assert _StubMatchDataFG.get_class_name() not in options.non_forwarded_group_keys

    def test_an_equal_existing_connection_handle_is_kept(self) -> None:
        options = Options(group={CONNECTION_HANDLE_OPTION: "handle"})
        dac = DataAccessCollection(connections={"handle": _StubConnection()})

        matched = _StubMatchDataFG.global_scope_data_access("some_feature", options, dac)

        assert matched is True
        assert options.get(CONNECTION_HANDLE_OPTION) == "handle"

    def test_a_different_existing_connection_handle_raises(self) -> None:
        options = Options(group={CONNECTION_HANDLE_OPTION: "other"})
        dac = DataAccessCollection(connections={"handle": _StubConnection()})

        with pytest.raises(ValueError, match="already set with different values"):
            _StubMatchDataFG.global_scope_data_access("some_feature", options, dac)

    def test_no_match_does_not_mark(self) -> None:
        """A non-None DAC reaches match_data_access, whose isinstance check declines a non-_StubConnection value."""
        options = Options()
        dac = DataAccessCollection(connections={"handle": "not_a_connection"})

        matched = _StubMatchDataFG.global_scope_data_access("some_feature", options, dac)

        assert matched is False
        assert options.non_forwarded_group_keys == frozenset()


class TestFeatureScopeDataAccessMarksNonForwarded:
    """feature_scope_data_access marks the class-name key non-forwarded after a successful match,
    even though the key was placed into options directly by the caller."""

    def test_match_marks_class_name_key_non_forwarded(self) -> None:
        options = Options(group={_StubMatchDataFG.get_class_name(): _StubConnection()})

        matched = _StubMatchDataFG.feature_scope_data_access(options, "some_feature")

        assert matched is True
        assert _StubMatchDataFG.get_class_name() in options.non_forwarded_group_keys

    def test_no_match_does_not_mark(self) -> None:
        """A truthy, non-_StubConnection value clears the early guard and reaches match_data_access,
        whose isinstance check declines it."""
        options = Options(group={_StubMatchDataFG.get_class_name(): "not_a_connection"})

        matched = _StubMatchDataFG.feature_scope_data_access(options, "some_feature")

        assert matched is False
        assert options.non_forwarded_group_keys == frozenset()


class TestAddBaseInputDataToOptionsMarksNonForwarded:
    """add_base_input_data_to_options routes the write through add_to_group(forward=False)."""

    def test_marks_class_name_key_non_forwarded(self) -> None:
        options = Options()
        conn = _StubConnection()

        _StubMatchDataFG.add_base_input_data_to_options(conn, options)

        assert _StubMatchDataFG.get_class_name() in options.non_forwarded_group_keys
        assert options.get(_StubMatchDataFG.get_class_name()) is conn

    def test_idempotent_early_return_also_marks_non_forwarded(self) -> None:
        """The 'already set with an equal value' early-return branch must ALSO mark the key
        non-forwarded: construct options with the key already present, unmarked, bypassing
        add_to_group's own forward=False, so only this call's branch can produce the mark."""
        conn = _StubConnection()
        cls_name = _StubMatchDataFG.get_class_name()
        options = Options()
        options.add_to_group(cls_name, conn)  # default forward=True: pre-existing, unmarked
        assert cls_name not in options.non_forwarded_group_keys

        _StubMatchDataFG.add_base_input_data_to_options(conn, options)

        assert cls_name in options.non_forwarded_group_keys
