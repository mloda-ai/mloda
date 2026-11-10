"""Shared SqliteFG host body for the database loads-into-framework suite, kept out of any test module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.source.loads_into_framework_suite import loads_data_access
from mloda.provider import ReadDBFG
from mloda.user import GlobalFilter, PluginCollector, mloda
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite

SQLITE_KEY = "sqlite"


class SqliteLoadsHelper:
    feature_group_class: type[ReadDBFG] = SqliteFG
    credential_key = SQLITE_KEY

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def test_a_global_filter_fetches_fewer_rows(
        self, adapter: FrameworkAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        columns = ("dbloads_a", "dbloads_b", "dbloads_c")
        path = tmp_path / "dbloads_filtered.db"
        self.write_database(path, {"dbloads_table": {name: [1, 2, 3, 4, 5, 6] for name in columns}})
        original = SqliteFG._fetch
        fetched: list[int] = []

        def spy(cls: Any, connection: Any, table: str, features: Any) -> Any:
            result = original(connection, table, features)
            fetched.append(len(result[1]))
            return result

        monkeypatch.setattr(SqliteFG, "_fetch", classmethod(spy))
        global_filter = GlobalFilter()
        global_filter.add_filter("dbloads_a", "range", {"min": 2, "max": 4, "max_exclusive": True})

        mloda.run_all(
            list(columns[:2]),
            compute_frameworks=[adapter.compute_framework()],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            data_access_collection=loads_data_access(adapter, credentials={"dbloads_handle": {SQLITE_KEY: str(path)}}),
            global_filter=global_filter,
        )

        assert fetched == [2]
