"""Failing tests for staged sqlite string value sampling (epic #518, perf follow-up).

``sample_string_values`` currently issues one ``SELECT <col> ... LIMIT 100`` query for
every string-typed equi-join key, even genuine string IDs that are never datetimes. That
forces a bounded scan (often materializing a temp VIEW) on hot join paths.

The refined behavior is fast-reject: issue a single ``LIMIT 1`` probe query first and only
fetch the fuller ``LIMIT 100`` sample when the probed first value is an ISO-8601 date/datetime
string. The ISO-8601 classifier disqualifies a column on its first non-ISO value, so this
preserves classification while skipping the larger scan for non-datetime string columns.

These tests use a lightweight stub connection (no real sqlite) that records each executed
SQL string and returns canned rows per call. They fail today because the current
implementation always issues exactly one full-limit query and never probes: the ISO case
below expects two execute calls but only one is issued.
"""

from typing import Any

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_value_sample import sample_string_values


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConnection:
    """Records each executed SQL string; returns canned rows in call order."""

    def __init__(self, responses: list[list[tuple[Any, ...]]]) -> None:
        self._responses = responses
        self.executed_sql: list[str] = []

    def execute(self, sql: str) -> _FakeCursor:
        index = len(self.executed_sql)
        self.executed_sql.append(sql)
        rows = self._responses[index] if index < len(self._responses) else []
        return _FakeCursor(rows)


class _FakeData:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.table_name = "t"


class TestStagedSampling:
    """Only ISO-8601 first values trigger the fuller second query."""

    def test_non_iso_first_value_issues_single_probe(self) -> None:
        connection = _FakeConnection([[("abc123",)]])
        data = _FakeData(connection)

        result = sample_string_values(data, "c")

        assert len(connection.executed_sql) == 1
        assert result == []

    def test_iso_first_value_issues_probe_then_full_sample(self) -> None:
        connection = _FakeConnection(
            [
                [("2024-01-01T00:00:00",)],
                [("2024-01-01T00:00:00",), ("2024-06-01T12:30:00",)],
            ]
        )
        data = _FakeData(connection)

        result = sample_string_values(data, "c")

        assert len(connection.executed_sql) == 2
        assert result == ["2024-01-01T00:00:00", "2024-06-01T12:30:00"]

    def test_empty_column_issues_single_probe(self) -> None:
        connection = _FakeConnection([[]])
        data = _FakeData(connection)

        result = sample_string_values(data, "c")

        assert len(connection.executed_sql) == 1
        assert result == []
