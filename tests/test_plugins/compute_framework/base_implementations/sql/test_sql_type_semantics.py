"""Failing tests for the SQL-family column-semantics helper (epic #518, Phase 1).

Defines the not-yet-implemented module-level function
``column_semantics_from_arrow(arrow_type, value_sample=None)`` in
``mloda_plugins.compute_framework.base_implementations.sql.sql_type_semantics``.
This helper backs both duckdb and sqlite (which store datetimes as ISO TEXT).
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

import pytest

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_type_semantics import column_semantics_from_arrow

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqlColumnSemanticsFromArrow:
    def test_timestamp_naive(self) -> None:
        sem = column_semantics_from_arrow(pa.timestamp("us"))
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit == "us"

    def test_timestamp_aware(self) -> None:
        sem = column_semantics_from_arrow(pa.timestamp("us", "UTC"))
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit == "us"

    def test_int(self) -> None:
        sem = column_semantics_from_arrow(pa.int64())
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_string(self) -> None:
        sem = column_semantics_from_arrow(pa.string())
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_string_storage_flag_does_not_value_scan(self) -> None:
        # sqlite stores datetimes as ISO TEXT: with a bare string type and no value_sample,
        # we do NOT value-scan, so it stays a plain non-ordered string.
        sem = column_semantics_from_arrow(pa.string())
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_large_string_value_sample_classifies_temporal(self) -> None:
        # pa.types.is_string() is False for large_string, so value-inspection must also
        # cover large_string (and string_view) column types.
        sem = column_semantics_from_arrow(pa.large_string(), value_sample=["2024-01-01T00:00:00"])
        assert sem.is_temporal is True
        assert sem.is_ordered is True


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqlTimezoneIdentity:
    @pytest.mark.parametrize("zone", ["UTC", "Europe/Berlin", "+05:30"])
    def test_timestamp_schema_keeps_zone(self, zone: str) -> None:
        sem = column_semantics_from_arrow(pa.timestamp("ns", tz=zone))
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True
        assert sem.unit == "ns"
        assert getattr(sem, "tz") == zone

    @pytest.mark.parametrize(
        "type_name", [name for name in ("string", "large_string", "string_view") if hasattr(pa, name)]
    )
    @pytest.mark.parametrize(
        ("values", "zone", "aware", "temporal"),
        [
            ([None, "2024-01-01T00:00:00Z", "2024-06-01T00:00:00+00:00"], "UTC", True, True),
            (["2024-01-01T00:00:00+05:30", "2024-06-01T00:00:00+05:30"], "UTC+05:30", True, True),
            (["2024-01-01T00:00:00+01:00", "2024-06-01T00:00:00+02:00"], None, True, True),
            (["2024-01-01T00:00:00", "2024-06-01T00:00:00"], None, False, True),
            ([None, None], None, False, False),
        ],
        ids=["utc-spellings", "fixed-offset", "mixed-offsets", "naive", "all-null"],
    )
    def test_string_sample_preserves_zone_information(
        self, type_name: str, values: list[str | None], zone: str | None, aware: bool, temporal: bool
    ) -> None:
        sem = column_semantics_from_arrow(getattr(pa, type_name)(), value_sample=values)
        assert sem.is_temporal is temporal
        assert sem.is_tz_aware is aware
        assert getattr(sem, "tz") == zone
