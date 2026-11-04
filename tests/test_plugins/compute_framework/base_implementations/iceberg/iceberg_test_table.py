"""Shared handle for FeatureGroup classmethods, which cannot take fixtures; the iceberg_table_source fixture sets it."""

from typing import Any

adapter_handle: list[Any] = []


def real_iceberg_table(arrow_table: Any) -> Any:
    """Build a real native IcebergTable from an Arrow table via the published adapter."""
    if not adapter_handle:
        raise RuntimeError("real_iceberg_table needs the iceberg_table_source fixture")
    return adapter_handle[0].from_arrow(arrow_table)
