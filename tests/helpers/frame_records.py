"""Backend-neutral extraction of native frame rows to plain Python dicts."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


def records_from_frame(data: Any) -> list[dict[str, Any]]:
    """Extract native frame rows to plain Python row dicts across backends.

    Framework-agnostic: a zero-row result is simply ``len(records_from_frame(result)) == 0``.
    """
    records: list[dict[str, Any]]
    if pl is not None and isinstance(data, pl.LazyFrame):
        records = data.collect().to_dicts()
    elif pl is not None and isinstance(data, pl.DataFrame):
        records = data.to_dicts()
    elif pa is not None and isinstance(data, pa.Table):
        records = data.to_pylist()
    elif isinstance(data, dict):
        # PythonDict columnar frame: {"col": [v0, v1, ...]}. Zero columns -> zero rows.
        if not data:
            records = []
        else:
            length = len(next(iter(data.values())))
            records = [{k: data[k][i] for k in data} for i in range(length)]
    elif isinstance(data, list):
        records = data
    elif hasattr(data, "to_dicts"):
        records = data.to_dicts()
    elif hasattr(data, "to_dict"):
        records = data.to_dict("records")
    elif hasattr(data, "toPandas"):
        # Spark DataFrame
        records = [row.asDict() for row in data.collect()]
    else:
        records = data.df().to_dict("records")
    return records
