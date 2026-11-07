import tempfile

import pyarrow as pa
import pytest

pyiceberg = pytest.importorskip("pyiceberg")
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402

from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer import (  # noqa: E402
    IcebergPyArrowTransformer,
)


@pytest.fixture()
def catalog() -> SqlCatalog:
    tmp = tempfile.mkdtemp()
    cat = SqlCatalog("c", uri=f"sqlite:///{tmp}/c.db", warehouse=f"file://{tmp}")
    cat.create_namespace("n")
    return cat


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("s", pa.string()),
            ("b", pa.binary()),
            ("l", pa.list_(pa.int64())),
        ]
    )


def _table(catalog: SqlCatalog, name: str, rows: int) -> pa.Table:
    schema = _schema()
    tbl = catalog.create_table(f"n.{name}", schema=schema)
    data = pa.table(
        {
            "s": ["a"] * rows,
            "b": [b"x"] * rows,
            "l": [[1, 2]] * rows,
        },
        schema=schema,
    )
    tbl.append(data)
    return tbl


def test_zero_rows_and_non_empty_share_schema(catalog: SqlCatalog) -> None:
    full = _table(catalog, "full", 1)
    empty = _table(catalog, "empty", 0)

    full_arrow = IcebergPyArrowTransformer.transform_fw_to_other_fw(full)
    empty_arrow = IcebergPyArrowTransformer.transform_fw_to_other_fw(empty)

    assert empty_arrow.schema == full_arrow.schema
    assert empty_arrow.schema.field("s").type == pa.string()
    assert empty_arrow.schema.field("b").type == pa.binary()
    assert empty_arrow.schema.field("l").type == pa.list_(pa.int64())
