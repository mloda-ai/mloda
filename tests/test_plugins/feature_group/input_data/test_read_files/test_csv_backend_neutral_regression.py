"""Regression guards for the compute-framework-neutral CSV seam.

These pass TODAY (they exercise the current pyarrow path) and must KEEP passing after
CsvReader is reframed around ``FileSource``:

  * PyArrowTable exercises the ``FileSource -> pa.Table`` direct materialization.
  * PandasDataFrame exercises the ``FileSource -> pa.Table -> pandas`` chained path.
  * PolarsDataFrame exercises the ``FileSource -> pa.Table -> polars`` chained path.

They only assert that reading a CSV via ``run_all`` yields the requested columns, so they
stay valid across the refactor. No custom reader/feature-group subclasses are defined, so
nothing leaks into the global plugin registry.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.user import DataAccessCollection
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature  # noqa: F401
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader  # noqa: F401


@pytest.fixture()
def csv_path() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B"])
        writer.writerow(["1", "3"])
        writer.writerow(["2", "4"])
    yield path
    os.remove(path)


def test_csv_reads_into_pyarrow_table(csv_path: str) -> None:
    """FileSource -> pa.Table direct path: PyArrowTable still returns A and B."""
    result = mloda.run_all(
        ["A", "B"],
        compute_frameworks=["PyArrowTable"],
        data_access_collection=DataAccessCollection(files={csv_path}),
    )
    columns = result[0].to_pydict()
    assert "A" in columns
    assert "B" in columns


def test_csv_reads_into_pandas_dataframe(csv_path: str) -> None:
    """FileSource -> pa.Table -> pandas chained path: PandasDataFrame still returns A and B."""
    result = mloda.run_all(
        ["A", "B"],
        compute_frameworks=["PandasDataFrame"],
        data_access_collection=DataAccessCollection(files={csv_path}),
    )
    columns: Any = list(result[0].columns)
    assert "A" in columns
    assert "B" in columns


def test_csv_reads_into_polars_dataframe(csv_path: str) -> None:
    """FileSource -> pa.Table -> polars chained path: PolarsDataFrame still returns A and B."""
    pytest.importorskip("polars")
    result = mloda.run_all(
        ["A", "B"],
        compute_frameworks=["PolarsDataFrame"],
        data_access_collection=DataAccessCollection(files={csv_path}),
    )
    columns: Any = list(result[0].columns)
    assert "A" in columns
    assert "B" in columns
