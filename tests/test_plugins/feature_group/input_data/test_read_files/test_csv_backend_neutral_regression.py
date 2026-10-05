"""Regression guards for the compute-framework-neutral file seam, then the format x framework matrix.

Reading a CSV via ``run_all`` yields the requested columns on PyArrowTable, PandasDataFrame and PolarsDataFrame;
the matrix classes below cover every stock format group per framework. No backend objects are built at import,
class or parametrize time.
"""

from __future__ import annotations

import csv
import importlib
import os
import tempfile
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.provider import ReadFileFG
from mloda.user import DataAccessCollection, mloda
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    FileLoaderNameMixin,
    FileLoadsIntoFrameworkMixin,
    PandasDataFrameAdapter,
    PolarsDataFrameAdapter,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.format_file_writers import (
    write_csv,
    write_feather,
    write_json,
    write_orc,
    write_parquet,
)
from tests.mixins.reader_feature_groups.lazy_format_group import lazy_group


@pytest.fixture(autouse=True)
def _stock_formats_loaded() -> None:
    importlib.import_module("mloda_plugins.feature_group.input_data.file_formats.stock_formats")
    importlib.import_module("mloda_plugins.compute_framework.base_implementations.pyarrow.table")
    importlib.import_module("mloda_plugins.compute_framework.base_implementations.pandas.dataframe")


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
    importlib.import_module("mloda_plugins.compute_framework.base_implementations.polars.dataframe")
    result = mloda.run_all(
        ["A", "B"],
        compute_frameworks=["PolarsDataFrame"],
        data_access_collection=DataAccessCollection(files={csv_path}),
    )
    columns: Any = list(result[0].columns)
    assert "A" in columns
    assert "B" in columns


class _CsvFiles:
    file_group = lazy_group("csv_fg", "CsvFG")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_csv(path, columns)


class _ParquetFiles:
    file_group = lazy_group("parquet_fg", "ParquetFG")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_parquet(path, columns)


class _JsonFiles:
    file_group = lazy_group("json_fg", "JsonFG")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_json(path, columns)


class _FeatherFiles:
    file_group = lazy_group("feather_fg", "FeatherFG")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_feather(path, columns)


class _OrcFiles:
    file_group = lazy_group("orc_fg", "OrcFG")

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_orc(path, columns)


class _DefaultNeutralParquetFG(ReadFileFG):
    """A third-party style Parquet group that keeps the default FileSource ``load_neutral``."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".dnparquet",)

    @classmethod
    def file_format(cls) -> str:
        return "parquet"

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        import pyarrow.parquet as pyarrow_parquet

        return list(pyarrow_parquet.read_schema(path).names)


class _DefaultNeutralParquetFiles:
    file_group = _DefaultNeutralParquetFG

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_parquet(path, columns)


class TestDefaultNeutralParquetIntoPythonDict(
    _DefaultNeutralParquetFiles, PythonDictAdapter, FileLoadsIntoFrameworkMixin
):
    pass


class TestDefaultNeutralParquetIntoPandasDataFrame(
    _DefaultNeutralParquetFiles, PandasDataFrameAdapter, FileLoadsIntoFrameworkMixin
):
    pass


class TestCsvIntoPyArrowTable(_CsvFiles, PyArrowTableAdapter, FileLoaderNameMixin):
    expected_loader = "PyArrowTable"


class TestCsvIntoPandasDataFrame(_CsvFiles, PandasDataFrameAdapter, FileLoaderNameMixin):
    expected_loader = "neutral"


class TestCsvIntoPolarsDataFrame(_CsvFiles, PolarsDataFrameAdapter, FileLoadsIntoFrameworkMixin):
    pass


class TestCsvIntoPythonDict(_CsvFiles, PythonDictAdapter, FileLoadsIntoFrameworkMixin):
    pass


class TestParquetIntoPyArrowTable(_ParquetFiles, PyArrowTableAdapter, FileLoaderNameMixin):
    expected_loader = "PyArrowTable"


class TestParquetIntoPandasDataFrame(_ParquetFiles, PandasDataFrameAdapter, FileLoaderNameMixin):
    expected_loader = "neutral"


class TestJsonIntoPyArrowTable(_JsonFiles, PyArrowTableAdapter, FileLoaderNameMixin):
    expected_loader = "PyArrowTable"


class TestJsonIntoPandasDataFrame(_JsonFiles, PandasDataFrameAdapter, FileLoaderNameMixin):
    expected_loader = "neutral"


class TestFeatherIntoPyArrowTable(_FeatherFiles, PyArrowTableAdapter, FileLoaderNameMixin):
    expected_loader = "PyArrowTable"


class TestFeatherIntoPandasDataFrame(_FeatherFiles, PandasDataFrameAdapter, FileLoaderNameMixin):
    expected_loader = "neutral"


class TestOrcIntoPyArrowTable(_OrcFiles, PyArrowTableAdapter, FileLoaderNameMixin):
    expected_loader = "PyArrowTable"


class TestOrcIntoPandasDataFrame(_OrcFiles, PandasDataFrameAdapter, FileLoaderNameMixin):
    expected_loader = "neutral"
