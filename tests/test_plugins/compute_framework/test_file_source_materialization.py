"""A ``FileSource`` descriptor materializes via the transformer chain (with pyarrow installed).

* ``PyArrowTable``  -> ``pa.Table`` via the direct ``(FileSource, pa.Table)`` transformer
  (``FileSourcePyArrowTransformer``), honoring the requested columns via include_columns.
* ``PandasDataFrame`` -> ``pandas.DataFrame`` via the 2-hop
  ``FileSource -> pa.Table -> pandas`` chain (FileSource is a descriptor, so multi-hop
  is allowed for it; a plain dict is NOT rerouted, see test_native_dict_ingestion.py).
* ``PythonDictFramework`` -> ``dict`` via the direct stdlib ``(FileSource, dict)`` transformer.
* ``FileSourcePyArrowTransformer`` rejects non-csv formats with ``ValueError`` and the
  reverse direction (pa.Table -> FileSource) with ``NotImplementedError``.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    FileSourcePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


@pytest.fixture()
def csv_path() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B", "C"])
        writer.writerow(["1", "3", "x"])
        writer.writerow(["2", "4", "y"])
    yield path
    os.remove(path)


def _file_source(csv_path: str) -> FileSource:
    return FileSource(path=csv_path, format="csv", columns=("A", "B"))


class TestFileSourceMaterializesPerFramework:
    def test_file_source_materializes_to_pyarrow_table(self, csv_path: str) -> None:
        fw = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        result = fw.transform(_file_source(csv_path), ["A", "B"])
        assert isinstance(result, pa.Table)
        assert set(result.column_names) == {"A", "B"}

    def test_file_source_materializes_to_pandas_dataframe(self, csv_path: str) -> None:
        """FileSource is a descriptor, so the 2-hop FileSource -> pa.Table -> pandas chain applies."""
        fw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        result = fw.transform(_file_source(csv_path), ["A", "B"])
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"A", "B"}

    def test_file_source_materializes_to_python_dict(self, csv_path: str) -> None:
        fw = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        result = fw.transform(_file_source(csv_path), ["A", "B"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B"}
        assert result["A"] == [1, 2]
        assert result["B"] == [3, 4]


class TestFileSourcePyArrowTransformerContract:
    def test_only_requested_columns_are_included(self, csv_path: str) -> None:
        """The pyarrow materialization selects exactly ``FileSource.columns`` (include_columns)."""
        table = FileSourcePyArrowTransformer.transform_fw_to_other_fw(_file_source(csv_path))
        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"A", "B"}
        assert table.column("A").to_pylist() == [1, 2]

    def test_non_csv_format_raises_valueerror(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            FileSourcePyArrowTransformer.transform_fw_to_other_fw(
                FileSource(path="/nonexistent.parquet", format="parquet", columns=("A",))
            )
        assert "parquet" in str(excinfo.value)

    def test_reverse_direction_raises_not_implemented(self) -> None:
        """pa.Table -> FileSource makes no sense; the reverse direction must raise."""
        with pytest.raises(NotImplementedError):
            FileSourcePyArrowTransformer.transform_other_fw_to_fw(pa.table({"A": [1]}))
