"""Guard tests: a ``FileSource`` descriptor MUST still materialize via the transformer chain.

These pin the FileSource path so the fix that stops plain dicts from being rerouted
through ``pa.Table`` does not over-restrict and also block the (legitimate) FileSource
materialization. With pyarrow installed:

  * ``PyArrowTable``  -> ``pa.Table`` via the direct ``(FileSource, pa.Table)`` transformer.
  * ``PandasDataFrame`` -> ``pandas.DataFrame`` via the 2-hop
    ``FileSource -> pa.Table -> pandas`` chain (FileSource is a descriptor, so multi-hop
    is allowed for it).
  * ``PythonDictFramework`` -> ``dict`` via the direct stdlib ``(FileSource, dict)`` transformer.

They pass today and must keep passing after the fix.
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
        writer.writerow(["A", "B"])
        writer.writerow(["1", "3"])
        writer.writerow(["2", "4"])
    yield path
    os.remove(path)


def _file_source(csv_path: str) -> FileSource:
    return FileSource(path=csv_path, format="csv", columns=("A", "B"))


def test_file_source_materializes_to_pyarrow_table(csv_path: str) -> None:
    fw = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    result = fw.transform(_file_source(csv_path), {"A", "B"})
    assert isinstance(result, pa.Table)
    assert set(result.column_names) == {"A", "B"}


def test_file_source_materializes_to_pandas_dataframe(csv_path: str) -> None:
    fw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    result = fw.transform(_file_source(csv_path), {"A", "B"})
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"A", "B"}


def test_file_source_materializes_to_python_dict(csv_path: str) -> None:
    fw = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    result = fw.transform(_file_source(csv_path), {"A", "B"})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"A", "B"}
    assert result["A"] == [1, 2]
    assert result["B"] == [3, 4]
