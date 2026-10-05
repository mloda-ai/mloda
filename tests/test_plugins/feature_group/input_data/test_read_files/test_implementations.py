"""Contract classes for the five stock file format groups, plus a read-equals-pyarrow check per format."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pyarrow_csv
import pyarrow.json as pyarrow_json
import pyarrow.orc as pyarrow_orc
import pyarrow.parquet as pyarrow_parquet
import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    FileSourcePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.file_format_feature_group_test_mixin import FileFormatFeatureGroupTestMixin
from tests.mixins.reader_feature_groups.format_file_writers import (
    write_csv,
    write_feather,
    write_json,
    write_orc,
    write_parquet,
)
from tests.mixins.reader_feature_groups.lazy_format_group import lazy_group, load_group


def _read_feather(source: Any, columns: list[str] | None = None) -> Any:
    """Non-deprecated Feather V2 reader used as the comparison oracle."""
    with pa.ipc.open_file(source) as reader:
        table = reader.read_all()
    return table.select(columns) if columns is not None else table


class TestCsvFG(FileFormatFeatureGroupTestMixin):
    feature_group_class = lazy_group("csv_fg", "CsvFG")
    present_column = "impl_csv_present"
    missing_column = "impl_csv_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_csv(path, columns)


class TestParquetFG(FileFormatFeatureGroupTestMixin):
    feature_group_class = lazy_group("parquet_fg", "ParquetFG")
    present_column = "impl_parquet_present"
    missing_column = "impl_parquet_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_parquet(path, columns)


class TestJsonFG(FileFormatFeatureGroupTestMixin):
    feature_group_class = lazy_group("json_fg", "JsonFG")
    present_column = "impl_json_present"
    missing_column = "impl_json_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_json(path, columns)


class TestFeatherFG(FileFormatFeatureGroupTestMixin):
    feature_group_class = lazy_group("feather_fg", "FeatherFG")
    present_column = "impl_feather_present"
    missing_column = "impl_feather_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_feather(path, columns)


class TestOrcFG(FileFormatFeatureGroupTestMixin):
    feature_group_class = lazy_group("orc_fg", "OrcFG")
    present_column = "impl_orc_present"
    missing_column = "impl_orc_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_orc(path, columns)


class _Features:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def get_all_names(self) -> list[str]:
        return self.columns


_FORMATS: list[tuple[str, str, str, Callable[..., None], Callable[..., Any]]] = [
    ("feather_fg", "FeatherFG", ".feather", write_feather, _read_feather),
    ("json_fg", "JsonFG", ".json", write_json, lambda path, columns=None: pyarrow_json.read_json(path)),
    ("csv_fg", "CsvFG", ".csv", write_csv, lambda path, columns=None: pyarrow_csv.read_csv(path)),
    ("orc_fg", "OrcFG", ".orc", write_orc, pyarrow_orc.read_table),
    ("parquet_fg", "ParquetFG", ".parquet", write_parquet, pyarrow_parquet.read_table),
]
_DATA = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
_WANTED = ["col1", "col2"]


class TestFormatGroupsReadLikePyArrow:
    """Each group's neutral form and its PyArrowTable loader hold what pyarrow reads from the same file."""

    @staticmethod
    def _expected(reader: Callable[..., Any], path: Path, whole_file_reader: bool) -> dict[str, Any]:
        table = reader(str(path)).select(_WANTED) if whole_file_reader else reader(str(path), columns=_WANTED)
        return dict(table.to_pydict())

    @pytest.mark.parametrize("module_name,class_name,suffix,writer,reader", _FORMATS, ids=[f[1] for f in _FORMATS])
    def test_load_neutral_equals_the_pyarrow_read(
        self, tmp_path: Path, module_name: str, class_name: str, suffix: str, writer: Any, reader: Any
    ) -> None:
        group = load_group(module_name, class_name)
        path = tmp_path / f"neutral{suffix}"
        writer(path, _DATA)

        result = group.load_neutral(SourceMatch(source=str(path), access=str(path)), _Features(_WANTED))

        if class_name == "CsvFG":
            assert isinstance(result, FileSource)
            result = FileSourcePyArrowTransformer.transform_fw_to_other_fw(result)
        whole_file = class_name in ("CsvFG", "JsonFG")
        assert result.to_pydict() == self._expected(reader, path, whole_file)

    @pytest.mark.parametrize("module_name,class_name,suffix,writer,reader", _FORMATS, ids=[f[1] for f in _FORMATS])
    def test_pyarrow_table_loader_equals_the_pyarrow_read(
        self, tmp_path: Path, module_name: str, class_name: str, suffix: str, writer: Any, reader: Any
    ) -> None:
        group = load_group(module_name, class_name)
        path = tmp_path / f"loader{suffix}"
        writer(path, _DATA)

        loader = group._loader_for(PyArrowTable)

        assert loader is not None
        result = loader(SourceMatch(source=str(path), access=str(path)), _Features(_WANTED))
        whole_file = class_name in ("CsvFG", "JsonFG")
        assert result.to_pydict() == self._expected(reader, path, whole_file)
