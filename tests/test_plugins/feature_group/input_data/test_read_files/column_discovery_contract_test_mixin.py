"""Shared describe_columns/get_column_names contract for pyarrow-backed ReadFile readers.

Without its optional pyarrow submodule, a reader must raise a bare NotImplementedError from
both methods; ReadFile.validate_columns relies on that to treat "can't confirm" as non-fatal.
Named without a ``Test`` prefix so pytest does not collect it standalone.
"""

import importlib
import json
from pathlib import Path

import pyarrow as pa
import pytest

from mloda.user import DataType
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader

PHYSICAL_COLUMNS = ["c1", "a1", "b1"]


class ColumnDiscoveryContractTestMixin:
    """Shared get_column_names/describe_columns contract for one ReadFile reader."""

    reader_cls: type[ReadFile]
    dependency_module: str
    dependency_attr: str

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        """Return a file path with columns c1, a1, b1 (all INT64). Override per reader."""
        raise NotImplementedError

    def test_describe_columns_returns_real_types(self, data_file: str) -> None:
        described = self.reader_cls.describe_columns(data_file)
        assert described == {name: DataType.INT64 for name in PHYSICAL_COLUMNS}

    def test_get_column_names_matches_describe_columns_keys(self, data_file: str) -> None:
        names = self.reader_cls.get_column_names(data_file)
        described = self.reader_cls.describe_columns(data_file)
        assert set(names) == set(described.keys())

    def test_raises_not_implemented_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, data_file: str
    ) -> None:
        module = importlib.import_module(self.dependency_module)
        monkeypatch.setattr(module, self.dependency_attr, None)

        with pytest.raises(NotImplementedError):
            self.reader_cls.get_column_names(data_file)
        with pytest.raises(NotImplementedError):
            self.reader_cls.describe_columns(data_file)


class TestFeatherColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    reader_cls = FeatherReader
    dependency_module = "mloda_plugins.feature_group.input_data.read_files.feather"
    dependency_attr = "pyarrow_ipc"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = str(tmp_path / "test.feather")
        with pa.OSFile(file_path, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        return file_path


class TestOrcColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    reader_cls = OrcReader
    dependency_module = "mloda_plugins.feature_group.input_data.read_files.orc"
    dependency_attr = "pyarrow_orc"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        import pyarrow.orc as pyarrow_orc

        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = str(tmp_path / "test.orc")
        with pa.OSFile(file_path, "wb") as f:
            pyarrow_orc.write_table(table, f)
        return file_path


class TestJsonColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    reader_cls = JsonReader
    dependency_module = "mloda_plugins.feature_group.input_data.read_files.json"
    dependency_attr = "pyarrow_json"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = tmp_path / "test.json"
        rows = [{col: table.column(col)[i].as_py() for col in PHYSICAL_COLUMNS} for i in range(table.num_rows)]
        with open(file_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(file_path)


class TestParquetColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    reader_cls = ParquetReader
    dependency_module = "mloda_plugins.feature_group.input_data.read_files.parquet"
    dependency_attr = "pyarrow_parquet"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        import pyarrow.parquet as pyarrow_parquet

        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = str(tmp_path / "test.parquet")
        pyarrow_parquet.write_table(table, file_path)
        return file_path
