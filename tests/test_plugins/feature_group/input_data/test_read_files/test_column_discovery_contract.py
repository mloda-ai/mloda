"""Concrete describe_columns/get_column_names contract runs, one per pyarrow-backed reader."""

import json
from pathlib import Path

import pyarrow as pa
import pytest

from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader

from tests.test_plugins.feature_group.input_data.test_read_files.column_discovery_contract_test_mixin import (
    PHYSICAL_COLUMNS,
    ColumnDiscoveryContractTestMixin,
)


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
