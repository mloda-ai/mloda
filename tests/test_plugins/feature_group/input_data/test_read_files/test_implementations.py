import json
import os
import shutil
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.json as pyarrow_json
import pyarrow.csv as pyarrow_csv
import pyarrow.orc as pyarrow_orc
import pyarrow.parquet as pyarrow_parquet
import pytest

from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    FileSourcePyArrowTransformer,
)
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader


def _read_feather_ipc(source: Any, columns: list[str] | None = None) -> Any:
    """Non-deprecated Feather V2 reader used as the test comparison oracle."""
    with pa.ipc.open_file(source) as reader:
        table = reader.read_all()
    return table.select(columns) if columns is not None else table


class FeatureSet:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def get_all_names(self) -> list[str]:
        return self.columns


class TestReadFilesImplementations:
    table: Any = None
    columns: Any = None
    feather_file: Any = None
    json_file: Any = None
    csv_file: Any = None
    orc_file: Any = None
    parquet_file: Any = None
    base_path: Any = None

    @classmethod
    def setup_class(cls) -> None:
        cls.base_path = tempfile.mkdtemp(prefix="arrow_test_")

        # Create a sample dataset
        cls.table = pa.Table.from_pydict({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        cls.columns = ["col1", "col2"]

        # Save datasets in different formats
        cls.feather_file = "test.feather"
        with pa.OSFile(cls.feather_file, "wb") as sink:
            with pa.ipc.new_file(sink, cls.table.schema) as writer:
                writer.write_table(cls.table)

        cls.json_file = "test.json"
        with open(cls.json_file, "w") as f:
            json_data = [
                {cls.columns[i]: value.as_py() for i, value in enumerate(row)} for row in zip(*cls.table.columns)
            ]
            for item in json_data:
                f.write(json.dumps(item) + "\n")

        cls.csv_file = f"{cls.base_path}/test.csv"
        pyarrow_csv.write_csv(cls.table, cls.csv_file)

        cls.orc_file = "test.orc"
        with pa.OSFile(cls.orc_file, "wb") as f:
            pyarrow_orc.write_table(cls.table, f)

        cls.parquet_file = "test.parquet"
        pyarrow_parquet.write_table(cls.table, cls.parquet_file)

    def teardown_method(self) -> None:
        shutil.rmtree(self.base_path, ignore_errors=True)

    def assert_read_file(self, cls: Any, path: Any, reader: Any, features: FeatureSet) -> None:
        result = cls.load_data(path, features)

        # CsvReader resolves a FileSource descriptor; materialize it into a pa.Table.
        if cls is CsvReader:
            result = FileSourcePyArrowTransformer.transform_fw_to_other_fw(result)

        if cls in (CsvReader, JsonReader):
            expected = reader(path).select(self.columns)
        else:
            expected = reader(source=path, columns=self.columns)

        result_data = result.to_pydict()
        expected_data = expected.to_pydict()

        assert result_data == expected_data

    def assert_check_files(self, cls: Any, path: Any, reader: Any, features: FeatureSet) -> None:
        if cls != CsvReader:  # only implemented for csv for now
            assert cls.check_files("col1", {path}) == set()
            return

        assert cls.check_files("col1", {path}) == {"path/test.csv"}
        assert cls.check_files("some_wrong", {path}) == set()
        assert cls.check_files("col2", {path[:-2]}) == set()

    def assert_check_folders(self, cls: Any, path: Any, reader: Any, features: FeatureSet) -> None:
        if cls != CsvReader:  # only implemented for csv for now
            assert cls.check_folders("col1", {self.base_path}) == set()
            return

        assert cls.check_folders("col1", {self.base_path}) == {"path/test.csv"}
        assert cls.check_folders("some_wrong", {self.base_path}) == set()
        with pytest.raises(FileNotFoundError):
            assert cls.check_folders("col2", {"wrong_folder"}) == set()

    def test_implementations_read(self) -> None:
        test_cases = [
            (FeatherReader, self.feather_file, _read_feather_ipc),
            (JsonReader, self.json_file, pyarrow_json.read_json),
            (CsvReader, self.csv_file, pyarrow_csv.read_csv),
            (OrcReader, self.orc_file, pyarrow_orc.read_table),
            (ParquetReader, self.parquet_file, pyarrow_parquet.read_table),
        ]

        features = FeatureSet(self.columns)

        try:
            for cls, path, reader in test_cases:
                self.assert_read_file(cls, path, reader, features)
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        finally:
            pass
