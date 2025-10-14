import json
import os
import shutil
import tempfile
from typing import Any, List
import unittest
import pyarrow as pa
import pyarrow.feather as pyarrow_feather
import pyarrow.json as pyarrow_json
import pyarrow.csv as pyarrow_csv
import pyarrow.orc as pyarrow_orc
import pyarrow.parquet as pyarrow_parquet
import pytest

from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader


class FeatureSet:
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def get_all_names(self) -> List[str]:
        return self.columns


class TestReadFilesImplementations(unittest.TestCase):
    table = None
    columns = None
    feather_file = None
    json_file = None
    csv_file = None
    orc_file = None
    parquet_file = None
    base_path = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_path = tempfile.mkdtemp(prefix="arrow_test_")

        # Create a sample dataset
        cls.table = pa.Table.from_pydict({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        cls.columns = ["col1", "col2"]

        # Save datasets in different formats
        cls.feather_file = "test.feather"
        pyarrow_feather.write_feather(cls.table, cls.feather_file)

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

    def tearDown(self) -> None:
        shutil.rmtree(self.base_path, ignore_errors=True)  # type: ignore

    def assert_read_file(self, cls: Any, path: Any, reader: Any, features: FeatureSet) -> None:
        result = cls.load_data(path, features)

        if cls in (CsvReader, JsonReader):
            expected = reader(path).select(self.columns)
        else:
            expected = reader(source=path, columns=self.columns)

        result_data = result.to_pydict()
        expected_data = expected.to_pydict()

        self.assertEqual(result_data, expected_data)

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
            (FeatherReader, self.feather_file, pyarrow_feather.read_table),
            (JsonReader, self.json_file, pyarrow_json.read_json),
            (CsvReader, self.csv_file, pyarrow_csv.read_csv),
            (OrcReader, self.orc_file, pyarrow_orc.read_table),
            (ParquetReader, self.parquet_file, pyarrow_parquet.read_table),
        ]

        features = FeatureSet(self.columns)  # type: ignore

        try:
            for cls, path, reader in test_cases:
                self.assert_read_file(cls, path, reader, features)
                try:
                    os.remove(path)  # type: ignore
                except FileNotFoundError:
                    pass
        finally:
            pass
