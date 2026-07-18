import json
import shutil
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.orc as pyarrow_orc
import pyarrow.parquet as pyarrow_parquet

from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader


def _loaded_column_names(result: Any) -> list[str]:
    """CsvReader resolves a FileSource descriptor whose ``columns`` pins the order;
    the other readers still return a materialized table with ``column_names``."""
    if isinstance(result, FileSource):
        return list(result.columns)
    return list(result.column_names)


class FeatureSetStub:
    """Mirrors FeatureSet.get_all_names() which returns an alphabetically sorted tuple."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def get_all_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.columns))


class TestDeterministicColumnOrder:
    """The five read-files plugins must materialize get_all_names() (a sorted tuple) in a
    deterministic, alphabetically sorted order so the output column order does not
    depend on PYTHONHASHSEED."""

    base_path: Any = None
    physical_columns: Any = None
    sorted_columns: Any = None
    feather_file: Any = None
    json_file: Any = None
    csv_file: Any = None
    orc_file: Any = None
    parquet_file: Any = None

    @classmethod
    def setup_class(cls) -> None:
        cls.base_path = tempfile.mkdtemp(prefix="arrow_order_test_")

        # Non-alphabetical physical column order to expose ordering bugs.
        cls.physical_columns = ["c1", "a1", "b1"]
        cls.sorted_columns = sorted(cls.physical_columns)  # ["a1", "b1", "c1"]

        table = pa.Table.from_pydict({"c1": [1, 2, 3], "a1": [4, 5, 6], "b1": [7, 8, 9]})

        cls.feather_file = f"{cls.base_path}/test.feather"
        with pa.OSFile(cls.feather_file, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

        cls.json_file = f"{cls.base_path}/test.json"
        with open(cls.json_file, "w") as f:
            rows = [{col: table.column(col)[i].as_py() for col in cls.physical_columns} for i in range(table.num_rows)]
            for row in rows:
                f.write(json.dumps(row) + "\n")

        cls.csv_file = f"{cls.base_path}/test.csv"
        import pyarrow.csv as pyarrow_csv

        pyarrow_csv.write_csv(table, cls.csv_file)

        cls.orc_file = f"{cls.base_path}/test.orc"
        with pa.OSFile(cls.orc_file, "wb") as f:
            pyarrow_orc.write_table(table, f)

        cls.parquet_file = f"{cls.base_path}/test.parquet"
        pyarrow_parquet.write_table(table, cls.parquet_file)

    @classmethod
    def teardown_class(cls) -> None:
        shutil.rmtree(cls.base_path, ignore_errors=True)

    def test_column_order_is_sorted(self) -> None:
        test_cases: list[tuple[Any, Any]] = [
            (FeatherReader, self.feather_file),
            (JsonReader, self.json_file),
            (CsvReader, self.csv_file),
            (OrcReader, self.orc_file),
            (ParquetReader, self.parquet_file),
        ]

        features = FeatureSetStub(self.physical_columns)

        for cls, path in test_cases:
            columns = _loaded_column_names(cls.load_data(path, features))
            assert columns == self.sorted_columns, (
                f"{cls.__name__} returned columns {columns}, expected {self.sorted_columns}"
            )

    def _all_readers(self) -> list[tuple[Any, Any]]:
        return [
            (FeatherReader, self.feather_file),
            (JsonReader, self.json_file),
            (CsvReader, self.csv_file),
            (OrcReader, self.orc_file),
            (ParquetReader, self.parquet_file),
        ]

    def test_empty_feature_set_returns_zero_columns(self) -> None:
        """An empty get_all_names() (empty tuple) must yield a zero-column table
        without raising. Regression lock for the already-sorted readers."""
        features = FeatureSetStub([])

        for cls, path in self._all_readers():
            columns = _loaded_column_names(cls.load_data(path, features))
            assert columns == [], f"{cls.__name__} returned columns {columns}, expected [] for empty feature set"

    def test_single_column_feature_set_returns_that_column(self) -> None:
        """A single-column feature set must return exactly that one column.
        Regression lock for the already-sorted readers."""
        features = FeatureSetStub(["a1"])

        for cls, path in self._all_readers():
            columns = _loaded_column_names(cls.load_data(path, features))
            assert columns == ["a1"], f"{cls.__name__} returned columns {columns}, expected ['a1']"
