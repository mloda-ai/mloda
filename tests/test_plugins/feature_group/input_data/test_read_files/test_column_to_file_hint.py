import csv
import os
import tempfile
from typing import Any

from mloda_plugins.feature_group.input_data.read_document import ReadDocument

import pytest

from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


class TestColumnToFileHint:
    def test_pins_correct_file(self) -> None:
        class TestRF(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["id", "val"]

            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

        dac = DataAccessCollection(files={"a.csv", "b.csv"}, column_to_file={"id": "a.csv", "val": "a.csv"})
        result = TestRF.match_subclass_data_access(dac, ["id", "val"], options=Options({}))
        assert result == "a.csv"

    def test_unpinned_feature_falls_through(self) -> None:
        class TestRF(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["id", "val", "other_col"]

            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

        dac = DataAccessCollection(files={"a.csv", "b.csv"}, column_to_file={"id": "a.csv"})
        # "other_col" not in map, so falls through to normal resolution
        result = TestRF.match_subclass_data_access(dac, ["other_col"], options=Options({}))
        assert result is not None

    def test_no_hint_preserves_behavior(self) -> None:
        class TestRF(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["id", "val"]

            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

        dac = DataAccessCollection(files={"a.csv"})
        result = TestRF.match_subclass_data_access(dac, ["id"], options=Options({}))
        assert result == "a.csv"

    def test_conflict_in_batch_raises(self) -> None:
        class TestRF(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["id", "val"]

            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

        dac = DataAccessCollection(
            files={"a.csv", "b.csv"},
            column_to_file={"id": "a.csv", "val": "b.csv"},
        )
        with pytest.raises(ValueError):
            TestRF.match_subclass_data_access(dac, ["id", "val"], options=Options({}))

    def test_mixed_batch_raises(self) -> None:
        class TestRF(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["id", "unpinned_col"]

            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

        dac = DataAccessCollection(
            files={"a.csv", "b.csv"},
            column_to_file={"id": "a.csv"},
        )
        with pytest.raises(ValueError):
            TestRF.match_subclass_data_access(dac, ["id", "unpinned_col"], options=Options({}))

    def test_wrong_suffix_falls_through(self) -> None:
        class TestRFCsv(ReadFile):
            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["id", "val"]

            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

        # column_to_file points to parquet files, but this reader only handles .csv
        dac = DataAccessCollection(
            files={"a.parquet", "b.parquet"},
            column_to_file={"id": "a.parquet", "val": "a.parquet"},
        )
        result = TestRFCsv.match_subclass_data_access(dac, ["id", "val"], options=Options({}))
        assert result is None

    def test_construction_rejects_unknown_file(self) -> None:
        with pytest.raises(ValueError):
            DataAccessCollection(files={"a.csv"}, column_to_file={"col": "b.csv"})

    def test_integration_two_csvs_sharing_id_column(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f1:
            train_path = f1.name
            writer = csv.writer(f1)
            writer.writerow(["id", "target"])
            writer.writerow([1, 0])
            writer.writerow([2, 1])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f2:
            bureau_path = f2.name
            writer = csv.writer(f2)
            writer.writerow(["id", "amount"])
            writer.writerow([1, 500])
            writer.writerow([2, 300])

        try:
            dac = DataAccessCollection(
                files={train_path, bureau_path},
                column_to_file={
                    "id": train_path,
                    "target": train_path,
                    "amount": bureau_path,
                },
            )

            result_train = CsvReader.match_subclass_data_access(dac, ["id", "target"], options=Options({}))
            assert result_train == train_path

            result_bureau = CsvReader.match_subclass_data_access(dac, ["id", "amount"], options=Options({}))
            assert result_bureau == bureau_path
        finally:
            os.remove(train_path)
            os.remove(bureau_path)


class TestColumnToFileHintReadDocument:
    def test_document_pins_correct_file(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(
            files={"a.txt", "b.txt"},
            column_to_file={"feature_a": "a.txt"},
        )
        result = TestRD.match_subclass_data_access(dac, ["feature_a"], options=Options({}))
        assert result == "a.txt"

    def test_document_no_hint_falls_through(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(files={"a.txt", "b.txt"})
        result = TestRD.match_subclass_data_access(dac, ["any_feature"], options=Options({}))
        assert result is not None
        assert result.endswith(".txt")

    def test_document_mixed_batch_raises(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(
            files={"a.txt", "b.txt"},
            column_to_file={"feature_a": "a.txt"},
        )
        with pytest.raises(ValueError):
            TestRD.match_subclass_data_access(dac, ["feature_a", "unpinned"], options=Options({}))

    def test_document_folder_traversal(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        tmp_dir = tempfile.mkdtemp()
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", dir=tmp_dir, delete=False) as f:
                _ = f.name
            dac = DataAccessCollection(folders={tmp_dir})
            result = TestRD.match_subclass_data_access(dac, ["any_feature"], options=Options({}))
            assert result is not None
            assert result.endswith(".txt")
        finally:
            import shutil

            shutil.rmtree(tmp_dir)

    def test_document_str_path_suffix_check(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        result = TestRD.match_subclass_data_access("file.csv", ["any_feature"], options=Options({}))
        assert result is None

    def test_document_str_path_correct_suffix(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        result = TestRD.match_subclass_data_access("file.txt", ["any_feature"], options=Options({}))
        assert result == "file.txt"
