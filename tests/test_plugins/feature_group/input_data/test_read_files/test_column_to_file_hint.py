import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass, resolve_or_raise
from mloda.user import DataAccessCollection, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from tests.helpers.suffix_file_reader import SuffixFileReader
from tests.mixins.reader_feature_groups.format_file_writers import write_csv
from tests.mixins.reader_feature_groups.lazy_format_group import load_group


def _csv_group() -> Any:
    return load_group("csv_fg", "CsvFG")


def _csv(directory: Path, name: str, columns: dict[str, list[Any]]) -> str:
    path = directory / f"{name}.csv"
    write_csv(path, columns)
    return str(path)


def _claim(feature: Feature, dac: DataAccessCollection) -> bool:
    group = _csv_group()
    return group in IdentifyFeatureGroupClass.evaluate(feature, {group: {PyArrowTable}}, None, dac).identified


def _source(feature: Feature) -> str:
    pair = feature.input_data_match
    assert pair is not None
    return cast(SourceMatch, pair[1]).source


class TestColumnToFileHint:
    def test_pins_correct_file(self, tmp_path: Path) -> None:
        a = _csv(tmp_path, "pin_a", {"cfh_pin_id": [1], "cfh_pin_val": [2]})
        b = _csv(tmp_path, "pin_b", {"cfh_pin_id": [3], "cfh_pin_val": [4]})
        dac = DataAccessCollection(files={"a": a, "b": b}, column_to_file={"cfh_pin_id": "a", "cfh_pin_val": "a"})

        for name in ("cfh_pin_id", "cfh_pin_val"):
            feature = Feature(name)
            assert _claim(feature, dac)
            assert _source(feature) == os.path.abspath(a)

    def test_unpinned_feature_aborts_on_ambiguity(self, tmp_path: Path) -> None:
        columns = {"cfh_amb_id": [1], "cfh_amb_other": [2]}
        a = _csv(tmp_path, "amb_a", columns)
        b = _csv(tmp_path, "amb_b", columns)
        dac = DataAccessCollection(files={"a": a, "b": b}, column_to_file={"cfh_amb_id": "a"})
        group = _csv_group()

        with pytest.raises(ValueError) as excinfo:
            resolve_or_raise(Feature("cfh_amb_other"), {group: {PyArrowTable}}, None, dac)

        assert "data_access_handle" in str(excinfo.value)
        assert os.path.abspath(a) in str(excinfo.value)
        assert os.path.abspath(b) in str(excinfo.value)

    def test_unpinned_feature_resolves_with_hint(self, tmp_path: Path) -> None:
        columns = {"cfh_hint_id": [1], "cfh_hint_other": [2]}
        a = _csv(tmp_path, "hint_a", columns)
        b = _csv(tmp_path, "hint_b", columns)
        dac = DataAccessCollection(files={"a": a, "b": b}, column_to_file={"cfh_hint_id": "a"})

        feature = Feature("cfh_hint_other", Options(context={"data_access_handle": "b"}))

        assert _claim(feature, dac)
        assert _source(feature) == os.path.abspath(b)

    def test_no_hint_preserves_behavior(self, tmp_path: Path) -> None:
        a = _csv(tmp_path, "nohint_a", {"cfh_nohint_id": [1], "cfh_nohint_val": [2]})
        feature = Feature("cfh_nohint_id")

        assert _claim(feature, DataAccessCollection(files={a}))
        assert _source(feature) == os.path.abspath(a)

    def test_wrong_suffix_declines_without_falling_back_to_an_unpinned_file(self, tmp_path: Path) -> None:
        # The pin points to a parquet file CsvFG can't serve; an unpinned .csv match must not be a fallback.
        csv_file = _csv(tmp_path, "fallback_b", {"cfh_wrong_id": [1], "cfh_wrong_val": [2]})
        dac = DataAccessCollection(
            files={"a.parquet", csv_file},
            column_to_file={"cfh_wrong_id": "a.parquet", "cfh_wrong_val": "a.parquet"},
        )

        assert not _claim(Feature("cfh_wrong_id"), dac)
        assert not _claim(Feature("cfh_wrong_val"), dac)

    def test_pin_wins_over_a_data_access_handle_hint_pointing_elsewhere(self, tmp_path: Path) -> None:
        # The pin (wrong suffix) beats a data_access_handle hint pointing at an otherwise-valid file.
        csv_file = _csv(tmp_path, "winner_b", {"cfh_win_id": [1]})
        dac = DataAccessCollection(files={"a": "a.parquet", "b": csv_file}, column_to_file={"cfh_win_id": "a"})

        feature = Feature("cfh_win_id", Options(context={"data_access_handle": "b"}))

        assert not _claim(feature, dac)

    def test_pinned_chain_shaped_name_resolves_while_an_unpinned_one_declines(self, tmp_path: Path) -> None:
        pinned_name = "cfh_price__scaled"
        path = _csv(tmp_path, "chain_shaped", {pinned_name: [1], "cfh_plain": [2]})
        dac = DataAccessCollection(files={"chain_handle": path}, column_to_file={pinned_name: "chain_handle"})

        pinned = Feature(pinned_name)
        assert _claim(pinned, dac)
        assert _source(pinned) == os.path.abspath(path)
        assert not _claim(Feature("cfh_other__scaled"), dac)

    def test_conflict_in_batch_raises(self) -> None:
        class TestRF(SuffixFileReader):
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
        with pytest.raises(ValueError) as excinfo:
            TestRF.match_subclass_data_access(dac, ["id", "val"], options=Options({}))
        assert "pinned to different files" in str(excinfo.value)

    def test_mixed_batch_raises(self) -> None:
        class TestRF(SuffixFileReader):
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
        with pytest.raises(ValueError) as excinfo:
            TestRF.match_subclass_data_access(dac, ["id", "unpinned_col"], options=Options({}))
        assert "Mixed batch" in str(excinfo.value)

    def test_construction_rejects_unknown_file(self) -> None:
        with pytest.raises(ValueError):
            DataAccessCollection(files={"a.csv"}, column_to_file={"col": "b"})

    def test_integration_two_csvs_sharing_id_column(self, tmp_path: Path) -> None:
        train_path = _csv(tmp_path, "train", {"cfh_int_id": [1, 2], "cfh_int_target": [0, 1]})
        bureau_path = _csv(tmp_path, "bureau", {"cfh_int_id": [1, 2], "cfh_int_amount": [500, 300]})
        dac = DataAccessCollection(
            files={train_path, bureau_path},
            column_to_file={"cfh_int_id": train_path, "cfh_int_target": train_path, "cfh_int_amount": bureau_path},
        )

        result = mloda.run_all(
            ["cfh_int_id", "cfh_int_target", "cfh_int_amount"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({_csv_group()}),
            data_access_collection=dac,
        )

        assert len(result) == 2
        by_columns = {frozenset(table.column_names): table for table in result}
        assert set(by_columns) == {frozenset({"cfh_int_id", "cfh_int_target"}), frozenset({"cfh_int_amount"})}
        assert by_columns[frozenset({"cfh_int_amount"})].to_pydict() == {"cfh_int_amount": [500, 300]}


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

    def test_document_no_hint_raises_on_ambiguity(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(files={"a.txt", "b.txt"})
        with pytest.raises(ValueError) as excinfo:
            TestRD.match_subclass_data_access(dac, ["any_feature"], options=Options({}))
        assert "data_access_handle" in str(excinfo.value)

    def test_document_hint_resolves_ambiguity(self) -> None:
        class TestRD(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".txt",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(files={"a": "a.txt", "b": "b.txt"})
        result = TestRD.match_subclass_data_access(
            dac, ["any_feature"], options=Options(context={"data_access_handle": "a"})
        )
        assert result == "a.txt"

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
