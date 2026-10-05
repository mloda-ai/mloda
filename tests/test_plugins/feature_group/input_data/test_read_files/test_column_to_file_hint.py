import os
from pathlib import Path
from typing import Any

import pytest

from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.user import DataAccessCollection, Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.mixins.reader_feature_groups.format_file_writers import write_csv
from tests.mixins.reader_feature_groups.lazy_format_group import load_document_group, load_group


def _csv_group() -> Any:
    return load_group("csv_fg", "CsvFG")


def _csv(directory: Path, name: str, columns: dict[str, list[Any]]) -> str:
    path = directory / f"{name}.csv"
    write_csv(path, columns)
    return str(path)


class TestColumnToFileHint:
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


class TestColumnToFileHintTextFG:
    def test_a_pin_on_the_text_output_loads_the_pinned_file_among_two(self, tmp_path: Path) -> None:
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("first body", encoding="utf-8")
        second.write_text("second body", encoding="utf-8")
        dac = DataAccessCollection(
            files={"first": str(first), "second": str(second)},
            column_to_file={"TextFG": "second", "TextFG~source": "second"},
        )

        result = mloda.run_all(
            ["TextFG", "TextFG~source"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({load_document_group("text_fg", "TextFG")}),
            data_access_collection=dac,
        )

        assert result == [{"TextFG": ["second body"], "TextFG~source": [str(second)]}]
