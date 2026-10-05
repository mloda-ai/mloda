"""TextFG and PyFG: the shared document contract, framework loads, and text content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mloda.user import DataAccessCollection, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    DocumentLoadsIntoFrameworkMixin,
    PandasDataFrameAdapter,
    PolarsDataFrameAdapter,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.document_format_feature_group_test_mixin import (
    DocumentFormatFeatureGroupTestMixin,
)
from tests.mixins.reader_feature_groups.lazy_format_group import lazy_document_group, load_document_group


class TestTextFG(DocumentFormatFeatureGroupTestMixin):
    feature_group_class = lazy_document_group("text_fg", "TextFG")
    present_column = "TextFG"
    missing_column = "docfmt_text_missing"
    expected_suffixes = frozenset({".text", ".txt", ".TXT"})
    sample_text = "plain txt body\n"


class TestPyFG(DocumentFormatFeatureGroupTestMixin):
    feature_group_class = lazy_document_group("text_fg", "PyFG")
    present_column = "PyFG"
    missing_column = "docfmt_py_missing"
    expected_suffixes = frozenset({".py"})
    sample_text = 'print("hello")\n'


class TestTextLoadsIntoPythonDict(PythonDictAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("text_fg", "TextFG")
    suffix = ".TXT"

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column])


class TestTextLoadsIntoPyArrowTable(PyArrowTableAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("text_fg", "TextFG")
    suffix = ".txt"

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result.column(column).to_pylist())


class TestTextLoadsIntoPandas(PandasDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("text_fg", "TextFG")
    suffix = ".txt"

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].tolist())


class TestTextLoadsIntoPolars(PolarsDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("text_fg", "TextFG")
    suffix = ".txt"

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].to_list())


class TestPyLoadsIntoPythonDict(PythonDictAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("text_fg", "PyFG")
    suffix = ".py"

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column])


class TestPyLoadsIntoPyArrowTable(PyArrowTableAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("text_fg", "PyFG")
    suffix = ".py"

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result.column(column).to_pylist())


class TestTextContent:
    def _run(self, group: str, path: Path, names: list[str]) -> Any:
        cls = load_document_group("text_fg", group)
        return mloda.run_all(
            [Feature(name, options=Options({group: str(path)})) for name in names],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({cls}),
        )

    def test_a_txt_file_runs_end_to_end_with_the_three_outputs(self, tmp_path: Path) -> None:
        file_path = tmp_path / "note.txt"
        file_path.write_text("plain txt body\n", encoding="utf-8")

        result = self._run("TextFG", file_path, ["TextFG", "TextFG~source", "TextFG~file_type"])

        assert result == [
            {"TextFG": ["plain txt body\n"], "TextFG~source": [str(file_path)], "TextFG~file_type": ["txt"]}
        ]

    def test_non_ascii_utf8_text_is_read_unchanged(self, tmp_path: Path) -> None:
        file_path = tmp_path / "unicode.txt"
        file_path.write_text("héllo über 世界\n", encoding="utf-8")

        assert self._run("TextFG", file_path, ["TextFG"]) == [{"TextFG": ["héllo über 世界\n"]}]

    def test_a_py_file_is_read_as_raw_source_with_the_py_file_type(self, tmp_path: Path) -> None:
        file_path = tmp_path / "module.py"
        file_path.write_text("print('hi')\n", encoding="utf-8")

        result = self._run("PyFG", file_path, ["PyFG", "PyFG~source", "PyFG~file_type"])

        assert result == [{"PyFG": ["print('hi')\n"], "PyFG~source": [str(file_path)], "PyFG~file_type": ["py"]}]

    def test_the_text_and_py_groups_each_resolve_only_their_own_file_in_one_folder(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("a note", encoding="utf-8")
        (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
        text_group = load_document_group("text_fg", "TextFG")
        py_group = load_document_group("text_fg", "PyFG")

        result = mloda.run_all(
            ["TextFG", "PyFG"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({text_group, py_group}),
            data_access_collection=DataAccessCollection(folders={"docs_dir": str(tmp_path)}),
        )

        assert len(result) == 2
        assert {str(name): values for table in result for name, values in table.items()} == {
            "TextFG": ["a note"],
            "PyFG": ["x = 1\n"],
        }
