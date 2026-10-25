"""TextFG and PyFG: the shared document contract, framework loads, and text content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.provider import (
    ComputeFramework,
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    PropertySpec,
)
from mloda.user import DataAccessCollection, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import records_from_frame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    DocumentLoadsIntoFrameworkMixin,
    PandasDataFrameAdapter,
    PolarsDataFrameAdapter,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.read_document_fg_test_mixin import (
    ReadDocumentFGTestMixin,
)
from tests.mixins.reader_feature_groups.lazy_source_group import lazy_document_group, load_document_group


class TestTextFG(ReadDocumentFGTestMixin):
    feature_group_class = lazy_document_group("text_fg", "TextFG")
    present_column = "TextFG"
    missing_column = "docfmt_text_missing"
    expected_suffixes = frozenset({".text", ".txt", ".TXT"})
    sample_text = "plain txt body\n"


class TestPyFG(ReadDocumentFGTestMixin):
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


class UpperChainedFG(FeatureChainParserMixin, FeatureGroup):
    """Test-only chained group: <source>__upper upper-cases the text of its source."""

    PREFIX_PATTERN = r".*__upper$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1
    PROPERTY_MAPPING = {DefaultOptionKeys.in_features: PropertySpec("Source feature", context=True)}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        """Name-only match so config-based matching never claims other tests' features."""
        if not str(feature_name).endswith("__upper"):
            return False
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework, PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        columns = {str(k): list(v) for k, v in data.items()} if isinstance(data, dict) else data.to_dict("list")
        out: dict[str, list[str]] = {}
        for feature in features.features:
            source = cls._extract_source_features(feature)[0]
            out[str(feature.name)] = [text.upper() for text in columns[str(source)]]
        return out if isinstance(data, dict) else pd.DataFrame(out)


def _columns(result: list[Any]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for frame in result:
        for row in records_from_frame(frame):
            for key, value in row.items():
                columns.setdefault(str(key), []).append(value)
    return columns


def _run_collection(
    path: Path, features: list[Feature | str], framework: type[ComputeFramework] = PythonDictFramework
) -> dict[str, list[Any]]:
    result = mloda.run_all(
        features,
        compute_frameworks=[framework],
        plugin_collector=PluginCollector.enabled_feature_groups(
            {load_document_group("text_fg", "TextFG"), UpperChainedFG}
        ),
        data_access_collection=DataAccessCollection(files={"doc": str(path)}),
    )
    return _columns(result)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "note.txt"
    path.write_text(text, encoding="utf-8")
    return path


class TestTextChainedFeature:
    @pytest.mark.parametrize("framework", [PythonDictFramework, PandasDataFrame])
    def test_chained_feature_and_source_column_resolve_together(
        self, tmp_path: Path, framework: type[ComputeFramework]
    ) -> None:
        path = _write(tmp_path, "mixed Case\n")

        columns = _run_collection(path, ["TextFG__upper", "TextFG~source"], framework)

        assert columns == {"TextFG__upper": ["MIXED CASE\n"], "TextFG~source": [str(path)]}


class TestTextLinesOption:
    TEXT = "alpha\n\n  \nbeta\ngamma"

    def test_the_option_is_declared_with_an_explanation_and_default_false(self) -> None:
        spec = load_document_group("text_fg", "TextFG").PROPERTY_MAPPING["text_lines"]

        assert spec.explanation
        assert spec.default is False

    def test_one_row_per_non_blank_line_with_repeated_source_and_file_type(self, tmp_path: Path) -> None:
        path = _write(tmp_path, self.TEXT)
        names = ["TextFG", "TextFG~source", "TextFG~file_type"]

        columns = _run_collection(path, [Feature(n, options=Options({"text_lines": True})) for n in names])

        assert columns == {
            "TextFG": ["alpha", "beta", "gamma"],
            "TextFG~source": [str(path)] * 3,
            "TextFG~file_type": ["txt"] * 3,
        }

    def test_without_the_option_the_whole_text_is_one_row(self, tmp_path: Path) -> None:
        path = _write(tmp_path, self.TEXT)

        assert _run_collection(path, ["TextFG"]) == {"TextFG": [self.TEXT]}

    def test_the_option_set_false_keeps_one_row(self, tmp_path: Path) -> None:
        path = _write(tmp_path, self.TEXT)

        columns = _run_collection(path, [Feature("TextFG", options=Options({"text_lines": False}))])

        assert columns == {"TextFG": [self.TEXT]}

    def test_the_option_reaches_the_source_through_a_chained_feature(self, tmp_path: Path) -> None:
        path = _write(tmp_path, self.TEXT)
        options = Options(group={"text_lines": True})

        columns = _run_collection(path, [Feature("TextFG__upper", options=options)])

        assert columns == {"TextFG__upper": ["ALPHA", "BETA", "GAMMA"]}

    def test_the_chained_feature_next_to_the_source_column_with_the_option(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a\n\nb\n")
        options = Options(group={"text_lines": True})

        columns = _run_collection(
            path, [Feature("TextFG__upper", options=options), Feature("TextFG~source", options=options)]
        )

        assert columns == {"TextFG__upper": ["A", "B"], "TextFG~source": [str(path)] * 2}


class TestTextCountRows:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("one line", 1), ("a\n\n  \nb\nc\n", 3), ("", 1), ("\n \n", 1)],
    )
    def test_count_rows_is_the_non_blank_line_count_with_a_floor_of_one(
        self, tmp_path: Path, text: str, expected: int
    ) -> None:
        path = _write(tmp_path, text)
        group = load_document_group("text_fg", "TextFG")

        count = group.count_rows(SourceMatch(source=str(path), access=str(path)), PythonDictFramework)

        assert count == expected
