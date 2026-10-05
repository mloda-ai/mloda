"""JsonDocumentFG: the shared document contract on a handed-over suffix, framework loads, canonical JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mloda.user import Feature, Options, PluginCollector, mloda
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

HANDOVER = {"document_suffixes": frozenset({".json", ".JSON"})}
PRETTY = '{\n  "a": 1,\n  "b": [true, null]\n}\n'
CANONICAL = json.dumps({"a": 1, "b": [True, None]})


class TestJsonDocumentFG(DocumentFormatFeatureGroupTestMixin):
    feature_group_class = lazy_document_group("json_document_fg", "JsonDocumentFG")
    present_column = "JsonDocumentFG"
    missing_column = "docfmt_json_missing"
    expected_suffixes = frozenset({".json", ".JSON"})
    context_options = HANDOVER
    handed_over = True
    sample_text = PRETTY
    expected_content = CANONICAL


class TestJsonDocumentLoadsIntoPythonDict(PythonDictAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("json_document_fg", "JsonDocumentFG")
    suffix = ".json"
    context = HANDOVER
    text = PRETTY
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column])


class TestJsonDocumentLoadsIntoPyArrowTable(PyArrowTableAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("json_document_fg", "JsonDocumentFG")
    suffix = ".json"
    context = HANDOVER
    text = PRETTY
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result.column(column).to_pylist())


class TestJsonDocumentLoadsIntoPandas(PandasDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("json_document_fg", "JsonDocumentFG")
    suffix = ".json"
    context = HANDOVER
    text = PRETTY
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].tolist())


class TestJsonDocumentLoadsIntoPolars(PolarsDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("json_document_fg", "JsonDocumentFG")
    suffix = ".json"
    context = HANDOVER
    text = PRETTY
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].to_list())


class TestJsonDocumentContent:
    def test_nested_json_is_returned_as_one_canonical_document(self, tmp_path: Path) -> None:
        nested = {
            "level1": {"level2": {"level3": ["a", "b", "c"]}, "items": [1, 2, 3]},
            "metadata": {"created": "2024-01-01", "tags": ["tag1", "tag2"]},
        }
        path = tmp_path / "nested.json"
        path.write_text(json.dumps(nested, indent=4), encoding="utf-8")

        result = mloda.run_all(
            [Feature("JsonDocumentFG", Options({"JsonDocumentFG": str(path)}, context=HANDOVER))],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {load_document_group("json_document_fg", "JsonDocumentFG")}
            ),
        )

        assert result == [{"JsonDocumentFG": [json.dumps(nested)]}]
        assert json.loads(result[0]["JsonDocumentFG"][0]) == nested
