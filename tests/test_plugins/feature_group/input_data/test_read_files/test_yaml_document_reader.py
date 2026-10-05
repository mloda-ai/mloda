"""YamlFG: the shared document contract, framework loads, and YAML content (nested, multi-document)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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

FLOW_STYLE = "{b: two, a: 1}\n"
CANONICAL = yaml.dump({"a": 1, "b": "two"})


class TestYamlFG(DocumentFormatFeatureGroupTestMixin):
    feature_group_class = lazy_document_group("yaml_fg", "YamlFG")
    present_column = "YamlFG"
    missing_column = "docfmt_yaml_missing"
    expected_suffixes = frozenset({".yaml", ".yml"})
    sample_text = FLOW_STYLE
    expected_content = CANONICAL


class TestYamlLoadsIntoPythonDict(PythonDictAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("yaml_fg", "YamlFG")
    suffix = ".yml"
    text = FLOW_STYLE
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column])


class TestYamlLoadsIntoPyArrowTable(PyArrowTableAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("yaml_fg", "YamlFG")
    suffix = ".yaml"
    text = FLOW_STYLE
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result.column(column).to_pylist())


class TestYamlLoadsIntoPandas(PandasDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("yaml_fg", "YamlFG")
    suffix = ".yaml"
    text = FLOW_STYLE
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].tolist())


class TestYamlLoadsIntoPolars(PolarsDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("yaml_fg", "YamlFG")
    suffix = ".yaml"
    text = FLOW_STYLE
    expected_text = CANONICAL

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].to_list())


class TestYamlContent:
    def _content(self, path: Path) -> str:
        result = mloda.run_all(
            [Feature("YamlFG", Options({"YamlFG": str(path)}))],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({load_document_group("yaml_fg", "YamlFG")}),
        )
        content: str = result[0]["YamlFG"][0]
        return content

    def test_nested_yaml_round_trips(self, tmp_path: Path) -> None:
        nested = {
            "level1": {"level2": {"level3": ["a", "b", "c"]}, "items": [1, 2, 3]},
            "metadata": {"created": "2024-01-01", "tags": ["tag1", "tag2"]},
        }
        path = tmp_path / "nested.yaml"
        path.write_text(yaml.dump(nested), encoding="utf-8")

        assert yaml.safe_load(self._content(path)) == nested

    def test_a_single_document_is_not_wrapped_in_a_list(self, tmp_path: Path) -> None:
        path = tmp_path / "single.yaml"
        path.write_text("a: 1\nb: two\n", encoding="utf-8")

        assert self._content(path) == yaml.dump({"a": 1, "b": "two"})

    def test_a_multi_document_stream_stays_a_list_of_documents(self, tmp_path: Path) -> None:
        path = tmp_path / "multi.yaml"
        path.write_text("---\nname: document1\nvalue: 1\n---\nname: document2\nvalue: 2\n", encoding="utf-8")

        parsed = yaml.safe_load(self._content(path))

        assert parsed == [{"name": "document1", "value": 1}, {"name": "document2", "value": 2}]
