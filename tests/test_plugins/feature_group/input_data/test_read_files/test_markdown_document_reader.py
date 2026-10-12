"""MarkdownFG: the shared document contract (raw Markdown is the sample, so it must come back unchanged) and loads."""

from __future__ import annotations

from typing import Any

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
from tests.mixins.reader_feature_groups.lazy_format_group import lazy_document_group

SAMPLE_MARKDOWN = """\
# Heading 1

Some paragraph text.

## Heading 2

- item one
- item two
- item three

```python
def hello():
    print("world")
```

> A blockquote line.
"""


class TestMarkdownFG(DocumentFormatFeatureGroupTestMixin):
    feature_group_class = lazy_document_group("markdown_fg", "MarkdownFG")
    present_column = "MarkdownFG"
    missing_column = "docfmt_markdown_missing"
    expected_suffixes = frozenset({".md"})
    sample_text = SAMPLE_MARKDOWN


class TestMarkdownLoadsIntoPythonDict(PythonDictAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column])


class TestMarkdownLoadsIntoPyArrowTable(PyArrowTableAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result.column(column).to_pylist())


class TestMarkdownLoadsIntoPandas(PandasDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].tolist())


class TestMarkdownLoadsIntoPolars(PolarsDataFrameAdapter, DocumentLoadsIntoFrameworkMixin):
    document_group = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].to_list())
