"""MarkdownFG: the shared document contract (raw Markdown is the sample, so it must come back unchanged) and loads."""

from __future__ import annotations


from mloda._testing.framework_adapter import PandasAdapter, PolarsAdapter, PyArrowAdapter, PythonDictAdapter
from mloda._testing.source.loads_into_framework_suite import DocumentLoadsIntoFrameworkTestMixin
from mloda._testing.source.read_document_suite import (
    ReadDocumentFGTestMixin,
)
from tests.mixins.reader_feature_groups.lazy_source_group import lazy_document_group

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


class TestMarkdownFG(ReadDocumentFGTestMixin):
    feature_group_class = lazy_document_group("markdown_fg", "MarkdownFG")
    present_column = "MarkdownFG"
    missing_column = "docfmt_markdown_missing"
    expected_suffixes = frozenset({".md"})
    sample_text = SAMPLE_MARKDOWN


class TestMarkdownLoadsIntoPythonDict(DocumentLoadsIntoFrameworkTestMixin):
    adapter_class = PythonDictAdapter
    feature_group_class = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN


class TestMarkdownLoadsIntoPyArrowTable(DocumentLoadsIntoFrameworkTestMixin):
    adapter_class = PyArrowAdapter
    feature_group_class = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN


class TestMarkdownLoadsIntoPandas(DocumentLoadsIntoFrameworkTestMixin):
    adapter_class = PandasAdapter
    feature_group_class = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN


class TestMarkdownLoadsIntoPolars(DocumentLoadsIntoFrameworkTestMixin):
    adapter_class = PolarsAdapter
    feature_group_class = lazy_document_group("markdown_fg", "MarkdownFG")
    suffix = ".md"
    text = SAMPLE_MARKDOWN
