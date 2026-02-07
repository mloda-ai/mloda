"""Tests for MarkdownDocumentReader - reads .md files returning PythonDict list format."""

import os
import tempfile
from typing import Any

import pytest

from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class MockFeatureSet:
    """Mock FeatureSet for testing document readers."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def get_options_key(self, key: str) -> Any:
        return self._options.get(key)


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


class TestMarkdownDocumentReaderInheritance:
    """Tests that MarkdownDocumentReader inherits from ReadDocument, not ReadFile."""

    def test_markdown_document_reader_inherits_from_read_document(self) -> None:
        """MarkdownDocumentReader must be a subclass of ReadDocument."""
        assert issubclass(MarkdownDocumentReader, ReadDocument)

    def test_markdown_document_reader_not_inherits_from_read_file(self) -> None:
        """MarkdownDocumentReader must NOT be a subclass of ReadFile."""
        assert not issubclass(MarkdownDocumentReader, ReadFile)


class TestMarkdownDocumentReaderLoadData:
    """Tests that load_data returns PythonDict list format with structured metadata."""

    def test_load_markdown_as_python_dict(self) -> None:
        """Load a markdown file and verify result is a list of dicts with content, source, and file_type."""
        fd, temp_path = tempfile.mkstemp(suffix=".md")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(SAMPLE_MARKDOWN)

            features = MockFeatureSet({"MarkdownDocumentReader": temp_path})
            result = MarkdownDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert isinstance(row, dict)
            assert "MarkdownDocumentReader" in row
            assert "source" in row
            assert "file_type" in row
            assert row["source"] == temp_path
            assert row["file_type"] == "md"
            assert row["MarkdownDocumentReader"] == SAMPLE_MARKDOWN
        finally:
            os.unlink(temp_path)

    def test_raw_markdown_formatting_is_preserved(self) -> None:
        """Verify that raw markdown formatting (headers, lists, code blocks) is preserved without conversion."""
        fd, temp_path = tempfile.mkstemp(suffix=".md")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(SAMPLE_MARKDOWN)

            features = MockFeatureSet({"MarkdownDocumentReader": temp_path})
            result = MarkdownDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            content = result[0]["MarkdownDocumentReader"]
            assert "# Heading 1" in content
            assert "## Heading 2" in content
            assert "- item one" in content
            assert "```python" in content
            assert "> A blockquote line." in content
        finally:
            os.unlink(temp_path)

    def test_structured_metadata_replaces_add_path_to_content(self) -> None:
        """Verify source and file_type metadata fields exist and AddPathToContent string prefix is gone."""
        fd, temp_path = tempfile.mkstemp(suffix=".md")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("# Simple doc\n")

            features = MockFeatureSet({"MarkdownDocumentReader": temp_path, "AddPathToContent": True})
            result = MarkdownDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert "source" in row
            assert "file_type" in row
            assert row["source"] == temp_path
            assert row["file_type"] == "md"
            content = row["MarkdownDocumentReader"]
            assert "The file path of the following file is" not in content
        finally:
            os.unlink(temp_path)


class TestMarkdownDocumentReaderClassMethods:
    """Tests for suffix and match_subclass_data_access."""

    def test_suffix_returns_md(self) -> None:
        """suffix() must return a tuple containing '.md'."""
        suffixes = MarkdownDocumentReader.suffix()
        assert ".md" in suffixes

    def test_match_subclass_data_access_returns_path_for_string(self) -> None:
        """match_subclass_data_access returns path for explicit string (feature scope)."""
        result = MarkdownDocumentReader.match_subclass_data_access("readme.md", ["feature1"])
        assert result == "readme.md"

    def test_match_subclass_data_access_returns_none_for_data_access_collection(self) -> None:
        """match_subclass_data_access returns None for DataAccessCollection (no auto-discovery)."""
        from mloda.user import DataAccessCollection

        dac = DataAccessCollection(files={"readme.md"})
        result = MarkdownDocumentReader.match_subclass_data_access(dac, ["feature1"])
        assert result is None
