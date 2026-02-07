"""Tests for TextFileReader and PyFileReader migration from ReadFile to ReadDocument.

These tests define the target behavior after migration:
- Parent class changes from ReadFile to ReadDocument
- Return format changes from Pandas DataFrame to list of PythonDict
- Metadata uses structured fields (source, file_type) instead of AddPathToContent string injection
"""

import os
import tempfile
from typing import Any

import pytest

from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader, TextFileReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class MockFeatureSet:
    """Mock FeatureSet for testing file readers."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def get_options_key(self, key: str) -> Any:
        return self._options.get(key)


class TestTextFileReaderInheritance:
    """Tests for TextFileReader class hierarchy after migration."""

    def test_text_file_reader_inherits_from_read_document(self) -> None:
        """TextFileReader should inherit from ReadDocument, not ReadFile."""
        assert issubclass(TextFileReader, ReadDocument)

    def test_text_file_reader_not_inherits_from_read_file(self) -> None:
        """TextFileReader should no longer inherit from ReadFile."""
        assert not issubclass(TextFileReader, ReadFile)

    def test_text_file_reader_suffix(self) -> None:
        """TextFileReader suffix should remain ('.text',)."""
        assert TextFileReader.suffix() == (".text",)


class TestTextFileReaderLoadData:
    """Tests for TextFileReader.load_data returning PythonDict list format."""

    def test_text_file_reader_loads_as_python_dict(self) -> None:
        """load_data should return a list of dicts, not a Pandas DataFrame."""
        fd, temp_path = tempfile.mkstemp(suffix=".text")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("hello world")

            features = MockFeatureSet({"TextFileReader": temp_path})
            result = TextFileReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert result[0]["TextFileReader"] == "hello world"
            assert result[0]["source"] == temp_path
            assert result[0]["file_type"] == "text"
        finally:
            os.unlink(temp_path)

    def test_text_file_reader_structured_metadata(self) -> None:
        """load_data should include structured metadata fields, not AddPathToContent string injection."""
        fd, temp_path = tempfile.mkstemp(suffix=".text")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("sample content")

            features = MockFeatureSet({"TextFileReader": temp_path})
            result = TextFileReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            record = result[0]

            assert "source" in record
            assert record["source"] == temp_path

            assert "file_type" in record
            assert record["file_type"] == "text"

            content_value = record["TextFileReader"]
            assert "The file path of the following file is" not in content_value
            assert content_value == "sample content"
        finally:
            os.unlink(temp_path)


class TestPyFileReaderInheritance:
    """Tests for PyFileReader class hierarchy after migration."""

    def test_py_file_reader_inherits_from_text_file_reader(self) -> None:
        """PyFileReader should still inherit from TextFileReader."""
        assert issubclass(PyFileReader, TextFileReader)

    def test_py_file_reader_inherits_from_read_document(self) -> None:
        """PyFileReader should transitively inherit from ReadDocument."""
        assert issubclass(PyFileReader, ReadDocument)

    def test_py_file_reader_suffix(self) -> None:
        """PyFileReader suffix should remain ('.py',)."""
        assert PyFileReader.suffix() == (".py",)


class TestPyFileReaderLoadData:
    """Tests for PyFileReader.load_data returning PythonDict list format."""

    def test_py_file_reader_loads_as_python_dict(self) -> None:
        """load_data should return a list of dicts with PyFileReader as the content key."""
        fd, temp_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write('print("hello")\n')

            features = MockFeatureSet({"PyFileReader": temp_path})
            result = PyFileReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert result[0]["PyFileReader"] == 'print("hello")\n'
            assert result[0]["source"] == temp_path
            assert result[0]["file_type"] == "py"
        finally:
            os.unlink(temp_path)


class TestMatchSubclassDataAccess:
    """Tests for match_subclass_data_access inherited from ReadDocument."""

    def test_match_subclass_data_access_returns_path_for_string(self) -> None:
        """TextFileReader returns path for explicit string (feature scope)."""
        result = TextFileReader.match_subclass_data_access("some_path.text", ["TextFileReader"])
        assert result == "some_path.text"

    def test_match_subclass_data_access_returns_none_for_data_access_collection(self) -> None:
        """TextFileReader returns None for DataAccessCollection (no auto-discovery)."""
        from mloda.user import DataAccessCollection

        dac = DataAccessCollection(files={"some_path.text"})
        result = TextFileReader.match_subclass_data_access(dac, ["TextFileReader"])
        assert result is None
