"""Tests for JsonDocumentReader - migrated to ReadDocument returning PythonDict list format."""

import json
import os
import tempfile
from typing import Any

import pytest

from mloda_plugins.feature_group.input_data.read_files.json_document_reader import JsonDocumentReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class MockFeatureSet:
    """Mock FeatureSet for testing document readers."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def get_options_key(self, key: str) -> Any:
        return self._options.get(key)


class TestJsonDocumentReaderInheritance:
    """Tests that JsonDocumentReader inherits from ReadDocument, not ReadFile."""

    def test_json_document_reader_inherits_from_read_document(self) -> None:
        """JsonDocumentReader must be a subclass of ReadDocument."""
        assert issubclass(JsonDocumentReader, ReadDocument)

    def test_json_document_reader_not_inherits_from_read_file(self) -> None:
        """JsonDocumentReader must NOT be a subclass of ReadFile after migration."""
        assert not issubclass(JsonDocumentReader, ReadFile)


class TestJsonDocumentReaderLoadData:
    """Tests that load_data returns PythonDict list format with structured metadata."""

    def test_load_simple_json_as_python_dict(self) -> None:
        """Load a simple JSON file and verify result is a list of dicts with content, source, and file_type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            simple_data = {"name": "test", "value": 42}
            json.dump(simple_data, f)
            temp_path = f.name

        try:
            features = MockFeatureSet({"JsonDocumentReader": temp_path})
            result = JsonDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert isinstance(row, dict)
            assert "JsonDocumentReader" in row
            assert "source" in row
            assert "file_type" in row
            assert row["source"] == temp_path
            assert row["file_type"] == "json"
            parsed = json.loads(row["JsonDocumentReader"])
            assert parsed == simple_data
        finally:
            os.unlink(temp_path)

    def test_load_nested_json_preserves_structure(self) -> None:
        """Load nested JSON and verify full structure is preserved through JSON round-trip."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            nested_data = {
                "level1": {"level2": {"level3": ["a", "b", "c"]}, "items": [1, 2, 3]},
                "metadata": {"created": "2024-01-01", "tags": ["tag1", "tag2"]},
            }
            json.dump(nested_data, f)
            temp_path = f.name

        try:
            features = MockFeatureSet({"JsonDocumentReader": temp_path})
            result = JsonDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            parsed = json.loads(row["JsonDocumentReader"])
            assert parsed == nested_data
            assert parsed["level1"]["level2"]["level3"] == ["a", "b", "c"]
            assert parsed["metadata"]["tags"] == ["tag1", "tag2"]
        finally:
            os.unlink(temp_path)

    def test_structured_metadata_replaces_add_path_to_content(self) -> None:
        """Verify source and file_type metadata fields exist and AddPathToContent string prefix is gone."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {"key": "value"}
            json.dump(data, f)
            temp_path = f.name

        try:
            features = MockFeatureSet({"JsonDocumentReader": temp_path, "AddPathToContent": True})
            result = JsonDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert "source" in row
            assert "file_type" in row
            assert row["source"] == temp_path
            assert row["file_type"] == "json"
            content = row["JsonDocumentReader"]
            assert "The file path of the following file is" not in content
        finally:
            os.unlink(temp_path)


class TestJsonDocumentReaderClassMethods:
    """Tests for suffix and match_subclass_data_access."""

    def test_suffix_includes_both_cases(self) -> None:
        """suffix() must return both .json and .JSON."""
        suffixes = JsonDocumentReader.suffix()
        assert ".json" in suffixes
        assert ".JSON" in suffixes

    def test_match_subclass_data_access_returns_path_for_string(self) -> None:
        """match_subclass_data_access returns path for explicit string (feature scope)."""
        result = JsonDocumentReader.match_subclass_data_access("some_path.json", ["feature1"])
        assert result == "some_path.json"

    def test_match_subclass_data_access_returns_none_for_data_access_collection(self) -> None:
        """match_subclass_data_access returns None for DataAccessCollection (no auto-discovery)."""
        from mloda.user import DataAccessCollection

        dac = DataAccessCollection(files={"some_path.json"})
        result = JsonDocumentReader.match_subclass_data_access(dac, ["feature1"])
        assert result is None
