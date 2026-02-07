"""Tests for YamlDocumentReader - follows ReadDocument pattern returning PythonDict list format."""

import os
import tempfile
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from mloda_plugins.feature_group.input_data.read_files.yaml_document_reader import YamlDocumentReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class MockFeatureSet:
    """Mock FeatureSet for testing document readers."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def get_options_key(self, key: str) -> Any:
        return self._options.get(key)


class TestYamlDocumentReaderInheritance:
    """Tests that YamlDocumentReader inherits from ReadDocument, not ReadFile."""

    def test_yaml_document_reader_inherits_from_read_document(self) -> None:
        """YamlDocumentReader must be a subclass of ReadDocument."""
        assert issubclass(YamlDocumentReader, ReadDocument)

    def test_yaml_document_reader_not_inherits_from_read_file(self) -> None:
        """YamlDocumentReader must NOT be a subclass of ReadFile."""
        assert not issubclass(YamlDocumentReader, ReadFile)


class TestYamlDocumentReaderLoadData:
    """Tests that load_data returns PythonDict list format with structured metadata."""

    def test_load_simple_yaml_as_python_dict(self) -> None:
        """Load a simple YAML file and verify result is a list of dicts with content, source, and file_type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            simple_data = {"name": "test", "value": 42}
            yaml.dump(simple_data, f)
            temp_path = f.name

        try:
            features = MockFeatureSet({"YamlDocumentReader": temp_path})
            result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert isinstance(row, dict)
            assert "YamlDocumentReader" in row
            assert "source" in row
            assert "file_type" in row
            assert row["source"] == temp_path
            assert row["file_type"] == "yaml"
            parsed = yaml.safe_load(row["YamlDocumentReader"])
            assert parsed == simple_data
        finally:
            os.unlink(temp_path)

    def test_load_nested_yaml_preserves_structure(self) -> None:
        """Load nested YAML and verify full structure is preserved through YAML round-trip."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            nested_data = {
                "level1": {"level2": {"level3": ["a", "b", "c"]}, "items": [1, 2, 3]},
                "metadata": {"created": "2024-01-01", "tags": ["tag1", "tag2"]},
            }
            yaml.dump(nested_data, f)
            temp_path = f.name

        try:
            features = MockFeatureSet({"YamlDocumentReader": temp_path})
            result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            parsed = yaml.safe_load(row["YamlDocumentReader"])
            assert parsed == nested_data
            assert parsed["level1"]["level2"]["level3"] == ["a", "b", "c"]
            assert parsed["metadata"]["tags"] == ["tag1", "tag2"]
        finally:
            os.unlink(temp_path)

    def test_load_multi_document_yaml(self) -> None:
        """Load YAML file with multiple documents separated by --- and verify all are loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("---\n")
            f.write("name: document1\n")
            f.write("value: 1\n")
            f.write("---\n")
            f.write("name: document2\n")
            f.write("value: 2\n")
            temp_path = f.name

        try:
            features = MockFeatureSet({"YamlDocumentReader": temp_path})
            result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]

            # Multi-document YAML should be re-serialized as list of documents
            parsed = yaml.safe_load(row["YamlDocumentReader"])
            assert isinstance(parsed, list)
            assert len(parsed) == 2
            assert parsed[0] == {"name": "document1", "value": 1}
            assert parsed[1] == {"name": "document2", "value": 2}
        finally:
            os.unlink(temp_path)

    def test_yml_extension_has_correct_file_type(self) -> None:
        """Verify .yml file has file_type='yml' and .yaml has file_type='yaml'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump({"key": "value"}, f)
            temp_path_yml = f.name

        try:
            features = MockFeatureSet({"YamlDocumentReader": temp_path_yml})
            result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert row["file_type"] == "yml"
        finally:
            os.unlink(temp_path_yml)

    def test_yaml_extension_has_correct_file_type(self) -> None:
        """Verify .yaml file has file_type='yaml'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value"}, f)
            temp_path_yaml = f.name

        try:
            features = MockFeatureSet({"YamlDocumentReader": temp_path_yaml})
            result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

            assert isinstance(result, list)
            assert len(result) == 1
            row = result[0]
            assert row["file_type"] == "yaml"
        finally:
            os.unlink(temp_path_yaml)


class TestYamlDocumentReaderClassMethods:
    """Tests for suffix and match_subclass_data_access."""

    def test_suffix_includes_both_yaml_and_yml(self) -> None:
        """suffix() must return both .yaml and .yml."""
        suffixes = YamlDocumentReader.suffix()
        assert ".yaml" in suffixes
        assert ".yml" in suffixes

    def test_match_subclass_data_access_returns_path_for_string(self) -> None:
        """match_subclass_data_access returns path for explicit string (feature scope)."""
        result = YamlDocumentReader.match_subclass_data_access("some_path.yaml", ["feature1"])
        assert result == "some_path.yaml"

    def test_match_subclass_data_access_returns_none_for_data_access_collection(self) -> None:
        """match_subclass_data_access returns None for DataAccessCollection (no auto-discovery)."""
        from mloda.user import DataAccessCollection

        dac = DataAccessCollection(files={"some_path.yaml"})
        result = YamlDocumentReader.match_subclass_data_access(dac, ["feature1"])
        assert result is None
