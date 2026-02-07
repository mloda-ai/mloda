from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest

from mloda.provider import BaseInputData, FeatureSet
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.feature_group.input_data.read_document import ReadDocument


class ConcreteReadDocument(ReadDocument):
    """Minimal concrete subclass for testing load/init_reader behavior."""

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".json",)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"content": "test_data"}


class TestReadDocumentInheritance:
    def test_read_document_is_subclass_of_base_input_data(self) -> None:
        assert issubclass(ReadDocument, BaseInputData)


class TestReadDocumentMatchSubclass:
    def test_match_subclass_data_access_returns_none_with_data_access_collection(self) -> None:
        data_access = DataAccessCollection(files={"doc.json"})
        result = ReadDocument.match_subclass_data_access(data_access, ["content"])
        assert result is None

    def test_match_subclass_data_access_returns_path_with_string(self) -> None:
        result = ReadDocument.match_subclass_data_access("/path/to/doc.json", ["content"])
        assert result == "/path/to/doc.json"

    def test_match_subclass_data_access_returns_path_with_path_object(self) -> None:
        result = ReadDocument.match_subclass_data_access(Path("/path/to/doc.json"), ["content"])
        assert result == Path("/path/to/doc.json")

    def test_match_subclass_data_access_returns_none_with_none(self) -> None:
        result = ReadDocument.match_subclass_data_access(None, ["content"])
        assert result is None

    def test_match_subclass_data_access_returns_none_with_arbitrary_object(self) -> None:
        result = ReadDocument.match_subclass_data_access(object(), ["content"])
        assert result is None


class TestReadDocumentAbstractMethods:
    def test_load_data_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            ReadDocument.load_data("any_access", FeatureSet())

    def test_suffix_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            ReadDocument.suffix()


class TestReadDocumentLoad:
    def _make_feature_set(self, options: Options) -> FeatureSet:
        fs = FeatureSet()
        fs.add(Feature("doc_content", options=options))
        return fs

    def test_load_delegates_to_reader(self) -> None:
        options = Options(group={"BaseInputData": (ConcreteReadDocument, "/path/doc.json")})
        features = self._make_feature_set(options)

        instance = ConcreteReadDocument()
        result = instance.load(features)

        assert result == {"content": "test_data"}

    def test_load_raises_when_options_none(self) -> None:
        features = FeatureSet()
        feature = Feature("doc_content")
        feature.options = None  # type: ignore[assignment]
        features.add(feature)

        instance = ConcreteReadDocument()
        with pytest.raises(ValueError):
            instance.load(features)

    def test_load_raises_when_data_is_none(self) -> None:
        class NoneReturningReader(ReadDocument):
            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".json",)

            @classmethod
            def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
                return None

        options = Options(group={"BaseInputData": (NoneReturningReader, "/path/doc.json")})
        features = FeatureSet()
        features.add(Feature("doc_content", options=options))

        instance = NoneReturningReader()
        with pytest.raises(ValueError):
            instance.load(features)


class TestReadDocumentInitReader:
    def test_init_reader_extracts_from_options(self) -> None:
        options = Options(group={"BaseInputData": (ConcreteReadDocument, "/data/doc.json")})

        instance = ConcreteReadDocument()
        reader, data_access = instance.init_reader(options)

        assert isinstance(reader, ConcreteReadDocument)
        assert data_access == "/data/doc.json"

    def test_init_reader_raises_when_options_none(self) -> None:
        instance = ConcreteReadDocument()
        with pytest.raises(ValueError):
            instance.init_reader(None)

    def test_init_reader_raises_when_base_input_data_missing(self) -> None:
        options = Options(group={"some_other_key": "value"})

        instance = ConcreteReadDocument()
        with pytest.raises(ValueError):
            instance.init_reader(options)
