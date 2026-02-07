from typing import Any, Tuple
from unittest.mock import patch

import pytest

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, Options
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature


class ConcreteReadDocument(ReadDocument):
    """Minimal concrete subclass for testing calculate_feature delegation."""

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".json",)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"doc_key": "doc_value"}


class TestReadDocumentFeatureInheritance:
    def test_read_document_feature_is_subclass_of_feature_group(self) -> None:
        assert issubclass(ReadDocumentFeature, FeatureGroup)


class TestReadDocumentFeatureInputData:
    def test_input_data_returns_read_document(self) -> None:
        result = ReadDocumentFeature.input_data()
        assert result is not None
        assert isinstance(result, ReadDocument)


class TestReadDocumentFeatureCalculate:
    def test_calculate_feature_delegates_to_reader(self) -> None:
        options = Options(group={"BaseInputData": (ConcreteReadDocument, "/path/to/doc.json")})
        features = FeatureSet()
        features.add(Feature("doc_key", options=options))

        result = ReadDocumentFeature.calculate_feature(None, features)

        assert result == {"doc_key": "doc_value"}

    def test_calculate_feature_raises_when_no_reader(self) -> None:
        features = FeatureSet()
        features.add(Feature("some_feature", options=Options()))

        with patch.object(ReadDocumentFeature, "input_data", return_value=None):
            with pytest.raises(ValueError, match="No reader available"):
                ReadDocumentFeature.calculate_feature(None, features)
