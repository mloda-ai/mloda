"""Tests for error messages of BaseInputData.load without a match."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import BaseInputData


class ConcreteReadFile(BaseInputData):
    """Minimal concrete BaseInputData reader for testing error messages."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return None

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class TestReadFileLoadWithoutMatchErrorMessages:
    """load() without an input_data_match names the reader class and the missing attribute."""

    @staticmethod
    def _unmatched_features() -> FeatureSet:
        features = FeatureSet()
        features.add(Feature("read_file_unmatched_col"))
        return features

    def test_error_mentions_input_data_match(self) -> None:
        with pytest.raises(ValueError, match="input_data_match"):
            ConcreteReadFile().load(self._unmatched_features())

    def test_error_names_the_reader_class(self) -> None:
        with pytest.raises(ValueError, match="ConcreteReadFile"):
            ConcreteReadFile().load(self._unmatched_features())
