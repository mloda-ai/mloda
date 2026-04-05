"""Tests for improved error messages in ReadFile."""

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class ConcreteReadFile(ReadFile):
    """Minimal concrete ReadFile for testing error messages."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestReadFileInitReaderErrorMessages:
    """Tests for init_reader error message clarity."""

    def test_options_none_error_mentions_base_input_data(self) -> None:
        reader = ConcreteReadFile()
        with pytest.raises(ValueError, match="BaseInputData"):
            reader.init_reader(None)

    def test_options_none_error_contains_example(self) -> None:
        reader = ConcreteReadFile()
        with pytest.raises(ValueError, match="Options\\(context="):
            reader.init_reader(None)

    def test_reader_data_access_none_error_mentions_base_input_data(self) -> None:
        reader = ConcreteReadFile()
        options = Options(context={"other_key": "value"})
        with pytest.raises(ValueError, match="BaseInputData"):
            reader.init_reader(options)

    def test_reader_data_access_none_error_contains_example(self) -> None:
        reader = ConcreteReadFile()
        options = Options(context={"other_key": "value"})
        with pytest.raises(ValueError, match="ReaderClass, data_access"):
            reader.init_reader(options)
