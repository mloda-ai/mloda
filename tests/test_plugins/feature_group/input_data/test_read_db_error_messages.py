"""Tests for improved error messages in ReadDB."""

from typing import Any, Dict, Optional, Set, Union

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB


class ConcreteReadDB(ReadDB):
    """Minimal concrete ReadDB for testing error messages."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        return None

    @classmethod
    def is_valid_credentials(cls, credentials: Dict[str, Any]) -> bool:
        return False


class TestReadDBConnectionErrorMessage:
    """Tests for get_connection error message clarity."""

    def test_error_contains_class_name(self) -> None:
        with pytest.raises(ValueError, match="ConcreteReadDB"):
            ConcreteReadDB.get_connection({"host": "localhost"})

    def test_error_contains_credentials_type(self) -> None:
        with pytest.raises(ValueError, match="Credentials type: dict"):
            ConcreteReadDB.get_connection({"host": "localhost"})

    def test_error_contains_troubleshooting_guidance(self) -> None:
        with pytest.raises(ValueError, match="database server is reachable"):
            ConcreteReadDB.get_connection({"host": "localhost"})

    def test_error_with_string_credentials_shows_type(self) -> None:
        with pytest.raises(ValueError, match="Credentials type: str"):
            ConcreteReadDB.get_connection("sqlite:///test.db")


class TestReadDBMatchFeatureNamesErrorMessage:
    """Tests for match_read_db_data_access error message when multiple feature names."""

    def test_error_contains_feature_names(self) -> None:
        with pytest.raises(ValueError, match="expects exactly one"):
            ConcreteReadDB.match_read_db_data_access([], ["feat_a", "feat_b"])

    def test_error_contains_actual_feature_names(self) -> None:
        with pytest.raises(ValueError, match="feat_a.*feat_b"):
            ConcreteReadDB.match_read_db_data_access([], ["feat_a", "feat_b"])
