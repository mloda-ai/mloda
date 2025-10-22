"""
Base test class for Feature Group testing.

This module provides a reusable base class that implements common test logic
for Feature Group operations across all compute frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, Union, Dict, List
import pandas as pd

from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_columns_exist,
    validate_row_count,
)


class FeatureGroupTestBase(ABC):
    """
    Base class for Feature Group tests.

    Subclasses must implement:
    - feature_group_class(): Return the FeatureGroup class to test

    This base class provides:
    - Data conversion utilities (to_pandas)
    - Framework-agnostic test logic
    - Common assertion logic
    """

    @abstractmethod
    def feature_group_class(self) -> Type[Any]:
        """Return the feature group class for this test."""
        pass

    def to_pandas(self, data: Union[Dict, Any]) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame.

        Args:
            data: Data to convert (dict or DataFrame)

        Returns:
            pandas DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data

        return pd.DataFrame(data)

    def to_framework(self, data: Union[Dict, Any], framework_type: Type[Any]) -> Any:
        """
        Convert data to target framework format.

        Args:
            data: Data to convert (dict or framework-specific type)
            framework_type: Target framework type (e.g., pd.DataFrame)

        Returns:
            Data in the target framework's native format
        """
        converter = DataConverter()
        if isinstance(data, dict):
            data_list = [{k: v[i] for k, v in data.items()} for i in range(len(next(iter(data.values()))))]
            return converter.to_framework(data_list, framework_type)
        return converter.to_framework(data, framework_type)

    def assert_columns_exist(self, result: Any, columns: List[str]) -> None:
        """
        Assert that specified columns exist in the result.

        Args:
            result: Result data to validate
            columns: List of column names that must exist

        Raises:
            AssertionError: If any expected columns are missing
        """
        validate_columns_exist(result, columns)

    def assert_row_count(self, result: Any, expected_count: int) -> None:
        """
        Assert that result has the expected number of rows.

        Args:
            result: Result data to validate
            expected_count: Expected number of rows

        Raises:
            AssertionError: If row count doesn't match expected
        """
        validate_row_count(result, expected_count)
