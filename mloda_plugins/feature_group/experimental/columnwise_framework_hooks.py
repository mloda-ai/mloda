"""Shared abstract framework hooks for column-wise experimental feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ColumnwiseFrameworkHooks(ABC):
    """Declares the check/add hooks every column-wise feature group delegates to its compute framework."""

    @classmethod
    @abstractmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        """
        Check that the resolved source features exist in the data.

        Args:
            data: The input data
            feature_names: Resolved source feature names (may contain ~N suffixes)

        Raises:
            ValueError: When the source features this feature group needs are absent. Tolerant
                groups raise only when none of the names exist, strict groups raise when any name
                is missing; see test_check_source_features_signature.py.
        """
        ...

    @classmethod
    @abstractmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add

        Returns:
            The updated data
        """
        ...


class ColumnDiscoveryFrameworkHooks(ColumnwiseFrameworkHooks):
    """Adds the column discovery hook for feature groups that resolve column names against the data."""

    @classmethod
    @abstractmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """
        Get the set of available column names from the data.

        Args:
            data: The input data

        Returns:
            Set of column names available in the data
        """
        ...
