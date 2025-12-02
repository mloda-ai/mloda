"""
Base test class for dataframe merge engine testing.

This module provides a reusable base class that implements common test logic
for merge operations across all compute frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes


class DataFrameTestBase(ABC):
    """
    Base class for dataframe merge engine tests.

    Subclasses must implement:
    - framework_class(): Return the framework class to test
    - create_dataframe(data): Create framework-specific dataframe from dict
    - get_connection(): Return framework connection object (or None)

    This base class provides:
    - Helper methods for creating test frameworks
    - Helper methods for assertions
    """

    left_data_dict: dict[str, list[Any]] = {"idx": [1, 3], "col1": ["a", "b"]}
    right_data_dict: dict[str, list[Any]] = {"idx": [1, 2], "col2": ["x", "z"]}
    idx: Index = Index(("idx",))

    @classmethod
    @abstractmethod
    def framework_class(cls) -> Type[Any]:
        """Return the framework class for this framework."""
        pass

    @abstractmethod
    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create framework-specific dataframe from dict."""
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return framework connection object, or None if not needed."""
        pass

    def setup_method(self) -> None:
        """Set up test data fixtures for each test method."""
        self.left_data = self.create_dataframe(self.left_data_dict)
        self.right_data = self.create_dataframe(self.right_data_dict)

    def _create_test_framework(self) -> Any:
        """Create a framework instance with sync mode and empty children."""
        framework_cls = self.framework_class()
        return framework_cls(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

    def _get_merge_engine(self, framework: Any) -> Any:
        """Get merge engine class from framework."""
        return framework.merge_engine()

    def _assert_row_count(self, result: Any, expected: int) -> None:
        """Assert that result has expected number of rows."""
        actual = len(result)
        assert actual == expected, f"Expected {expected} rows, got {actual}"

    def _assert_result_equals(self, result: Any, expected: Any, sort_columns: Optional[List[str]] = None) -> None:
        """Perform framework-aware equality check."""
        if sort_columns is not None:
            result = result.sort(sort_columns)
            expected = expected.sort(sort_columns)

        assert result.equals(expected)

    def test_merge_inner(self) -> None:
        framework = self._create_test_framework()
        framework.data = self.left_data
        merge_engine = self._get_merge_engine(framework)
        result = merge_engine().merge(framework.data, self.right_data, JoinType.INNER, self.idx, self.idx)
        self._assert_row_count(result, 1)

    def test_merge_left(self) -> None:
        framework = self._create_test_framework()
        framework.data = self.left_data
        merge_engine = self._get_merge_engine(framework)
        result = merge_engine().merge(framework.data, self.right_data, JoinType.LEFT, self.idx, self.idx)
        self._assert_row_count(result, 2)

    def test_merge_right(self) -> None:
        framework = self._create_test_framework()
        framework.data = self.left_data
        merge_engine = self._get_merge_engine(framework)
        result = merge_engine().merge(framework.data, self.right_data, JoinType.RIGHT, self.idx, self.idx)
        self._assert_row_count(result, 2)

    def test_merge_full_outer(self) -> None:
        framework = self._create_test_framework()
        framework.data = self.left_data
        merge_engine = self._get_merge_engine(framework)
        result = merge_engine().merge(framework.data, self.right_data, JoinType.OUTER, self.idx, self.idx)
        self._assert_row_count(result, 3)

    def test_merge_append(self) -> None:
        framework = self._create_test_framework()
        framework.data = self.left_data
        merge_engine = self._get_merge_engine(framework)
        result = merge_engine().merge(framework.data, self.right_data, JoinType.APPEND, self.idx, self.idx)
        self._assert_row_count(result, 4)

    def test_merge_union(self) -> None:
        framework = self._create_test_framework()
        framework.data = self.left_data
        merge_engine = self._get_merge_engine(framework)
        result = merge_engine().merge(framework.data, self.right_data, JoinType.UNION, self.idx, self.idx)
        self._assert_row_count(result, 4)
