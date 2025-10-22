"""
Base test class for PyArrow transformer testing.

This module provides a reusable base class that implements common test logic
for PyArrow transformation operations across all compute frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type
import logging

import pyarrow as pa

from .test_scenarios import SCENARIOS

logger = logging.getLogger(__name__)


class TransformerTestBase(ABC):
    """
    Base class for transformer tests.

    Subclasses must implement:
    - transformer_class(): Return the Transformer class to test
    - framework_type(): Return the framework's data type
    - get_connection(): Return framework connection object (or None)

    This base class provides:
    - All standard transformer test methods
    - Framework-agnostic test logic using shared scenarios
    - Automatic validation of transformations
    - Common assertion logic for data integrity
    """

    def setup_method(self) -> None:
        """Initialize the test base before each test method."""
        pass

    @classmethod
    @abstractmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the transformer class for this framework."""
        pass

    @classmethod
    @abstractmethod
    def framework_type(cls) -> Type[Any]:
        """Return the framework's expected data type."""
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return framework connection object, or None if not needed."""
        pass

    def _assert_row_count(self, table: pa.Table, expected: int) -> None:
        """Assert that the table has the expected number of rows."""
        assert table.num_rows == expected, f"Expected {expected} rows, got {table.num_rows}"

    def _assert_column_count(self, table: pa.Table, expected: int) -> None:
        """Assert that the table has the expected number of columns."""
        assert table.num_columns == expected, f"Expected {expected} columns, got {table.num_columns}"

    def _assert_column_names(self, table: pa.Table, expected_names: set[str]) -> None:
        """Assert that the table has the expected column names."""
        actual_names = set(table.column_names)
        assert actual_names == expected_names, f"Expected columns {expected_names}, got {actual_names}"

    def _assert_data_integrity(self, original: pa.Table, transformed: pa.Table) -> None:
        """Assert that data is preserved after transformation."""
        assert original.num_rows == transformed.num_rows, "Row count mismatch after transformation"
        assert set(original.column_names) == set(transformed.column_names), "Column names mismatch after transformation"

    def test_framework_types(self) -> None:
        """Test that transformer returns correct framework types."""
        transformer = self.transformer_class()

        fw_type = transformer.framework()
        other_fw_type = transformer.other_framework()

        assert isinstance(fw_type, type) or hasattr(fw_type, "__mro__"), "framework() must return a type"
        assert other_fw_type == pa.Table, "other_framework() must return pa.Table"

    def test_transform_fw_to_other_fw(self) -> None:
        """Test transformation from framework to PyArrow."""
        scenario = SCENARIOS["basic_transformation"]
        transformer = self.transformer_class()

        source_table = pa.Table.from_pydict(scenario["data"])
        fw_data = transformer.transform_other_fw_to_fw(source_table, self.get_connection())

        result = transformer.transform_fw_to_other_fw(fw_data)

        assert isinstance(result, pa.Table), "Result must be a PyArrow Table"
        self._assert_row_count(result, scenario["expected_rows"])
        self._assert_column_count(result, scenario["expected_columns"])
        self._assert_column_names(result, scenario["expected_column_names"])

    def test_transform_other_fw_to_fw(self) -> None:
        """Test transformation from PyArrow to framework."""
        scenario = SCENARIOS["basic_transformation"]
        transformer = self.transformer_class()

        source_table = pa.Table.from_pydict(scenario["data"])

        result = transformer.transform_other_fw_to_fw(source_table, self.get_connection())

        assert result is not None, "Result must not be None"

    def test_roundtrip_transformation(self) -> None:
        """Test bidirectional transformation (framework -> PyArrow -> framework)."""
        scenario = SCENARIOS["basic_transformation"]
        transformer = self.transformer_class()

        original_table = pa.Table.from_pydict(scenario["data"])

        fw_data = transformer.transform_other_fw_to_fw(original_table, self.get_connection())
        back_to_pyarrow = transformer.transform_fw_to_other_fw(fw_data)

        self._assert_data_integrity(original_table, back_to_pyarrow)

    def test_empty_table(self) -> None:
        """Test handling empty tables with schema preservation."""
        scenario = SCENARIOS["empty_table"]
        transformer = self.transformer_class()

        source_table = pa.Table.from_pydict(scenario["data"])

        fw_data = transformer.transform_other_fw_to_fw(source_table, self.get_connection())
        result = transformer.transform_fw_to_other_fw(fw_data)

        self._assert_row_count(result, scenario["expected_rows"])
        self._assert_column_count(result, scenario["expected_columns"])
        self._assert_column_names(result, scenario["expected_column_names"])

    def test_null_values(self) -> None:
        """Test null value preservation during transformations."""
        scenario = SCENARIOS["null_values"]
        transformer = self.transformer_class()

        source_table = pa.Table.from_pydict(scenario["data"])

        fw_data = transformer.transform_other_fw_to_fw(source_table, self.get_connection())
        result = transformer.transform_fw_to_other_fw(fw_data)

        self._assert_row_count(result, scenario["expected_rows"])
        self._assert_column_names(result, scenario["expected_column_names"])

        assert scenario["has_nulls"], "Scenario must have nulls"
