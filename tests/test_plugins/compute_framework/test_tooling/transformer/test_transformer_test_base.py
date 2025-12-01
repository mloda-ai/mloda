"""
Test for TransformerTestBase abstract base class.

This test validates that the TransformerTestBase provides a reusable testing framework
for PyArrow transformer implementations across different compute frameworks.
"""

from typing import Any, Optional, Type
import pytest

import pyarrow as pa

from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import (
    TransformerTestBase,
)
from tests.test_plugins.compute_framework.test_tooling.transformer.test_scenarios import SCENARIOS


class MockTransformer:
    """Mock transformer class for testing the TransformerTestBase."""

    @classmethod
    def framework(cls) -> Any:
        """Return mock framework type."""
        return type("MockFramework", (), {})

    @classmethod
    def other_framework(cls) -> Any:
        """Return PyArrow Table type."""
        return pa.Table

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """Transform from mock framework to PyArrow."""
        # Mock implementation: convert dict to PyArrow Table
        if hasattr(data, "to_pydict"):
            return data
        return pa.Table.from_pydict(data)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """Transform from PyArrow to mock framework."""
        # Mock implementation: return PyArrow table as-is (simulating framework data)
        return data


class TestTransformerTestBaseImplementation(TransformerTestBase):
    """Concrete test class implementing TransformerTestBase abstract methods."""

    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the mock transformer class."""
        return MockTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return the mock framework type."""
        return MockTransformer.framework()  # type: ignore[no-any-return]

    def get_connection(self) -> Optional[Any]:
        """No connection needed for mock transformer."""
        return None


class TestTransformerTestBase:
    """Test that TransformerTestBase provides required test methods."""

    def test_transformer_test_base_provides_framework_types_test(self) -> None:
        """
        Test that TransformerTestBase provides test_framework_types() method.

        This test validates that the base class includes a test method that verifies
        the transformer's framework() and other_framework() return correct types.
        """
        # Create an instance of our test implementation
        test_instance = TestTransformerTestBaseImplementation()

        # Verify the base class provides test_framework_types method
        assert hasattr(test_instance, "test_framework_types"), (
            "TransformerTestBase must provide test_framework_types() method"
        )

        # Execute the test method - it should validate framework types
        test_instance.test_framework_types()

    def test_transformer_test_base_provides_transform_fw_to_other_fw_test(self) -> None:
        """
        Test that TransformerTestBase provides test_transform_fw_to_other_fw() method.

        This test validates that the base class includes a test method for testing
        transformation from framework to PyArrow using scenarios.
        """
        test_instance = TestTransformerTestBaseImplementation()

        assert hasattr(test_instance, "test_transform_fw_to_other_fw"), (
            "TransformerTestBase must provide test_transform_fw_to_other_fw() method"
        )

        # Execute the test - it should use SCENARIOS to validate transformation
        test_instance.test_transform_fw_to_other_fw()

    def test_transformer_test_base_provides_transform_other_fw_to_fw_test(self) -> None:
        """
        Test that TransformerTestBase provides test_transform_other_fw_to_fw() method.

        This test validates that the base class includes a test method for testing
        transformation from PyArrow to framework.
        """
        test_instance = TestTransformerTestBaseImplementation()

        assert hasattr(test_instance, "test_transform_other_fw_to_fw"), (
            "TransformerTestBase must provide test_transform_other_fw_to_fw() method"
        )

        test_instance.test_transform_other_fw_to_fw()

    def test_transformer_test_base_provides_roundtrip_test(self) -> None:
        """
        Test that TransformerTestBase provides test_roundtrip_transformation() method.

        This test validates that the base class includes a test method for bidirectional
        transformation testing (framework -> PyArrow -> framework).
        """
        test_instance = TestTransformerTestBaseImplementation()

        assert hasattr(test_instance, "test_roundtrip_transformation"), (
            "TransformerTestBase must provide test_roundtrip_transformation() method"
        )

        test_instance.test_roundtrip_transformation()

    def test_transformer_test_base_provides_empty_table_test(self) -> None:
        """
        Test that TransformerTestBase provides test_empty_table() method.

        This test validates that the base class includes a test method for handling
        empty tables with schema preservation.
        """
        test_instance = TestTransformerTestBaseImplementation()

        assert hasattr(test_instance, "test_empty_table"), "TransformerTestBase must provide test_empty_table() method"

        test_instance.test_empty_table()

    def test_transformer_test_base_provides_null_values_test(self) -> None:
        """
        Test that TransformerTestBase provides test_null_values() method.

        This test validates that the base class includes a test method for verifying
        null value preservation during transformations.
        """
        test_instance = TestTransformerTestBaseImplementation()

        assert hasattr(test_instance, "test_null_values"), "TransformerTestBase must provide test_null_values() method"

        test_instance.test_null_values()

    def test_transformer_test_base_uses_scenarios(self) -> None:
        """
        Test that TransformerTestBase uses predefined scenarios from test_scenarios.py.

        This validates that the base class leverages the SCENARIOS dictionary for
        framework-agnostic testing.
        """
        # Verify SCENARIOS is accessible and contains expected test cases
        assert "basic_transformation" in SCENARIOS
        assert "empty_table" in SCENARIOS
        assert "null_values" in SCENARIOS

        # The base class should use these scenarios in its test methods
        test_instance = TestTransformerTestBaseImplementation()

        # When setup_method is called, it should be ready to use scenarios
        if hasattr(test_instance, "setup_method"):
            test_instance.setup_method()
