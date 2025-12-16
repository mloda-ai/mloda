from typing import Any
from unittest.mock import patch

import pytest

from mloda import ComputeFramework
from mloda.core.prepare.accessible_plugins import PreFilterPlugins


class TestComputeFrameworkAvailability:
    """Test that compute frameworks are properly filtered based on dependency availability."""

    def test_base_compute_framework_is_available_by_default(self) -> None:
        """Test that the base ComputeFramework class returns True for is_available()."""
        assert ComputeFramework.is_available()

    def test_pyarrow_framework_available_when_pyarrow_installed(self) -> None:
        """Test that PyArrowTable.is_available() returns True when pyarrow is installed."""
        try:
            import pyarrow  # noqa: F401

            # If we can import pyarrow, test that it's available
            from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

            assert PyArrowTable.is_available()
        except ImportError:
            # If pyarrow is not installed, skip this test
            pytest.skip("PyArrow not installed, skipping availability test")

    @patch("mloda_plugins.compute_framework.base_implementations.pyarrow.table.PyArrowTable.is_available")
    def test_get_cfw_subclasses_exclude_unavailable_frameworks(self, mock_is_available: Any) -> None:
        """Test that get_cfw_subclasses() includes available compute frameworks."""

        # PyArrowTable should be in the available frameworks
        framework_names = {fw.__name__ for fw in PreFilterPlugins.get_cfw_subclasses()}
        assert "PyArrowTable" in framework_names

        # Mock PyArrowTable as available
        mock_is_available.return_value = False

        # Get all available compute framework subclasses
        available_frameworks = PreFilterPlugins.get_cfw_subclasses()

        # Import PyArrowTable to check if it's included
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401

        # PyArrowTable should be in the available frameworks
        framework_names = {fw.__name__ for fw in available_frameworks}
        assert "PyArrowTable" not in framework_names

    def test_available_frameworks_only_include_available_ones(self) -> None:
        """Test that get_cfw_subclasses() only returns frameworks that report as available."""
        # Get available frameworks
        available_frameworks = PreFilterPlugins.get_cfw_subclasses()

        # Verify that all returned frameworks report as available
        for framework in available_frameworks:
            assert framework.is_available(), (
                f"Framework {framework.__name__} should be available but reports as unavailable"
            )

    def test_pyarrow_availability_check_directly(self) -> None:
        """Test PyArrowTable availability check directly."""
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        # Test the availability check
        is_available = PyArrowTable.is_available()
        assert is_available is True
