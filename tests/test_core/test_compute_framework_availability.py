from typing import Any
import unittest
from unittest.mock import patch

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.prepare.accessible_plugins import PreFilterPlugins


class TestComputeFrameworkAvailability(unittest.TestCase):
    """Test that compute frameworks are properly filtered based on dependency availability."""

    def test_base_compute_framework_is_available_by_default(self) -> None:
        """Test that the base ComputeFrameWork class returns True for is_available()."""
        self.assertTrue(ComputeFrameWork.is_available())

    def test_pyarrow_framework_available_when_pyarrow_installed(self) -> None:
        """Test that PyarrowTable.is_available() returns True when pyarrow is installed."""
        try:
            import pyarrow

            # If we can import pyarrow, test that it's available
            from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

            self.assertTrue(PyarrowTable.is_available())
        except ImportError:
            # If pyarrow is not installed, skip this test
            self.skipTest("PyArrow not installed, skipping availability test")

    @patch("mloda_plugins.compute_framework.base_implementations.pyarrow.table.PyarrowTable.is_available")
    def test_get_cfw_subclasses_exclude_unavailable_frameworks(self, mock_is_available: Any) -> None:
        """Test that get_cfw_subclasses() includes available compute frameworks."""

        # PyarrowTable should be in the available frameworks
        framework_names = {fw.__name__ for fw in PreFilterPlugins.get_cfw_subclasses()}
        self.assertIn("PyarrowTable", framework_names)

        # Mock PyarrowTable as available
        mock_is_available.return_value = False

        # Get all available compute framework subclasses
        available_frameworks = PreFilterPlugins.get_cfw_subclasses()

        # Import PyarrowTable to check if it's included
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

        # PyarrowTable should be in the available frameworks
        framework_names = {fw.__name__ for fw in available_frameworks}
        self.assertNotIn("PyarrowTable", framework_names)

    def test_available_frameworks_only_include_available_ones(self) -> None:
        """Test that get_cfw_subclasses() only returns frameworks that report as available."""
        # Get available frameworks
        available_frameworks = PreFilterPlugins.get_cfw_subclasses()

        # Verify that all returned frameworks report as available
        for framework in available_frameworks:
            self.assertTrue(
                framework.is_available(),
                f"Framework {framework.__name__} should be available but reports as unavailable",
            )

    def test_pyarrow_availability_check_directly(self) -> None:
        """Test PyarrowTable availability check directly."""
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

        # Test the availability check
        is_available = PyarrowTable.is_available()
        assert is_available is True
