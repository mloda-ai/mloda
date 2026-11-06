"""Shared custom compute framework classes for testing.

These classes are used by multiple test files for testing compute framework
switching and join operations across different frameworks.
"""

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class SecondCfw(PyArrowTable):
    """Second custom compute framework for multi-CFW testing."""

    pass


class ThirdCfw(PyArrowTable):
    """Third custom compute framework for multi-CFW testing."""

    pass


class FourthCfw(PyArrowTable):
    """Fourth custom compute framework for multi-CFW testing."""

    pass
