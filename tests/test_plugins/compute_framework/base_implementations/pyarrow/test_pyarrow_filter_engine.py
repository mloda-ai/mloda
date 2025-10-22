from typing import Any, Optional, Type
import pytest

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine
from tests.test_plugins.compute_framework.test_tooling.filter import FilterEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowFilterEngine(FilterEngineTestBase):
    """Test PyArrowFilterEngine using shared filter test scenarios."""

    @classmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the PyArrowFilterEngine class."""
        return PyArrowFilterEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return PyArrow Table type."""
        if pa is None:
            raise ImportError("PyArrow is not installed")
        table_type: Type[Any] = pa.Table
        return table_type

    def get_connection(self) -> Optional[Any]:
        """PyArrow does not require a connection object."""
        return None
