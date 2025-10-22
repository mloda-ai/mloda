from typing import Any, Optional, Type
import pytest

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import PythonDictFilterEngine
from tests.test_plugins.compute_framework.test_tooling.filter import FilterEngineTestBase


class TestPythonDictFilterEngine(FilterEngineTestBase):
    """Test PythonDictFilterEngine using shared filter test scenarios."""

    @classmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the PythonDictFilterEngine class."""
        return PythonDictFilterEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return List[Dict] type."""
        return list

    def get_connection(self) -> Optional[Any]:
        """Python Dict does not require a connection object."""
        return None
