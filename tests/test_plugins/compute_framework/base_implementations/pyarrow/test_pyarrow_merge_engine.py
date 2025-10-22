from typing import Any, Optional, Type
import pytest

from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)

try:
    import pyarrow as pa
except ImportError:
    pa = None


class TestPyArrowMergeEngineMultiIndex(MultiIndexMergeEngineTestBase):
    """Test PyArrowMergeEngine using shared multi-index test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> Type[BaseMergeEngine]:
        """Return the PyArrowMergeEngine class."""
        return PyArrowMergeEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return pyarrow Table type."""
        if pa is None:
            raise ImportError("PyArrow is not installed")
        # mypy can't infer pa.Table type correctly
        table_type: Type[Any] = pa.Table
        return table_type

    def get_connection(self) -> Optional[Any]:
        """PyArrow does not require a connection object."""
        return None

    @pytest.mark.skip(reason="PyArrow does not support UNION operations - see GitHub issue #30950")
    def test_merge_union_with_multi_index(self) -> None:
        """Skip UNION test for PyArrow as it's not supported."""
        pass
