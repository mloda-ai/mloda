"""
Tests for PyArrowMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. PyArrow is both the framework
under test and the interchange format used by the DataConverter.
"""

from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for PyArrowMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PyArrowMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pa is None:
            raise ImportError("PyArrow is not installed")
        table_type: type[Any] = pa.Table
        return table_type

    def get_connection(self) -> Optional[Any]:
        return None

    def test_nearest_direction(self) -> None:
        """Vector F: PyArrow supports 'nearest' -> closer right row (t=8, gap 2) matched."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_dicts = result.to_pylist()
        assert len(result_dicts) == 1
        assert result_dicts[0]["rv"] == "A"
