"""
Tests for PythonDictMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. PythonDict's native format is
list[dict], so conversion is a passthrough; the DataConverter still routes
through PyArrow for other frameworks, so PyArrow must be available.
"""

from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPythonDictAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for PythonDictMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PythonDictMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        return list

    def get_connection(self) -> Optional[Any]:
        return None

    def test_nearest_direction(self) -> None:
        """Vector F: PythonDict supports 'nearest' -> closer right row (t=8, gap 2) matched."""
        left = [{"k": 1, "t": 10, "lv": 100}]
        right = [{"k": 1, "t": 8, "rv": "A"}, {"k": 1, "t": 15, "rv": "B"}]

        engine = PythonDictMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result[0]["rv"] == "A"
