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

    def test_tie_breaks_ascending_and_is_order_independent(self) -> None:
        """
        Tie determinism: two right rows share the identical boundary timestamp within a
        by-key. The winner must be the row whose right non-key column values sort ASCENDING
        (smallest wins), matching the sqlite backend fix (fc805f0), and the result must be
        INDEPENDENT of right-input ordering.

        Left {k:1, t:10, lv:1}; two right rows tie at t=10 (== left t, backward, allow_exact
        default True): {k:1, t:10, rv:5} and {k:1, t:10, rv:2}. Expected exactly one row with
        rv==2 regardless of which order the tied right rows are supplied in.
        """
        left = [{"k": 1, "t": 10, "lv": 1}]
        right_a = [{"k": 1, "t": 10, "rv": 5}, {"k": 1, "t": 10, "rv": 2}]
        right_b = [{"k": 1, "t": 10, "rv": 2}, {"k": 1, "t": 10, "rv": 5}]

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            allow_exact_matches=True,
        )

        result_a = PythonDictMergeEngine().merge_asof(left, right_a, Index(("k",)), Index(("k",)), cfg)
        assert len(result_a) == 1
        assert result_a[0]["rv"] == 2

        # Order-independence: reversed right input must yield the identical winner.
        result_b = PythonDictMergeEngine().merge_asof(left, right_b, Index(("k",)), Index(("k",)), cfg)
        assert len(result_b) == 1
        assert result_b[0]["rv"] == 2
