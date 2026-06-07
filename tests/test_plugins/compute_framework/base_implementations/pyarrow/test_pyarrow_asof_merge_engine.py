"""
Tests for PyArrowMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. PyArrow is both the framework
under test and the interchange format used by the DataConverter.
"""

from datetime import timedelta
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
        """Vector F: native PyArrow Acero (Table.join_asof) cannot express 'nearest' -> ValueError.

        Acero's asof join only supports a directional (forward/backward) tolerance match;
        a symmetric 'nearest' search is not available, so it must be rejected like the SQL engines.
        """
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_allow_exact_matches_false(self) -> None:
        """Override Vector C: Acero's asof match range always includes exact matches.

        Native PyArrow ``Table.join_asof`` performs an inclusive tolerance comparison, so
        there is no way to exclude an exact-time match. ``allow_exact_matches=False`` must
        therefore be rejected with a ValueError rather than silently returning the exact row.
        """
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            allow_exact_matches=False,
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tolerance_timedelta_rejected(self) -> None:
        """Acero requires a numeric (integer) tolerance; a timedelta cannot be expressed -> ValueError."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tolerance_float_rejected(self) -> None:
        """Acero's asof tolerance must be an integer; a float tolerance is rejected -> ValueError."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=5.0,
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
