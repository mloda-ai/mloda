"""
Tests for PandasMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase.
"""

from datetime import timedelta
from typing import Any, Optional

import pytest

from mloda.provider import FeatureGroup
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import Options
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig, JoinSpec, Link
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


class _TimedeltaAsofFG(FeatureGroup):
    """Mock feature group with a single by-key index column 'k' for asof factory tests."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("k",))]


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for PandasMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PandasMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pd is None:
            raise ImportError("Pandas is not installed")
        # mypy can't infer pd.DataFrame type correctly
        dataframe_type: type[Any] = pd.DataFrame
        return dataframe_type

    def get_connection(self) -> Optional[Any]:
        return None

    def test_timedelta_tolerance_filters_far_match_on_datetime_column(self) -> None:
        """timedelta tolerance must be honored on a real datetime time column (parity with
        pandas merge_asof): the near row (gap 2s <= 5s) matches; the far row (gap 92s > 5s)
        gets a null right value.

        This exercises the widened ``AsOfJoinConfig.tolerance`` type (``... | timedelta``)
        at runtime. pandas natively supports a timedelta tolerance on a datetime64 ``on``
        column and the engine forwards it through, so this is a characterization test of the
        promised behavior.
        """
        left = pd.DataFrame(
            {
                "k": [1, 1],
                "t": [
                    pd.Timestamp("2021-01-01 00:00:10"),
                    pd.Timestamp("2021-01-01 00:01:42"),  # +92s from the right row
                ],
                "lv": [1, 2],
            }
        )
        right = pd.DataFrame({"k": [1], "t": [pd.Timestamp("2021-01-01 00:00:08")], "rv": [7]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        rows = {row["t"]: self._normalize_value(row["rv"]) for _, row in result.iterrows()}
        assert rows[pd.Timestamp("2021-01-01 00:00:10")] == 7
        assert rows[pd.Timestamp("2021-01-01 00:01:42")] is None

    def test_boolean_time_column_raises_value_error(self) -> None:
        """A boolean time column is not orderable for an as-of join. pandas would otherwise pass
        it via is_numeric_dtype; the guard must reject it consistently with the other engines and
        raise a clear ValueError naming the column."""
        left = pd.DataFrame({"k": [1], "t": [True], "lv": [1]})
        right = pd.DataFrame({"k": [1], "t": [False], "rv": [7]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")
        with pytest.raises(ValueError, match=r"'t'.*(datetime|numeric)"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_config_and_factories_accept_timedelta_tolerance(self) -> None:
        """AsOfJoinConfig, Link.asof and Link.asof_on accept a timedelta tolerance and carry it
        through unchanged (type-widening characterization)."""
        td = timedelta(seconds=5)

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", tolerance=td)
        assert cfg.tolerance == td

        link = Link.asof(
            JoinSpec(_TimedeltaAsofFG, "k"),
            JoinSpec(_TimedeltaAsofFG, "k"),
            left_time_column="t",
            right_time_column="t",
            tolerance=td,
        )
        assert link.asof_config is not None
        assert link.asof_config.tolerance == td

        link_on = Link.asof_on(
            _TimedeltaAsofFG,
            _TimedeltaAsofFG,
            left_time_column="t",
            right_time_column="t",
            tolerance=td,
        )
        assert link_on.asof_config is not None
        assert link_on.asof_config.tolerance == td
