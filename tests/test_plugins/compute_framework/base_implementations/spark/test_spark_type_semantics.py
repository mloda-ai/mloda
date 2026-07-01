"""Failing tests for the spark column-semantics introspector (epic #518, Phase 1).

Defines the not-yet-implemented module-level function ``column_semantics(sdf, column)``
in ``mloda_plugins.compute_framework.base_implementations.spark.spark_type_semantics``.

Spark exposes no sub-type unit, so ``unit`` must be None for all columns. Timezone
awareness maps to the Spark type: ``TimestampType`` => aware, ``TimestampNTZType`` => naive.
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

import pytest
from datetime import datetime
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.spark.spark_type_semantics import column_semantics

# Import shared fixtures and availability flags from conftest.py
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        LongType,
        TimestampType,
        TimestampNTZType,
    )
else:
    StructType = None
    StructField = None
    StringType = None
    LongType = None
    TimestampType = None
    TimestampNTZType = None


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkColumnSemantics:
    @pytest.fixture
    def sdf(self, spark_session: Any) -> Any:
        schema = StructType(
            [
                StructField("ts_naive", TimestampNTZType()),
                StructField("ts_aware", TimestampType()),
                StructField("num", LongType()),
                StructField("s", StringType()),
            ]
        )
        rows = [
            (datetime(2021, 1, 1), datetime(2021, 1, 1), 1, "a"),
            (datetime(2021, 1, 2), datetime(2021, 1, 2), 2, "b"),
        ]
        return spark_session.createDataFrame(rows, schema=schema)

    def test_ts_naive(self, sdf: Any) -> None:
        sem = column_semantics(sdf, "ts_naive")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_ts_aware(self, sdf: Any) -> None:
        sem = column_semantics(sdf, "ts_aware")
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit is None

    def test_num(self, sdf: Any) -> None:
        sem = column_semantics(sdf, "num")
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_s(self, sdf: Any) -> None:
        sem = column_semantics(sdf, "s")
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None
