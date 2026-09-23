from decimal import Decimal
from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.spark.spark_mask_engine import (
    SparkMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark not available")
class TestSparkMaskEngine(MaskEngineTestMixin):
    mask_engine_class = SparkMaskEngine

    @pytest.fixture
    def sample_data(self, spark_session: Any) -> Any:
        return spark_session.createDataFrame(
            [
                ("active", 10),
                ("inactive", 20),
                ("active", 30),
                ("inactive", 40),
            ],
            ["status", "value"],
        )

    @pytest.fixture
    def empty_data(self, spark_session: Any) -> Any:
        return spark_session.createDataFrame([], "status string, value bigint")

    @pytest.fixture
    def null_data(self, spark_session: Any) -> Any:
        return spark_session.createDataFrame(
            [
                ("active", 10, 1),
                (None, 20, None),
                ("inactive", 30, 3),
                (None, 40, None),
            ],
            "status string, value bigint, score bigint",
        )

    @pytest.fixture
    def decimal_sample_data(self, spark_session: Any) -> Any:
        rows = [(Decimal("12.34"),), (Decimal("5.50"),), (None,)]
        return spark_session.createDataFrame(rows, "d decimal(10,2)")

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return all(isinstance(v, bool) for v in mask)

    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        kept = [row for row, keep in zip(data.collect(), mask) if keep]
        return {column: [row[column] for row in kept] for column in data.columns}
