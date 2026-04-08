from typing import Any

import pytest

from mloda.provider import BaseMaskEngine
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

if PYSPARK_AVAILABLE:
    from mloda_plugins.compute_framework.base_implementations.spark.spark_mask_engine import (
        SparkMaskEngine,
    )


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark not available")
class TestSparkMaskEngine(MaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return SparkMaskEngine

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

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)
