from typing import Any

import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

if PYSPARK_AVAILABLE:
    from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_mask_engine import (
        SparkFilterMaskEngine,
    )


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark not available")
class TestSparkFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return SparkFilterMaskEngine

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
