from collections.abc import Iterator
from typing import Any

import pytest

from mloda._testing.compute_framework.mask_suite import MaskEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, SparkAdapter


class TestSparkMaskEngine(MaskEngineSuite):
    adapter_class = SparkAdapter

    @pytest.fixture
    def adapter(self, spark_session: Any) -> Iterator[FrameworkAdapter]:
        """SparkAdapter borrows the active session, so spark_session must exist first."""
        spark_adapter = self.adapter_class()
        with spark_adapter:
            yield spark_adapter

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        return all(isinstance(v, bool) for v in mask)

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> Any:
        kept = [row for row, keep in zip(data.collect(), mask) if keep]
        return adapter.connection.createDataFrame(kept, data.schema)
