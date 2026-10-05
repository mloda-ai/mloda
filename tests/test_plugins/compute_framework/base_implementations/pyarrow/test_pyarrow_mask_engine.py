from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import (
    PyArrowMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class TestPyArrowMaskEngine(MaskEngineTestMixin):
    mask_engine_class = PyArrowMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    @pytest.fixture
    def empty_data(self) -> Any:
        return pa.table({"status": pa.array([], type=pa.string()), "value": pa.array([], type=pa.int64())})

    @pytest.fixture
    def null_data(self) -> Any:
        return pa.table(
            {
                "status": pa.array(["active", None, "inactive", None], type=pa.string()),
                "value": pa.array([10, 20, 30, 40], type=pa.int64()),
                "score": pa.array([1, None, 3, None], type=pa.int64()),
                "ratio": pa.array([1.0, float("nan"), 3.0, None], type=pa.float64()),
            }
        )

    @pytest.fixture
    def decimal_sample_data(self) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), None]
        return pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))})

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return mask.to_pylist()  # type: ignore[no-any-return]

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return bool(mask.type == pa.bool_())

    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = data.filter(mask).to_pydict()
        return result
