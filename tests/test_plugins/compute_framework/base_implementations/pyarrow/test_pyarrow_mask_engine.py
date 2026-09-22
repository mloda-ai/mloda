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

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return mask.to_pylist()  # type: ignore[no-any-return]

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return bool(mask.type == pa.bool_())

    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = data.filter(mask).to_pydict()
        return result
