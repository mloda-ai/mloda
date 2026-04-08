from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import (
    PyArrowMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class TestPyArrowMaskEngine(MaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return PyArrowMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return mask.to_pylist()  # type: ignore[no-any-return]
