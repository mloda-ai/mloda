from typing import Any

import pyarrow as pa
import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_mask_engine import (
    PyArrowFilterMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)


class TestPyArrowFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return PyArrowFilterMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def mask_to_list(self, mask: Any) -> list[bool]:
        return mask.to_pylist()  # type: ignore[no-any-return]
