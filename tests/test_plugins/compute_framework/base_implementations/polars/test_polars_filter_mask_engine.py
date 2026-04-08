from typing import Any

import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_mask_engine import (
        PolarsFilterMaskEngine,
    )
except ImportError:
    pl = None  # type: ignore[assignment]
    PolarsFilterMaskEngine = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return PolarsFilterMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pl.DataFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)
