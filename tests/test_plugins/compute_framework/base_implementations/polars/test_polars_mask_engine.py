from typing import Any

import pytest

from mloda.provider import BaseMaskEngine
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.polars_mask_engine import (
        PolarsMaskEngine,
    )
except ImportError:
    pl = None  # type: ignore[assignment]
    PolarsMaskEngine = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsMaskEngine(MaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return PolarsMaskEngine

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
