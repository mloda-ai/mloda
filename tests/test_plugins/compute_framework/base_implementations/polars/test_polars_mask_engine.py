from decimal import Decimal
from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.polars.polars_mask_engine import (
    PolarsMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsMaskEngine(MaskEngineTestMixin):
    mask_engine_class = PolarsMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pl.DataFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    @pytest.fixture
    def empty_data(self) -> Any:
        return pl.DataFrame({"status": pl.Series([], dtype=pl.String), "value": pl.Series([], dtype=pl.Int64)})

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return bool(mask.dtype == pl.Boolean)

    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = data.filter(mask).to_dict(as_series=False)
        return result

    def test_is_in_decimal_raises_invalid_operation(self) -> None:
        data = pl.DataFrame({"d": [Decimal("12.34"), Decimal("5.50"), None]}, schema={"d": pl.Decimal(10, 2)})

        with pytest.raises(pl.exceptions.InvalidOperationError):
            PolarsMaskEngine.is_in(data, "d", [Decimal("12.34")])
