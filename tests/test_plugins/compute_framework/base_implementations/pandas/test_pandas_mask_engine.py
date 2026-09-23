from decimal import Decimal
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class TestPandasMaskEngine(MaskEngineTestMixin):
    mask_engine_class = PandasMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pd.DataFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    @pytest.fixture
    def empty_data(self) -> Any:
        return pd.DataFrame({"status": pd.Series([], dtype=str), "value": pd.Series([], dtype="int64")})

    @pytest.fixture
    def decimal_sample_data(self) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), None]
        return pd.DataFrame({"d": pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(10, 2)))})

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return bool(mask.dtype == bool)

    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = data[mask].to_dict("list")
        return result
