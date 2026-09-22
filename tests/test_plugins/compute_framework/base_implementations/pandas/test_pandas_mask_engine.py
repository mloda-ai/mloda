from typing import Any

import pandas as pd
import pytest

from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class TestPandasMaskEngine(MaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return PandasMaskEngine

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

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return bool(mask.dtype == bool)

    def test_all_true_on_empty_frame_keeps_columns(self, engine: type[BaseMaskEngine], empty_data: Any) -> None:
        mask = engine.all_true(empty_data)
        assert empty_data[mask].shape == (0, 2)
