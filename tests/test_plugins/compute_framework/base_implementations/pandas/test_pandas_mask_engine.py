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

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)
