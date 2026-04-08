from typing import Any

import pandas as pd
import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_mask_engine import (
    PandasFilterMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)


class TestPandasFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return PandasFilterMaskEngine

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
