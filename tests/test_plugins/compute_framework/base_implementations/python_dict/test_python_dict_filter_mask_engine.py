from typing import Any

import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_mask_engine import (
    PythonDictFilterMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)


class TestPythonDictFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return PythonDictFilterMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return [
            {"status": "active", "value": 10},
            {"status": "inactive", "value": 20},
            {"status": "active", "value": 30},
            {"status": "inactive", "value": 40},
        ]

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)
