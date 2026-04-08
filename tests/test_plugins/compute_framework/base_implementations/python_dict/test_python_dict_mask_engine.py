from typing import Any

import pytest

from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class TestPythonDictMaskEngine(MaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return PythonDictMaskEngine

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
