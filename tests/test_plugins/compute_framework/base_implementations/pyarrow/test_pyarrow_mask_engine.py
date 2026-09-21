from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import (
    PyArrowMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class TestPyArrowMaskEngine(MaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return PyArrowMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return mask.to_pylist()  # type: ignore[no-any-return]

    def test_all_true_on_empty_table_combines_with_condition(self, engine: type[BaseMaskEngine]) -> None:
        table = pa.table({"a": pa.array([], type=pa.string())})
        all_true = engine.all_true(table)
        assert all_true.type == pa.bool_()
        assert engine.combine(all_true, engine.equal(table, "a", "x")).to_pylist() == []
