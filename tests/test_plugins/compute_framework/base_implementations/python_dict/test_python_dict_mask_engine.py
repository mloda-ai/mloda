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
        return {
            "status": ["active", "inactive", "active", "inactive"],
            "value": [10, 20, 30, 40],
        }

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return list(mask)


class TestPythonDictMaskEngineMissingColumn:
    """Red-phase regression tests for Defect 3 (Opus #1): a mask over a MISSING column must
    return a row-count-length all-False mask, not a length-0 list.

    ``PythonDictMaskEngine`` primitives read ``data.get(column, [])``. For a column NOT present
    in the columnar data this yields ``[]``, so the mask has length 0 instead of the row count.
    ``combine`` uses ``zip``, which silently truncates to the shorter operand, so
    ``combine(all_true(data), equal(data, "missing", v))`` collapses an AND-combination to
    ZERO rows even though every row should simply be excluded (all-False).

    Desired behavior: a mask over a missing column is a list of length = row count, all False;
    combining it with ``all_true`` preserves the row count.

    These tests are expected to FAIL against the current implementation: the missing-column mask
    is length 0 and the combined mask is length 0 (not 3).
    """

    @staticmethod
    def _engine() -> type[PythonDictMaskEngine]:
        return PythonDictMaskEngine

    def test_equal_on_missing_column_is_row_length_all_false(self) -> None:
        """``equal`` over a missing column returns length-3 all-False, not length-0."""
        data = {"a": [1, 2, 3]}
        mask = self._engine().equal(data, "missing_col", 1)
        assert list(mask) == [False, False, False]

    def test_greater_equal_on_missing_column_is_row_length_all_false(self) -> None:
        """``greater_equal`` over a missing column returns length-3 all-False, not length-0."""
        data = {"a": [1, 2, 3]}
        mask = self._engine().greater_equal(data, "missing_col", 1)
        assert list(mask) == [False, False, False]

    def test_is_in_on_missing_column_is_row_length_all_false(self) -> None:
        """``is_in`` over a missing column returns length-3 all-False, not length-0."""
        data = {"a": [1, 2, 3]}
        mask = self._engine().is_in(data, "missing_col", (1, 2))
        assert list(mask) == [False, False, False]

    def test_combine_all_true_with_missing_column_preserves_row_count(self) -> None:
        """``combine(all_true(data), equal(data, "missing", v))`` keeps 3 rows (all False).

        This is the corruption the defect describes: today ``zip`` truncates to the length-0
        missing-column mask, collapsing the AND-combination to zero rows.
        """
        engine = self._engine()
        data = {"a": [1, 2, 3]}
        combined = engine.combine(engine.all_true(data), engine.equal(data, "missing_col", 1))
        assert list(combined) == [False, False, False]
