"""Shared test mixin for all BaseMaskEngine implementations.

This mixin provides common test methods that verify the mask engine contract.
Each framework-specific test class should inherit from this mixin and provide:
- engine fixture: Returns the mask engine class
- sample_data fixture: Returns framework-specific test data
- evaluate_mask method: Converts framework-specific mask to a Python list of booleans
"""

from abc import abstractmethod
from typing import Any

import pytest

from mloda.provider import BaseMaskEngine


class MaskEngineTestMixin:
    """Shared tests for all BaseMaskEngine implementations.

    Each framework test class must provide:
    - engine fixture returning the engine class
    - sample_data fixture returning data with columns:
        status: ["active", "inactive", "active", "inactive"]
        value: [10, 20, 30, 40]
    - evaluate_mask(mask, data) converting the mask to list[bool]
    """

    @pytest.fixture
    @abstractmethod
    def engine(self) -> type[BaseMaskEngine]:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        raise NotImplementedError

    def test_equal(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.equal(sample_data, "status", "active")
        assert self.evaluate_mask(mask, sample_data) == [True, False, True, False]

    def test_greater_equal(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.greater_equal(sample_data, "value", 20)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, True]

    def test_less_equal(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.less_equal(sample_data, "value", 30)
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, False]

    def test_less_than(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.less_than(sample_data, "value", 30)
        assert self.evaluate_mask(mask, sample_data) == [True, True, False, False]

    def test_less_equal_inclusive(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.less_equal(sample_data, "value", 30)
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, False]

    def test_is_in(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.is_in(sample_data, "status", ("active",))
        assert self.evaluate_mask(mask, sample_data) == [True, False, True, False]

    def test_all_true(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.all_true(sample_data)
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, True]

    def test_combine_and(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 20)
        mask_max = engine.less_equal(sample_data, "value", 30)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_range_via_combine(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 15)
        mask_max = engine.less_equal(sample_data, "value", 35)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_range_exclusive_via_combine(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 15)
        mask_max = engine.less_than(sample_data, "value", 35)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]
