"""Shared test mixin for all BaseFilterMaskEngine implementations.

This mixin provides common test methods that verify the filter mask engine contract.
Each framework-specific test class should inherit from this mixin and provide:
- engine fixture: Returns the filter mask engine class
- sample_data fixture: Returns framework-specific test data
- evaluate_mask method: Converts framework-specific mask to a Python list of booleans
"""

from abc import abstractmethod
from typing import Any

import pytest

from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import FeatureSet, FilterMask, BaseFilterMaskEngine


def _make_features(
    filters: set[SingleFilter] | None,
    engine: type[BaseFilterMaskEngine],
) -> FeatureSet:
    """Create a FeatureSet with pre-resolved mask engine and filters."""
    fs = FeatureSet()
    fs.filters = filters
    fs.filter_mask_engine = engine
    return fs


class FilterMaskEngineTestMixin:
    """Shared tests for all BaseFilterMaskEngine implementations.

    Each framework test class must provide:
    - engine fixture returning the engine class
    - sample_data fixture returning data with columns:
        status: ["active", "inactive", "active", "inactive"]
        value: [10, 20, 30, 40]
    - evaluate_mask(mask, data) converting the mask to list[bool]
    """

    @pytest.fixture
    @abstractmethod
    def engine(self) -> type[BaseFilterMaskEngine]:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        raise NotImplementedError

    def test_equal_filter(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("status", "equal", {"value": "active"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert self.evaluate_mask(mask, sample_data) == [True, False, True, False]

    def test_min_filter(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "min", {"value": 20})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, True]

    def test_max_filter_simple(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "max", {"value": 30})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, False]

    def test_max_filter_exclusive(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "max", {"max": 30, "max_exclusive": True})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [True, True, False, False]

    def test_max_filter_inclusive(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "max", {"max": 30, "max_exclusive": False})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, False]

    def test_greater_than(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.greater_than(sample_data, "value", 20)
        assert self.evaluate_mask(result, sample_data) == [False, False, True, True]

    def test_range_filter(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "range", {"min": 15, "max": 35})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_range_filter_exclusive(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "range", {"min": 15, "max": 35, "max_exclusive": True})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_categorical_inclusion(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("status", "categorical_inclusion", {"values": ("active",)})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert self.evaluate_mask(mask, sample_data) == [True, False, True, False]

    def test_no_matching_filters(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("other", "equal", {"value": "x"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, True]

    def test_none_filters(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        features = _make_features(None, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, True]

    def test_empty_filters(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        features = _make_features(set(), engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, True]

    def test_multiple_filters_same_column(self, engine: type[BaseFilterMaskEngine], sample_data: Any) -> None:
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        features = _make_features({sf_min, sf_max}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]
