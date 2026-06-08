"""Unit tests for the FeatureGroup-declared ``allow_empty_result`` policy.

- ``PythonDictFramework.select_data_by_column_names`` / ``.transform`` are
  unconditionally empty-safe: on empty rows they return ``[]``. They only
  propagate emptiness, they never judge it.
- The empty short-circuit triggers only on empty rows, never on missing columns:
  non-empty data with an absent requested column still raises ``ValueError``.
- ``ComputeFramework._extract_column_names(data)`` reports schema presence;
  ``PythonDictFramework`` returns the union of row keys (empty set for ``[]``).
- ``_validate_filter_columns`` consults ``feature_group.allow_empty_result()``
  to decide whether to skip validation on zero-row data.
- ``EmptyResultError`` is a ``ValueError`` subclass.
"""

from typing import Any

import pytest

from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.single_filter import SingleFilter
from tests.test_plugins.compute_framework.base_implementations.empty_result_test_mixin import (
    EmptyResultFrameworkTestMixin,
)


class _FakeFeaturesWithFilter:
    """Minimal stand-in for a ``Features`` object as consumed by ``_validate_filter_columns``.

    The validator only touches ``filter_engine()``, and (via ``applicable_filters``)
    ``filters`` and ``get_all_names()``. ``get_all_names`` mirrors the exact expression
    in ``BaseFilterEngine.applicable_filters`` so the single filter is guaranteed applicable.
    """

    def __init__(self, filters: list[SingleFilter]) -> None:
        self.filters = filters

    def filter_engine(self) -> type[PythonDictFilterEngine]:
        return PythonDictFilterEngine

    def get_all_names(self) -> set[Any]:
        return {sf.filter_feature.name for sf in self.filters}


class _FakeFeatureGroup:
    """Minimal stand-in for a FeatureGroup exposing the ``allow_empty_result`` policy.

    ``_validate_filter_columns`` reads ``feature_group.allow_empty_result()`` to decide
    whether to skip validation on zero-row data, and ``feature_group.get_class_name()``
    for error messages.
    """

    def __init__(self, allow_empty: bool) -> None:
        self._allow_empty = allow_empty

    def allow_empty_result(self) -> bool:
        return self._allow_empty

    def get_class_name(self) -> str:
        return "FakeFeatureGroup"


def _framework() -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


class TestPythonDictAllowEmptyResultPolicy:
    """The framework propagates emptiness unconditionally; the FG declares the policy."""

    def test_select_empty_data_returns_empty_list(self) -> None:
        """select on empty rows returns [] unconditionally."""
        result = _framework().select_data_by_column_names([], {FeatureName("col1")})
        assert result == []

    def test_select_missing_column_still_raises(self) -> None:
        """Non-empty data with an absent requested column still raises ValueError.

        The empty short-circuit must trigger only on empty ROWS, not on missing
        columns. With non-empty rows, identify_naming_convention raises a plain
        ValueError ('No columns found ...').
        """
        with pytest.raises(ValueError, match="No columns found"):
            _framework().select_data_by_column_names([{"a": 1}], {FeatureName("missing")})

    def test_transform_empty_list_returns_empty_list(self) -> None:
        """transform on [] returns [] unconditionally."""
        assert _framework().transform([], set()) == []

    def test_transform_none_returns_empty_list(self) -> None:
        """transform on None returns [] unconditionally (None is empty)."""
        assert _framework().transform(None, set()) == []

    def test_validate_filter_columns_empty_data_allow_empty_does_not_raise(self) -> None:
        """On zero-row data, validation is skipped when the FG allows empty results.

        The applicable filter targets a column 'age' that is absent (the data is
        empty). When ``feature_group.allow_empty_result()`` is True, validation
        returns early instead of raising the 'missing filter column' error.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup(allow_empty=True)

        # Must not raise.
        _framework()._validate_filter_columns([], features, feature_group)

    def test_validate_filter_columns_empty_data_default_raises(self) -> None:
        """On zero-row data, validation still raises when the FG forbids empty results.

        With ``feature_group.allow_empty_result()`` False, the absent filter column
        'age' triggers the 'missing filter column' error. Matching that exact message
        also proves the filter was applicable (the early-return path was not taken).
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup(allow_empty=False)

        with pytest.raises(ValueError, match="missing filter column"):
            _framework()._validate_filter_columns([], features, feature_group)

    def test_empty_result_error_is_value_error_subclass(self) -> None:
        """EmptyResultError remains a subclass of ValueError."""
        from mloda.core.abstract_plugins.compute_framework import EmptyResultError

        assert issubclass(EmptyResultError, ValueError)


class TestPythonDictEmptyResult(EmptyResultFrameworkTestMixin):
    """Test PythonDictFramework schema detection via the shared mixin.

    PythonDict's native data is a list of row dicts: empty is ``[]`` (no schema, state B),
    non-empty is a one-row list. ``empty_data_carries_schema`` is False so the mixin asserts
    ``_extract_column_names([])`` yields an empty set.
    """

    empty_data_carries_schema = False

    @pytest.fixture
    def framework_instance(self) -> Any:
        return _framework()

    @pytest.fixture
    def empty_data(self) -> Any:
        return []

    @pytest.fixture
    def non_empty_data(self) -> Any:
        return [{"a": 1}]
