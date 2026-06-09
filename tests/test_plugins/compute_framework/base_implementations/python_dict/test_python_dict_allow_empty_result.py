"""Unit tests for the FeatureGroup-declared ``allow_empty_result`` policy.

- ``PythonDictFramework.select_data_by_column_names`` is unconditionally
  empty-safe: on empty rows it returns ``[]``. It only propagates emptiness,
  it never judges it. (``transform`` emptiness is covered in
  ``test_python_dict_framework.py``.)
- The empty short-circuit triggers only on empty rows, never on missing columns:
  non-empty data with an absent requested column still raises ``ValueError``.
- ``ComputeFramework._extract_column_names(data)`` reports schema presence;
  ``PythonDictFramework`` returns the union of row keys (empty set for ``[]``).
- ``_validate_filter_columns`` skips the column-presence check ONLY for a
  schema-less EMPTY result, concretely ``isinstance(data, list) and not data``,
  regardless of ``allow_empty_result()``: emptiness judgement belongs solely to
  the output guard, and ``apply_filters([])`` is ``[]``. Any other data on which
  the framework sees no columns (filters run BEFORE transform, so raw
  pre-transform objects such as a columnar dict reach the validator) still
  raises, as does non-empty row data missing a filter column.
- ``EmptyResultError`` is a ``ValueError`` subclass and part of the public
  ``mloda.provider`` API.
"""

from typing import Any

import pytest

import mloda.provider

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

    def test_validate_filter_columns_empty_data_allow_empty_does_not_raise(self) -> None:
        """The skip is driven by the data being a schema-less empty list, not by the policy.

        The applicable filter targets a column 'age' that is absent (the data is
        the schema-less empty list ``[]``). Validation returns early because the
        data is an empty list; ``allow_empty_result()`` being True is irrelevant
        to the skip. Together with the default-policy twin test below, this pins
        policy-independence of the skip.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup(allow_empty=True)

        # Must not raise.
        _framework()._validate_filter_columns([], features, feature_group)

    def test_validate_filter_columns_empty_data_default_does_not_raise(self) -> None:
        """The skip is driven by the data being a schema-less empty list, not by the policy.

        Emptiness judgement belongs solely to the output guard, and
        ``apply_filters([])`` is ``[]``. Even with ``feature_group.allow_empty_result()``
        False, the schema-less empty list ``[]`` must not trigger the column-presence
        check: the ``allow_empty_result`` policy is irrelevant to the skip. Together
        with the allow-empty twin test above, this pins policy-independence of the skip.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup(allow_empty=False)

        # Must not raise.
        _framework()._validate_filter_columns([], features, feature_group)

    def test_validate_filter_columns_non_empty_data_missing_column_raises(self) -> None:
        """Non-empty data missing the filter column still raises, regardless of the policy.

        The schema-less skip applies only to zero-column data. With a present schema
        that lacks the filter column 'age', the 'missing filter column' error must
        fire even though ``allow_empty_result()`` is False.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup(allow_empty=False)

        with pytest.raises(ValueError, match="missing filter column"):
            _framework()._validate_filter_columns([{"other": 1}], features, feature_group)

    def test_validate_filter_columns_columnar_dict_missing_column_raises(self) -> None:
        """Raw pre-transform data the framework sees no columns on must still raise.

        Filters run BEFORE transform (run_calculation), so the validator can receive
        a raw NON-empty object the framework cannot read columns from: PythonDict's
        ``_extract_column_names`` only reads list-of-dict rows and returns an empty
        set for the columnar dict ``{"other": [1]}``. The skip applies ONLY to a
        schema-less empty result (``isinstance(data, list) and not data``); for any
        other no-columns-visible data, silently skipping hands raw data to the filter
        engine, which crashes confusingly. The loud 'missing filter column' ValueError
        must fire instead, regardless of ``allow_empty_result()``.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup(allow_empty=False)

        with pytest.raises(ValueError, match="missing filter column"):
            _framework()._validate_filter_columns({"other": [1]}, features, feature_group)

    def test_empty_result_error_is_value_error_subclass(self) -> None:
        """EmptyResultError remains a subclass of ValueError."""
        from mloda.core.abstract_plugins.compute_framework import EmptyResultError

        assert issubclass(EmptyResultError, ValueError)

    def test_empty_result_error_exported_from_provider(self) -> None:
        """EmptyResultError is part of the public ``mloda.provider`` API.

        It must be re-exported as the same object as the core definition
        (mirroring how ``DataTypeMismatchError`` is exported) and listed in
        ``mloda.provider.__all__``.
        """
        from mloda.core.abstract_plugins.compute_framework import EmptyResultError as core_empty_result_error
        from mloda.provider import EmptyResultError

        assert EmptyResultError is core_empty_result_error
        assert "EmptyResultError" in mloda.provider.__all__


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
