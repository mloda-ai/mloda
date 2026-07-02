"""Unit tests for the empty-result policy on the COLUMNAR PythonDict framework.

- ``PythonDictFramework.select_data_by_column_names`` is unconditionally empty-safe:
  on the schema-less ``{}`` it returns ``{}``. It only propagates emptiness, it never
  judges it. (``transform`` emptiness is covered in ``test_python_dict_framework.py``
  and ``test_python_dict_columnar_contract.py``.)
- The empty short-circuit triggers only on the schema-less ``{}``, never on missing
  columns: non-empty data with an absent requested column still raises ``ValueError``.
- ``ComputeFramework._extract_column_names(data)`` reports schema presence;
  ``PythonDictFramework`` returns the columnar dict's keys (empty set only for ``{}``).
- ``_validate_filter_columns`` skips the column-presence check ONLY for the framework's
  schema-less EMPTY result, as judged by the framework-owned hook ``_is_schemaless_empty``
  (for PythonDict: only ``{}``). Any other data on which the framework sees the filter
  column missing still raises.
- ``EmptyResultError`` is a ``ValueError`` subclass and part of the public
  ``mloda.provider`` API.
"""

from typing import Any

import pyarrow as pa
import pytest

import mloda.provider

from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
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
    """Minimal stand-in for a FeatureGroup exposing ``get_class_name`` for error messages."""

    def get_class_name(self) -> str:
        return "FakeFeatureGroup"


def _framework() -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


class TestPythonDictEmptyResultPolicy:
    """The framework propagates emptiness unconditionally; the guard judges it elsewhere."""

    def test_select_empty_data_returns_empty_dict(self) -> None:
        """select on the schema-less ``{}`` returns ``{}`` unconditionally."""
        result = _framework().select_data_by_column_names({}, {FeatureName("col1")})
        assert result == {}

    def test_select_missing_column_still_raises(self) -> None:
        """Non-empty data with an absent requested column still raises ValueError.

        The empty short-circuit must trigger only on the schema-less ``{}``, not on a
        missing column. With a present schema, identify_naming_convention raises a plain
        ValueError ('No columns found ...').
        """
        with pytest.raises(ValueError, match="No columns found"):
            _framework().select_data_by_column_names({"a": [1]}, {FeatureName("missing")})

    def test_validate_filter_columns_empty_dict_does_not_raise(self) -> None:
        """The skip is driven by the data being the schema-less ``{}``.

        The applicable filter targets a column 'age' that is absent (the data is the
        schema-less ``{}``). Validation returns early because the data is schema-less.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup()

        # Must not raise.
        _framework()._validate_filter_columns({}, features, feature_group)

    def test_validate_filter_columns_non_empty_data_missing_column_raises(self) -> None:
        """Non-empty data missing the filter column still raises.

        The schema-less skip applies only to the zero-column ``{}``. With a present schema
        that lacks the filter column 'age', the 'missing filter column' error must fire.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup()

        with pytest.raises(ValueError, match="missing filter column"):
            _framework()._validate_filter_columns({"other": [1]}, features, feature_group)

    def test_validate_filter_columns_zero_row_schema_missing_column_raises(self) -> None:
        """A schema-bearing zero-row frame missing the filter column still raises.

        ``{"other": []}`` carries a schema (one column) and is NOT the schema-less empty,
        so the loud 'missing filter column' ValueError fires for the absent 'age' column.
        """
        single_filter = SingleFilter("age", FilterType.MIN, {"value": 30})
        features = _FakeFeaturesWithFilter([single_filter])
        feature_group = _FakeFeatureGroup()

        with pytest.raises(ValueError, match="missing filter column"):
            _framework()._validate_filter_columns({"other": []}, features, feature_group)

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


class TestIsSchemalessEmptyHook:
    """Pins the ``_is_schemaless_empty(data) -> bool`` hook on ComputeFramework.

    The filter-column skip in ``_validate_filter_columns`` must be driven by a
    framework-owned predicate. Under the columnar model PythonDict has exactly one
    representational schema-less empty: ``{}`` (zero columns). Everything else, including
    a schema-bearing zero-row frame ``{"a": []}`` and falsy non-container garbage, is NOT a
    schema-less empty. The base-class default returns False.
    """

    def test_python_dict_empty_dict_is_schemaless_empty(self) -> None:
        """``{}`` is PythonDict's only schema-less empty form: True."""
        assert _framework()._is_schemaless_empty({}) is True

    def test_python_dict_zero_row_column_is_not_schemaless_empty(self) -> None:
        """``{"a": []}`` carries a schema (one column) and is not schema-less: False."""
        assert _framework()._is_schemaless_empty({"a": []}) is False

    def test_python_dict_non_empty_data_is_not_schemaless_empty(self) -> None:
        """Non-empty columnar data carries a schema and is not empty: False."""
        assert _framework()._is_schemaless_empty({"a": [1]}) is False

    def test_python_dict_empty_list_is_not_schemaless_empty(self) -> None:
        """Only ``{}`` is schema-less; a bare ``[]`` is not the native empty: False."""
        assert _framework()._is_schemaless_empty([]) is False

    @pytest.mark.parametrize("garbage", [0, False, ""])
    def test_python_dict_falsy_non_container_values_are_not_schemaless_empty(self, garbage: Any) -> None:
        """Falsy non-container values are unsupported garbage, not empty results: False."""
        assert _framework()._is_schemaless_empty(garbage) is False

    def test_base_class_default_is_false_even_for_schemaless_data(self) -> None:
        """The base-class default returns False, even for zero-column data.

        Only PythonDict's representational empty gets the skip. A zero-column Arrow table is
        schema-less (``_extract_column_names`` yields an empty set) but PyArrowTable does not
        override the hook, so the loud filter-column check must still fire for it.
        """
        pyarrow_framework = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert pyarrow_framework._is_schemaless_empty(pa.table({})) is False


class TestPythonDictEmptyResult(EmptyResultFrameworkTestMixin):
    """Test PythonDictFramework schema detection via the shared mixin.

    PythonDict's native data is a columnar dict. A schema-bearing zero-row frame keeps its
    columns (``{"a": []}`` -> ``{"a"}``), so ``empty_data_carries_schema`` is True, matching
    every other backend.
    """

    empty_data_carries_schema = True

    @pytest.fixture
    def framework_instance(self) -> Any:
        return _framework()

    @pytest.fixture
    def empty_data(self) -> Any:
        return {"a": []}

    @pytest.fixture
    def non_empty_data(self) -> Any:
        return {"a": [1]}
