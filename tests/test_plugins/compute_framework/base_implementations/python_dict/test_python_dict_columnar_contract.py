"""Red-phase contract tests for the COLUMNAR PythonDict compute framework.

These tests pin the NEW native representation of ``PythonDictFramework``: a columnar
dict ``{"col": [v0, v1, ...]}`` where each top-level key is a column name and each value
is that column's list of cell values, all value-lists sharing one length (= row count).

They are expected to FAIL against the current row-wise (``list[dict]``) implementation and
to PASS once the columnar behavior is implemented. Contract points pinned (1-8) mirror the
task specification, plus the retirement of ``FeatureGroup.allow_empty_result()``.

Not to be confused with the legacy ``test_python_dict_framework.py`` which still encodes the
row-wise contract; that module is superseded by this one under the columnar model.
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import EmptyResultError
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    _EmptyResultMatchData,
)


def _framework() -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


# ---------------------------------------------------------------------------
# Point 3: expected_data_framework() must return dict (was list)
# ---------------------------------------------------------------------------
class TestExpectedDataFramework:
    def test_expected_data_framework_is_dict(self) -> None:
        """The native container type is ``dict`` (columnar), no longer ``list``."""
        assert PythonDictFramework.expected_data_framework() is dict


# ---------------------------------------------------------------------------
# Point 4: transform() normalizes accepted inputs to columnar storage
# ---------------------------------------------------------------------------
class TestTransformColumnar:
    def test_columnar_dict_passthrough(self) -> None:
        """A columnar dict with equal-length list values is returned as-is."""
        data = {"a": [1, 2], "b": ["x", "y"]}
        assert _framework().transform(data, set()) == {"a": [1, 2], "b": ["x", "y"]}

    def test_list_of_row_dicts_becomes_columnar(self) -> None:
        """A list of row dicts pivots into a columnar dict."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert _framework().transform(data, set()) == {"a": [1, 3], "b": [2, 4]}

    def test_single_row_list_becomes_columnar(self) -> None:
        """A single-element list of a row dict becomes a one-row columnar dict."""
        assert _framework().transform([{"a": 1}], set()) == {"a": [1]}

    def test_empty_list_becomes_empty_dict(self) -> None:
        """An empty list has no schema and normalizes to the schema-less ``{}``."""
        assert _framework().transform([], set()) == {}

    def test_empty_dict_passthrough(self) -> None:
        """``{}`` (zero columns) is already the schema-less empty; returned as ``{}``."""
        assert _framework().transform({}, set()) == {}

    def test_zero_row_columnar_dict_passthrough(self) -> None:
        """A columnar dict with an empty column keeps its schema (one known column)."""
        assert _framework().transform({"a": []}, set()) == {"a": []}

    def test_columnar_dict_unequal_lengths_raises(self) -> None:
        """Value-lists of differing length are invalid columnar input."""
        with pytest.raises(ValueError):
            _framework().transform({"a": [1, 2], "b": [3]}, set())

    def test_dict_with_non_list_values_raises(self) -> None:
        """A dict whose values are not lists is not columnar and is rejected.

        Single-row dicts are no longer special-cased under the columnar model.
        """
        with pytest.raises(ValueError):
            _framework().transform({"a": 1, "b": 2}, set())

    def test_none_raises(self) -> None:
        """``None`` is a missing result, not an empty one: unsupported type."""
        with pytest.raises(ValueError, match="Data type .* is not supported"):
            _framework().transform(None, set())

    def test_int_raises(self) -> None:
        """A bare int is an unsupported type."""
        with pytest.raises(ValueError, match="Data type .* is not supported"):
            _framework().transform(5, set())


# ---------------------------------------------------------------------------
# Points 2 & 5: schema = top-level keys, present even at zero rows
# ---------------------------------------------------------------------------
class TestExtractColumnNames:
    def test_multi_column_zero_rows(self) -> None:
        """Zero-row columnar dict still exposes all its column names."""
        assert _framework()._extract_column_names({"a": [], "b": []}) == {"a", "b"}

    def test_populated_columns(self) -> None:
        assert _framework()._extract_column_names({"a": [1, 2], "b": [3, 4]}) == {"a", "b"}

    def test_single_empty_column_is_schema_bearing(self) -> None:
        """``{"col_a": []}`` is a valid schema-bearing empty (one known column)."""
        assert _framework()._extract_column_names({"col_a": []}) == {"col_a"}

    def test_empty_dict_has_no_columns(self) -> None:
        assert _framework()._extract_column_names({}) == set()


# ---------------------------------------------------------------------------
# Point 6: _is_schemaless_empty is True ONLY for {}
# ---------------------------------------------------------------------------
class TestIsSchemalessEmpty:
    def test_empty_dict_is_schemaless(self) -> None:
        """``{}`` (zero columns) is the only schema-less value."""
        assert _framework()._is_schemaless_empty({}) is True

    def test_single_empty_column_is_not_schemaless(self) -> None:
        """``{"a": []}`` carries a schema (one column) and is NOT schema-less."""
        assert _framework()._is_schemaless_empty({"a": []}) is False

    def test_populated_columnar_is_not_schemaless(self) -> None:
        assert _framework()._is_schemaless_empty({"a": [1, 2]}) is False

    def test_empty_list_is_not_schemaless(self) -> None:
        """Only ``{}`` is schema-less; a bare ``[]`` is not the native empty."""
        assert _framework()._is_schemaless_empty([]) is False


# ---------------------------------------------------------------------------
# Point 7: column dtype / data-type read the column list
# ---------------------------------------------------------------------------
class TestExtractColumnTypes:
    def test_data_type_int(self) -> None:
        assert _framework()._extract_column_data_type({"i": [1, 2, 3]}, "i") == DataType.INT64

    def test_data_type_string(self) -> None:
        assert _framework()._extract_column_data_type({"s": ["a", "b"]}, "s") == DataType.STRING

    def test_data_type_float(self) -> None:
        assert _framework()._extract_column_data_type({"f": [1.0, 2.0]}, "f") == DataType.DOUBLE

    def test_data_type_bool(self) -> None:
        assert _framework()._extract_column_data_type({"b": [True, False]}, "b") == DataType.BOOLEAN

    def test_data_type_first_non_none_wins(self) -> None:
        """The first non-None value in the column list determines the type."""
        assert _framework()._extract_column_data_type({"i": [None, 2, 3]}, "i") == DataType.INT64

    def test_data_type_all_none_is_unknown(self) -> None:
        """An all-None column yields an unknown (None) data type."""
        assert _framework()._extract_column_data_type({"i": [None, None]}, "i") is None

    def test_data_type_empty_column_is_unknown_but_present(self) -> None:
        """An empty column has an unknown type yet still counts as a present column."""
        fw = _framework()
        assert fw._extract_column_data_type({"i": []}, "i") is None
        assert "i" in fw._extract_column_names({"i": []})

    def test_dtype_int(self) -> None:
        assert _framework()._extract_column_dtype({"i": [1, 2]}, "i") == "int"

    def test_dtype_string(self) -> None:
        assert _framework()._extract_column_dtype({"s": ["a"]}, "s") == "str"

    def test_dtype_first_non_none_wins(self) -> None:
        assert _framework()._extract_column_dtype({"i": [None, 7]}, "i") == "int"

    def test_dtype_all_none_is_none(self) -> None:
        assert _framework()._extract_column_dtype({"i": [None, None]}, "i") is None


# ---------------------------------------------------------------------------
# Point 8: select_data_by_column_names operates columnar
# ---------------------------------------------------------------------------
class TestSelectColumnar:
    def test_select_subset_preserves_rows(self) -> None:
        """Selection returns a columnar dict of only the requested columns."""
        data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        result = _framework().select_data_by_column_names(data, {FeatureName("a"), FeatureName("b")})
        assert result == {"a": [1, 2], "b": [3, 4]}

    def test_select_empty_dict_returns_empty_dict(self) -> None:
        """Selecting from ``{}`` yields ``{}``."""
        result = _framework().select_data_by_column_names({}, {FeatureName("a")})
        assert result == {}

    def test_select_zero_row_columns_kept(self) -> None:
        """Selecting a present-but-empty column keeps the column at zero rows."""
        result = _framework().select_data_by_column_names({"a": [], "b": []}, {FeatureName("a")})
        assert result == {"a": []}


# ---------------------------------------------------------------------------
# Retirement of FeatureGroup.allow_empty_result()
# ---------------------------------------------------------------------------
class TestAllowEmptyResultRetired:
    def test_feature_group_has_no_allow_empty_result(self) -> None:
        """``allow_empty_result`` is removed entirely from the FeatureGroup API."""
        assert not hasattr(FeatureGroup, "allow_empty_result")


class _ColumnarZeroRowFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FG returning a schema-bearing zero-row columnar dict, with NO opt-in flag."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"columnar_zero_row_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"columnar_zero_row_col": []}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"columnar_zero_row_col"}


class _ColumnarZeroColumnFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FG returning a schema-less ``{}`` (zero columns), with NO opt-in flag."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"columnar_zero_column_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"columnar_zero_column_col"}


_ENABLED_ZERO_ROW = PluginCollector.enabled_feature_groups({_ColumnarZeroRowFeatureGroup})
_ENABLED_ZERO_COLUMN = PluginCollector.enabled_feature_groups({_ColumnarZeroColumnFeatureGroup})


def test_zero_row_with_columns_succeeds_without_flag(flight_server: Any) -> None:
    """A zero-row result WITH a column succeeds on PythonDict without any opt-in.

    Under the columnar model ``transform`` keeps ``{"col": []}`` (schema-bearing), so the
    output guard passes and the run returns the columnar dict with zero rows.
    """
    result = mloda.run_all(
        [Feature(name="columnar_zero_row_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED_ZERO_ROW,
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
    )

    assert len(result) == 1
    assert result[0] == {"columnar_zero_row_col": []}


def test_zero_column_result_raises_without_flag(flight_server: Any) -> None:
    """A zero-column ``{}`` result raises ``EmptyResultError`` with NO opt-in."""
    with pytest.raises(Exception) as excinfo:
        mloda.run_all(
            [Feature(name="columnar_zero_column_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ZERO_COLUMN,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )
    assert isinstance(excinfo.value, EmptyResultError)
