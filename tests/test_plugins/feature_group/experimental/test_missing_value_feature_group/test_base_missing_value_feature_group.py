import math
import statistics
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import FeatureSet, SubtypeDeclaration
from tests.test_plugins.feature_group.experimental.reference_contract import same, same_on_pandas
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys

from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda.provider import FeatureChainParser
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueImputation
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow import PyArrowMissingValueImputation
from mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict import (
    PythonDictMissingValueImputation,
)

IMPLEMENTATIONS = [PandasMissingValueImputation, PyArrowMissingValueImputation, PythonDictMissingValueImputation]
FRAMEWORKS = [PandasDataFrame, PyArrowTable, PythonDictFramework]
IMPL_DATA = [
    (PandasMissingValueImputation, lambda: pd.DataFrame({"b": [1.0], "a": [2.0]})),
    (PyArrowMissingValueImputation, lambda: pa.table({"b": [1.0], "a": [2.0]})),
    (PythonDictMissingValueImputation, lambda: {"b": [1.0], "a": [2.0]}),
]
IMPL_IDS = ["pandas", "pyarrow", "python_dict"]
SOURCE_ERROR = "Source features not found in data: {missing}. Available columns: ['a', 'b']"


def _feature_set(name: str, **context: Any) -> FeatureSet:
    feature_set = FeatureSet()
    feature_set.add(Feature(name, options=Options(context=context)))
    return feature_set


class TestMissingValueFeatureGroup:
    """Tests for the MissingValueFeatureGroup class."""

    def test_extract_config_feature_with_double_underscore_name(self) -> None:
        options = Options(
            context={
                MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                DefaultOptionKeys.in_features: frozenset([Feature("income")]),
            }
        )
        result = MissingValueFeatureGroup._extract_imputation_method_and_source_feature(
            Feature("a__b", options=options)
        )
        assert result == ("mean", "income")

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Test valid feature names
        feature_name = "income__mean_imputed"

        # Test that the PREFIX_PATTERN is correctly defined
        assert hasattr(MissingValueFeatureGroup, "PREFIX_PATTERN")

        # Test that FeatureChainParser methods work with the PREFIX_PATTERN
        assert FeatureChainParser.extract_in_feature(feature_name, MissingValueFeatureGroup.PREFIX_PATTERN) == "income"

    def test_get_imputation_method(self) -> None:
        """Test extraction of imputation method from feature name."""
        assert MissingValueFeatureGroup.get_imputation_method("income__mean_imputed") == "mean"
        assert MissingValueFeatureGroup.get_imputation_method("age__median_imputed") == "median"
        assert MissingValueFeatureGroup.get_imputation_method("category__mode_imputed") == "mode"
        assert MissingValueFeatureGroup.get_imputation_method("status__constant_imputed") == "constant"
        assert MissingValueFeatureGroup.get_imputation_method("temperature__ffill_imputed") == "ffill"
        assert MissingValueFeatureGroup.get_imputation_method("humidity__bfill_imputed") == "bfill"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            MissingValueFeatureGroup.get_imputation_method("invalid_feature_name")

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert MissingValueFeatureGroup.match_feature_group_criteria("income__mean_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("age__median_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("category__mode_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("status__constant_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("temperature__ffill_imputed", options)
        assert MissingValueFeatureGroup.match_feature_group_criteria("humidity__bfill_imputed", options)

        # Test with FeatureName objects
        assert MissingValueFeatureGroup.match_feature_group_criteria(FeatureName("income__mean_imputed"), options)
        assert MissingValueFeatureGroup.match_feature_group_criteria(FeatureName("age__median_imputed"), options)

        # Test with invalid feature names
        assert not MissingValueFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not MissingValueFeatureGroup.match_feature_group_criteria("mean_filled_income", options)
        assert not MissingValueFeatureGroup.match_feature_group_criteria("unknown_imputed_income", options)

    def test_bogus_name_method_is_rejected_despite_a_valid_option(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options(context={MissingValueFeatureGroup.IMPUTATION_METHOD: "mean"})
        assert MissingValueFeatureGroup.match_feature_group_criteria("x__bogus_imputed", options) is False
        assert len(rejection_window) == 1
        assert "bogus" in next(iter(rejection_window.values())).reason

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = MissingValueFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("income__mean_imputed"))
        assert input_features == {Feature("income")}

        input_features = feature_group.input_features(options, FeatureName("age__median_imputed"))
        assert input_features == {Feature("age")}

        input_features = feature_group.input_features(options, FeatureName("category__mode_imputed"))
        assert input_features == {Feature("category")}

        input_features = feature_group.input_features(options, FeatureName("status__constant_imputed"))
        assert input_features == {Feature("status")}

        input_features = feature_group.input_features(options, FeatureName("temperature__ffill_imputed"))
        assert input_features == {Feature("temperature")}

        input_features = feature_group.input_features(options, FeatureName("humidity__bfill_imputed"))
        assert input_features == {Feature("humidity")}

    @pytest.mark.parametrize(
        ("group_by", "expected"),
        [
            (["region"], {"income", "region"}),
            (["region", "store"], {"income", "region", "store"}),
            (None, {"income"}),
            ([], {"income"}),
            (["income"], {"income"}),
        ],
    )
    def test_input_features_declare_group_by_columns(self, group_by: Any, expected: set[str]) -> None:
        options = Options(context={"group_by_features": group_by})
        result = MissingValueFeatureGroup().input_features(options, FeatureName("income__mean_imputed"))
        assert result is not None
        assert {str(f.name) for f in result} == expected
        assert len(result) == len(expected)

    @pytest.mark.parametrize(
        ("source", "source_name", "expected_group", "expected_context"),
        [
            pytest.param(
                Feature("income", Options(group={"scope": "a"})), "income", {"scope": "a"}, {}, id="scoped_source"
            ),
            pytest.param(Feature("income"), "income", {}, {}, id="plain_source"),
            pytest.param(
                Feature(
                    "income_sum",
                    Options(group={"aggregation_type": "sum", DefaultOptionKeys.in_features: "income"}),
                ),
                "income_sum",
                {"aggregation_type": "sum"},
                {},
                id="chained_source",
            ),
            pytest.param(
                Feature(
                    "income_sum",
                    Options(
                        context={DefaultOptionKeys.in_features: "income", "tenant": "t1"},
                        propagate_context_keys=frozenset({"tenant"}),
                    ),
                ),
                "income_sum",
                {},
                {"tenant": "t1"},
                id="context_chained_source_pushing_context_key",
            ),
        ],
    )
    def test_group_by_features_carry_only_forwarded_group_options_of_source(
        self,
        source: Feature,
        source_name: str,
        expected_group: dict[str, Any],
        expected_context: dict[str, Any],
    ) -> None:
        options = Options(
            group={"imputation_method": "mean"},
            context={DefaultOptionKeys.in_features: frozenset([source]), "group_by_features": ["region"]},
        )
        result = MissingValueFeatureGroup().input_features(options, FeatureName("x"))
        assert result is not None
        by_name = {str(f.name): f for f in result}
        assert set(by_name) == {source_name, "region"}
        assert dict(by_name["region"].options.group) == expected_group
        assert DefaultOptionKeys.in_features not in by_name["region"].options.group
        assert dict(by_name["region"].options.context) == expected_context
        if not expected_group and not expected_context:
            assert by_name["region"] == Feature("region")

    def test_python_dict_calculate_rejects_empty_data_before_the_group(self) -> None:
        group = Mock()
        with pytest.raises(ValueError, match="Data cannot be empty"):
            PythonDictMissingValueImputation.calculate(group, {}, _feature_set("income__mean_imputed"), None)
        group.impute.assert_not_called()

    @pytest.mark.parametrize(("impl", "make_data"), IMPL_DATA, ids=IMPL_IDS)
    def test_impute_source_check_comes_first_and_lists_sorted_columns(self, impl: Any, make_data: Any) -> None:
        feature_set = _feature_set("missing__mean_imputed", group_by_features=["g1", "g2"])
        with pytest.raises(ValueError) as exc_info:
            impl.calculate(MissingValueFeatureGroup, make_data(), feature_set, None)
        assert str(exc_info.value) == SOURCE_ERROR.format(missing=["missing"])

    @pytest.mark.parametrize(("impl", "make_data"), IMPL_DATA, ids=IMPL_IDS)
    def test_impute_names_one_missing_group_by_feature_per_error(self, impl: Any, make_data: Any) -> None:
        feature_set = _feature_set("a__mean_imputed", group_by_features=["g1", "g2"])
        with pytest.raises(ValueError) as exc_info:
            impl.calculate(MissingValueFeatureGroup, make_data(), feature_set, None)
        assert str(exc_info.value) == SOURCE_ERROR.format(missing=["g1"])

    @pytest.mark.parametrize(("impl", "make_data"), IMPL_DATA, ids=IMPL_IDS)
    def test_impute_checks_the_constant_after_the_columns(self, impl: Any, make_data: Any) -> None:
        with pytest.raises(ValueError, match="Constant value must be provided for constant imputation method"):
            impl.calculate(MissingValueFeatureGroup, make_data(), _feature_set("a__constant_imputed"), None)
        with pytest.raises(ValueError, match="Source features not found in data"):
            impl.calculate(MissingValueFeatureGroup, make_data(), _feature_set("zzz__constant_imputed"), None)

    def test_impute_reports_all_missing_resolved_names_in_one_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolved = Mock(return_value=["x~0", "x~1", "a"])
        monkeypatch.setattr(MissingValueFeatureGroup, "resolve_multi_column_feature", resolved)
        perform = Mock()
        add_column = Mock()
        with pytest.raises(ValueError) as exc_info:
            MissingValueFeatureGroup.impute(
                object(),
                _feature_set("x__mean_imputed"),
                perform=perform,
                add_column=add_column,
                columns=Mock(return_value={"b", "a"}),
            )
        assert str(exc_info.value) == SOURCE_ERROR.format(missing=["x~0", "x~1"])
        perform.assert_not_called()
        add_column.assert_not_called()

    def test_impute_hands_the_resolved_arguments_to_perform_and_add_column(self) -> None:
        data = object()
        perform = Mock(return_value="result")
        add_column = Mock(return_value="done")
        feature_set = _feature_set("a__constant_imputed", constant_value=0, group_by_features=["b"])

        result = MissingValueFeatureGroup.impute(
            data, feature_set, perform=perform, add_column=add_column, columns=Mock(return_value={"a", "b"})
        )

        perform.assert_called_once_with(data, "constant", ["a"], 0, ["b"])
        add_column.assert_called_once_with(data, "a__constant_imputed", "result")
        assert result == "done"


METHOD_UNIVERSE = frozenset(MissingValueFeatureGroup.IMPUTATION_METHODS)


class TestMissingValueSubtypeDeclaration:
    """MissingValueFeatureGroup declares its imputation methods via SUBTYPES."""

    def test_declaration_key_is_imputation_method(self) -> None:
        declaration = MissingValueFeatureGroup.SUBTYPES
        assert isinstance(declaration, SubtypeDeclaration)
        assert declaration.key == "imputation_method"

    def test_universe_matches_declared_methods(self) -> None:
        assert MissingValueFeatureGroup.subtype_universe() == METHOD_UNIVERSE

    def test_resolves_from_feature_name(self) -> None:
        assert MissingValueFeatureGroup.SUBTYPES is not None
        assert MissingValueFeatureGroup.resolve_subtype("income__mean_imputed", Options()) == "mean"

    def test_resolves_from_options_path(self) -> None:
        assert MissingValueFeatureGroup.SUBTYPES is not None
        options = Options(context={MissingValueFeatureGroup.IMPUTATION_METHOD: "median"})
        assert MissingValueFeatureGroup.resolve_subtype("imputation_placeholder", options) == "median"

    @pytest.mark.parametrize("framework", FRAMEWORKS, ids=IMPL_IDS)
    def test_every_method_stays_supported(self, framework: Any) -> None:
        assert MissingValueFeatureGroup.subtype_universe() == METHOD_UNIVERSE
        for method in sorted(METHOD_UNIVERSE):
            supported = MissingValueFeatureGroup.supports_compute_framework(
                f"income__{method}_imputed", Options(), framework
            )
            assert supported is True, f"'{method}' must stay supported on {framework.__name__}"


_NAN = float("nan")
_Cells = list[Any]
_Columns = dict[str, _Cells]


def _missing(cell: Any) -> bool:
    return cell is None or (isinstance(cell, float) and math.isnan(cell))


def _statistic(method: str, cells: _Cells) -> Any:
    """The contract statistic over the present cells: mode ties take the smallest value."""
    present = [cell for cell in cells if not _missing(cell)]
    if not present:
        return None
    if method == "mean":
        return statistics.fmean(present)
    if method == "median":
        return statistics.median(present)
    counts = {value: present.count(value) for value in present}
    return min(value for value, count in counts.items() if count == max(counts.values()))


def _reference_fill(method: str, cells: _Cells, constant: Any) -> _Cells:
    """Fill missing cells of one column (or one group) by the contract."""
    if method == "constant":
        return [constant if _missing(cell) else cell for cell in cells]
    if method in ("ffill", "bfill"):
        order = range(len(cells)) if method == "ffill" else range(len(cells) - 1, -1, -1)
        result = list(cells)
        last: Any = None
        for i in order:
            if _missing(result[i]):
                result[i] = last
            else:
                last = result[i]
        return result
    fill = _statistic(method, cells)
    return [fill if _missing(cell) else cell for cell in cells]


def _reference_single(method: str, columns: _Columns, constant: Any) -> _Cells:
    return _reference_fill(method, columns["a"], constant)


def _reference_multi(method: str, columns: _Columns, constant: Any) -> _Cells:
    """Row-wise: the statistic across the columns of each row, skipping missing cells."""
    names = list(columns)
    rows = range(len(columns[names[0]]))
    if method == "constant":
        return [constant for _ in rows]
    return [_statistic(method, [columns[name][row] for name in names]) for row in rows]


def _reference_grouped(method: str, columns: _Columns, constant: Any) -> _Cells:
    """Per group fill; a constant fills everywhere; a group without a statistic falls back to the overall one."""
    values = columns["a"]
    if method == "constant":
        return _reference_fill(method, values, constant)
    result = list(values)
    keys = columns["g"]
    for key in dict.fromkeys(keys):
        indices = [i for i, k in enumerate(keys) if k == key]
        group = [values[i] for i in indices]
        if method in ("mean", "median", "mode") and _statistic(method, group) is None:
            filled = _reference_fill(method, group, constant)
            overall = _statistic(method, values)
            filled = [overall if _missing(cell) else cell for cell in filled]
        else:
            filled = _reference_fill(method, group, constant)
        for i, cell in zip(indices, filled):
            result[i] = cell
    return result


def _pandas_data(columns: _Columns) -> pd.DataFrame:
    def column(cells: _Cells) -> _Cells:
        numeric = any(isinstance(cell, float) for cell in cells)
        return [np.nan if numeric and cell is None else cell for cell in cells]

    return pd.DataFrame({name: column(cells) for name, cells in columns.items()})


def _pyarrow_data(columns: _Columns) -> pa.Table:
    return pa.table({name: pa.array(cells) for name, cells in columns.items()})


def _python_dict_data(columns: _Columns) -> dict[str, _Cells]:
    return {name: list(cells) for name, cells in columns.items()}


_MAKE_DATA: dict[type[Implementation[Any]], Callable[[_Columns], Any]] = {
    PandasMissingValueImputation: _pandas_data,
    PyArrowMissingValueImputation: _pyarrow_data,
    PythonDictMissingValueImputation: _python_dict_data,
}


def _to_list(result: Any) -> _Cells:
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return list(result.to_pylist())
    return list(result.tolist()) if hasattr(result, "tolist") else list(result)


def _impute(
    implementation: Any,
    columns: _Columns,
    method: str,
    in_features: list[str],
    constant: Any,
    group_by: list[str] | None,
) -> _Cells:
    data = _MAKE_DATA[implementation](columns)
    return _to_list(implementation._perform_imputation(data, method, in_features, constant, group_by))


def _differences(implementation: Any, actual: _Cells, expected: _Cells) -> list[str]:
    equal = same_on_pandas if implementation is PandasMissingValueImputation else same
    assert len(actual) == len(expected), f"{actual} vs {expected}"
    return [f"row {i}: {a!r} != reference {e!r}" for i, (a, e) in enumerate(zip(actual, expected)) if not equal(a, e)]


# Methods are chosen so the filled values differ between a NaN-skipping and a NaN-keeping implementation.
_SINGLE: _Columns = {"a": [3.0, _NAN, 1.0, 3.0, None, 1.0, 5.0]}
_STRINGS: _Columns = {"a": ["b", None, "a", "b", "a"]}
_MULTI: _Columns = {
    "a": [3.0, _NAN, None, 2.0, 4.0],
    "b": [3.0, 2.0, 6.0, 1.0, None],
    "c": [_NAN, 8.0, None, None, 4.0],
}
_GROUPED: _Columns = {
    "a": [4.0, _NAN, 7.0, 2.0, None, 4.0, 2.0, _NAN, 9.0, 5.0, None, _NAN, None],
    "g": ["x", "x", "y", "x", "y", "x", "x", "x", "y", "x", "z", "z", "w"],
}
_CONSTANT = 0.5
_ALL_METHODS = sorted(METHOD_UNIVERSE)
_MULTI_METHODS = [method for method in _ALL_METHODS if method not in ("ffill", "bfill")]


_MIXED_TIE = [2.0, 1, 2.0, 1, None]
_MIXED_MULTI_INT = [1, 3]
_MIXED_MULTI_FLOAT = [2.0, 3.0]


class TestImplementationsConformToTheReferenceContract:
    """Null and NaN are missing, statistics skip them, missing cells are filled, mode ties take the smallest."""

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_declares_no_divergence(self, implementation: type[Implementation[Any]]) -> None:
        assert dict(implementation.divergent) == {}

    @pytest.mark.parametrize("method", _ALL_METHODS)
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_single_float_column(self, implementation: Any, method: str) -> None:
        actual = _impute(implementation, _SINGLE, method, ["a"], _CONSTANT, None)
        assert not _differences(implementation, actual, _reference_single(method, _SINGLE, _CONSTANT))

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_mode_tie_of_int_and_float_values_takes_the_smallest(self, implementation: Any) -> None:
        cells: _Cells = [2.0, 1.0, 2.0, 1.0, None]
        if implementation is PythonDictMissingValueImputation:
            cells = list(_MIXED_TIE)
        actual = _impute(implementation, {"a": cells}, "mode", ["a"], None, None)
        assert not _differences(implementation, actual, [2.0, 1.0, 2.0, 1.0, 1.0])

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_multi_column_mode_tie_of_int_and_float_columns(self, implementation: Any) -> None:
        columns: _Columns = {"a": [1.0, 3.0], "b": list(_MIXED_MULTI_FLOAT)}
        if implementation is PythonDictMissingValueImputation:
            columns = {"a": list(_MIXED_MULTI_INT), "b": list(_MIXED_MULTI_FLOAT)}
        actual = _impute(implementation, columns, "mode", ["a", "b"], None, None)
        assert not _differences(implementation, actual, [1.0, 3.0])

    @pytest.mark.parametrize("method", ["mode", "constant", "ffill", "bfill"])
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_string_column_with_a_mode_tie(self, implementation: Any, method: str) -> None:
        actual = _impute(implementation, _STRINGS, method, ["a"], "z", None)
        assert not _differences(implementation, actual, _reference_single(method, _STRINGS, "z"))

    @pytest.mark.parametrize("method", _MULTI_METHODS)
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_multi_column_feature(self, implementation: Any, method: str) -> None:
        actual = _impute(implementation, _MULTI, method, ["a", "b", "c"], _CONSTANT, None)
        assert not _differences(implementation, actual, _reference_multi(method, _MULTI, _CONSTANT))

    @pytest.mark.parametrize("method", ["ffill", "bfill"])
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_multi_column_fill_methods_raise(self, implementation: Any, method: str) -> None:
        with pytest.raises(ValueError):
            _impute(implementation, _MULTI, method, ["a", "b", "c"], _CONSTANT, None)

    @pytest.mark.parametrize("method", _ALL_METHODS)
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_grouped_imputation(self, implementation: Any, method: str) -> None:
        actual = _impute(implementation, _GROUPED, method, ["a"], _CONSTANT, ["g"])
        assert not _differences(implementation, actual, _reference_grouped(method, _GROUPED, _CONSTANT))


class TestNanNormalizationIsLimitedToFloatColumns:
    """Only floating source columns have NaN treated as missing; ints, strings and group keys are untouched."""

    @pytest.mark.parametrize("method", ["mode", "constant"])
    @pytest.mark.parametrize("implementation", [PyArrowMissingValueImputation, PythonDictMissingValueImputation])
    def test_int_column_with_nulls_keeps_its_values(self, implementation: Any, method: str) -> None:
        columns: _Columns = {"a": [1, None, 1, 3, 3, 0]}
        actual = _impute(implementation, columns, method, ["a"], 7, None)
        expected = _reference_single(method, columns, 7)
        assert actual == expected
        assert all(isinstance(cell, int) for cell in actual)

    def test_pyarrow_int_column_keeps_its_type(self) -> None:
        table = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        result = PyArrowMissingValueImputation._perform_imputation(table, "constant", ["a"], 7, None)
        assert result.type == pa.int64()

    @pytest.mark.parametrize("implementation", [PyArrowMissingValueImputation, PythonDictMissingValueImputation])
    def test_string_column_keeps_its_values(self, implementation: Any) -> None:
        columns: _Columns = {"a": ["nan", None, "NaN", "nan"]}
        actual = _impute(implementation, columns, "mode", ["a"], None, None)
        assert actual == ["nan", "nan", "NaN", "nan"]

    @pytest.mark.parametrize("method", ["mean", "mode", "ffill"])
    @pytest.mark.parametrize("implementation", [PyArrowMissingValueImputation, PythonDictMissingValueImputation])
    def test_float_group_keys_are_never_altered(self, implementation: Any, method: str) -> None:
        values: _Cells = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, None]
        float_keys: _Columns = {"a": values, "g": [1.0, 1.0, float("nan"), float("nan"), None, None, 2.0, 2.0]}
        string_keys: _Columns = {"a": values, "g": ["one", "one", "nan", "nan", "null", "null", "two", "two"]}
        with_float_keys = _impute(implementation, float_keys, method, ["a"], None, ["g"])
        with_string_keys = _impute(implementation, string_keys, method, ["a"], None, ["g"])
        assert not _differences(implementation, with_float_keys, with_string_keys)

    def test_python_dict_input_is_not_mutated(self) -> None:
        data = _python_dict_data(_GROUPED)
        PythonDictMissingValueImputation._perform_imputation(data, "mean", ["a"], None, ["g"])
        assert data["g"] == _GROUPED["g"]
        assert all(same(a, e) for a, e in zip(data["a"], _GROUPED["a"]))
