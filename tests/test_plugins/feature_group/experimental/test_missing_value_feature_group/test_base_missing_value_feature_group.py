import datetime
from decimal import Decimal
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
    return cell is None or (np.ndim(cell) == 0 and bool(pd.isna(cell)))


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


def _reference_grouped(method: str, columns: _Columns, constant: Any, group_by: list[str] | None = None) -> _Cells:
    """Per group fill; rows with a missing key part join no group; no group statistic falls back to the overall one."""
    values = columns["a"]
    if method == "constant":
        return _reference_fill(method, values, constant)
    result = list(values)
    keys = list(zip(*(columns[name] for name in group_by or ["g"])))
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, key in enumerate(keys):
        if not any(_missing(part) for part in key):
            groups.setdefault(key, []).append(i)
    for key, indices in groups.items():
        group = [values[i] for i in indices]
        if method in ("mean", "median", "mode") and _statistic(method, group) is None:
            filled = _reference_fill(method, group, constant)
            overall = _statistic(method, values)
            filled = [overall if _missing(cell) else cell for cell in filled]
        else:
            filled = _reference_fill(method, group, constant)
        for i, cell in zip(indices, filled):
            result[i] = cell
    if method in ("mean", "median", "mode"):
        overall = _statistic(method, values)
        grouped_rows = {i for indices in groups.values() for i in indices}
        for i, cell in enumerate(values):
            if i not in grouped_rows and _missing(cell):
                result[i] = overall
    if method in ("ffill", "bfill"):
        grouped_rows = {i for indices in groups.values() for i in indices}
        result = [None if i not in grouped_rows and _missing(cell) else cell for i, cell in enumerate(result)]
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
_SINGLE: _Columns = {"a": [_NAN, 3.0, _NAN, 1.0, 3.0, None, 1.0, 5.0, None]}
_STRINGS: _Columns = {"a": ["b", None, "a", "b", "a"], "g": [1.0, 1.0, 2.0, 2.0, 2.0]}
_MULTI: _Columns = {
    "a": [3.0, _NAN, None, 2.0, 4.0, _NAN],
    "b": [3.0, 2.0, 6.0, 1.0, None, None],
    "c": [_NAN, 8.0, None, None, 4.0, _NAN],
}
_GROUPED: _Columns = {
    "a": [4.0, _NAN, 7.0, 2.0, None, 4.0, 2.0, _NAN, 9.0, 5.0, None, _NAN, None],
    "g": ["x", "x", "y", "x", "y", "x", "x", "x", "y", "x", "z", "z", "w"],
}
_MISSING_KEYS: _Columns = {
    "a": [10.0, None, None, 100.0, None, 20.0, None, _NAN, 60.0, None, 100.0],
    "g": [1.0, 1.0, float("nan"), float("nan"), float("nan"), 2.0, None, None, None, 2.0, float("nan")],
}
_ALL_KEYS_MISSING: _Columns = {
    "a": [1.0, None, 3.0, None, 5.0, 3.0],
    "g": [float("nan"), None, float("nan"), None, float("nan"), None],
}
_TWO_KEYS: _Columns = {
    "a": [4.0, None, 8.0, None, 30.0, None, 5.0, 30.0, None, None],
    "g": [1.0, 1.0, 1.0, 1.0, _NAN, 2.0, 2.0, 1.0, _NAN, 3.0],
    "h": ["p", "p", "p", None, "q", "q", "q", None, "q", "r"],
}
_F32_NAN = np.float32("nan")
_FLOAT32_NAN_KEY: _Columns = {
    "a": [1.0, None, 10.0, None, 3.0, None, 20.0, None, 7.0],
    "g": [1.0, 1.0, _F32_NAN, _F32_NAN, 2.0, 2.0, _F32_NAN, 1.0, 2.0],
}
_GROUPING_CASES = [
    pytest.param(_GROUPED, ["g"], id="grouped"),
    pytest.param(_FLOAT32_NAN_KEY, ["g"], id="float32_nan_key"),
    pytest.param(_MISSING_KEYS, ["g"], id="missing_keys"),
    pytest.param(_ALL_KEYS_MISSING, ["g"], id="all_keys_missing"),
    pytest.param(_TWO_KEYS, ["g", "h"], id="two_keys"),
]
_CONSTANT = 0.5
_ALL_METHODS = sorted(METHOD_UNIVERSE)
_MULTI_METHODS = [method for method in _ALL_METHODS if method not in ("ffill", "bfill")]


_NAT64 = np.datetime64("NaT")
_DAYS = [datetime.datetime(2024, 1, d) for d in (1, 2)]
_DAYS64 = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
_NAN_LIKE_KEYS = [
    pytest.param(pd.NA, ["x", "y"], id="pd_na"),
    pytest.param(pd.NaT, _DAYS, id="pd_nat"),
    pytest.param(_NAT64, _DAYS64, id="np_datetime64_nat"),
]
_NAN_LIKE_VALUES = [
    pytest.param(np.float32("nan"), id="np_float32_nan"),
    pytest.param(pd.NA, id="pd_na"),
    pytest.param(pd.NaT, id="pd_nat"),
]
_ARRAY_CELLS: _Cells = [np.array([1.0, 2.0]), None, np.array([3.0, 4.0]), None]

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
    @pytest.mark.parametrize("group_by", [None, ["g"]], ids=["ungrouped", "grouped"])
    def test_string_column_with_a_mode_tie(self, implementation: Any, method: str, group_by: list[str] | None) -> None:
        actual = _impute(implementation, _STRINGS, method, ["a"], "z", group_by)
        if group_by is None:
            expected = _reference_single(method, _STRINGS, "z")
        else:
            expected = _reference_grouped(method, _STRINGS, "z", group_by)
        assert not _differences(implementation, actual, expected)

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

    @pytest.mark.parametrize(("columns", "group_by"), _GROUPING_CASES)
    @pytest.mark.parametrize("method", _ALL_METHODS)
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
    def test_grouped_imputation(self, implementation: Any, method: str, columns: _Columns, group_by: list[str]) -> None:
        actual = _impute(implementation, columns, method, ["a"], _CONSTANT, group_by)
        assert not _differences(implementation, actual, _reference_grouped(method, columns, _CONSTANT, group_by))

    @pytest.mark.parametrize(("columns", "group_by"), _GROUPING_CASES)
    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_pandas_grouped_imputation_with_a_duplicate_index(
        self, method: str, columns: _Columns, group_by: list[str]
    ) -> None:
        data = _pandas_data(columns)
        data.index = pd.Index([i // 2 for i in range(len(data))])
        result = PandasMissingValueImputation._perform_imputation(data, method, ["a"], _CONSTANT, group_by)
        expected = _reference_grouped(method, columns, _CONSTANT, group_by)
        assert not _differences(PandasMissingValueImputation, _to_list(result), expected)

    @pytest.mark.parametrize("method", ["mean", "mode", "ffill", "bfill"])
    @pytest.mark.parametrize(
        "implementation",
        [PandasMissingValueImputation, PythonDictMissingValueImputation],
        ids=["pandas", "python_dict"],
    )
    @pytest.mark.parametrize(("scalar", "keys"), _NAN_LIKE_KEYS)
    def test_nan_like_group_key_joins_no_group(
        self, implementation: Any, method: str, scalar: Any, keys: _Cells
    ) -> None:
        k1, k2 = keys
        values: _Cells = [1.0, None, 3.0, None, 5.0, None, 9.0, 2.0]
        key_cells = [k1, k1, scalar, scalar, k2, k2, scalar, k1]
        columns: _Columns = {"a": values, "g": key_cells}
        with_none: _Columns = {"a": values, "g": [None if c is scalar else c for c in key_cells]}
        actual = _impute(implementation, columns, method, ["a"], None, ["g"])
        assert not _differences(implementation, actual, _reference_grouped(method, columns, None, ["g"]))
        baseline = _impute(implementation, with_none, method, ["a"], None, ["g"])
        assert not _differences(implementation, actual, baseline)


class NAType:
    """A user class that only shares its name with the pandas NA type."""


class _ArrayNe:
    def __eq__(self, other: object) -> np.ndarray:  # type: ignore[override]
        return np.array([True, False])

    def __ne__(self, other: object) -> np.ndarray:  # type: ignore[override]
        return np.array([False, True])

    def __hash__(self) -> int:
        return 1


class _RaisingNe:
    def __ne__(self, other: object) -> bool:
        raise RuntimeError("no comparison")

    def __hash__(self) -> int:
        return 2


class _NeverEqual:
    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 3


_NEVER_EQUAL_KEY = _NeverEqual()
_ODD_CELL_KINDS = [
    pytest.param(lambda i: np.array([float(i), 2.0]), id="numpy_array"),
    pytest.param(lambda i: _ArrayNe(), id="array_ne"),
    pytest.param(lambda i: _RaisingNe(), id="raising_ne"),
    pytest.param(lambda i: _NeverEqual(), id="never_equal"),
    pytest.param(lambda i: NAType(), id="user_class_named_natype"),
]


class TestNanNormalizationIsLimitedToFloatColumns:
    """NaN-like cells are missing; strings such as "nan", ints and other present values are untouched."""

    @pytest.mark.parametrize("method", ["mean", "ffill"])
    @pytest.mark.parametrize("scalar", _NAN_LIKE_VALUES)
    def test_python_dict_nan_like_value_cells_are_missing(self, method: str, scalar: Any) -> None:
        cells: _Cells = [1.0, scalar, 3.0, scalar, 8.0, scalar]
        columns: _Columns = {"a": cells}
        with_none: _Columns = {"a": [None if c is scalar else c for c in cells]}
        actual = _impute(PythonDictMissingValueImputation, columns, method, ["a"], None, None)
        expected = _reference_single(method, columns, None)
        assert not _differences(PythonDictMissingValueImputation, actual, expected)
        baseline = _impute(PythonDictMissingValueImputation, with_none, method, ["a"], None, None)
        assert not _differences(PythonDictMissingValueImputation, actual, baseline)

    @pytest.mark.parametrize(("method", "constant"), [("ffill", None), ("constant", 0.0)])
    @pytest.mark.parametrize("make", _ODD_CELL_KINDS)
    def test_python_dict_odd_cells_are_left_alone(self, method: str, constant: Any, make: Callable[[int], Any]) -> None:
        first, second = make(1), make(3)
        actual = _impute(
            PythonDictMissingValueImputation, {"a": [first, None, second, None]}, method, ["a"], constant, None
        )
        fill = first if method == "ffill" else constant
        expected = [first, fill, second, fill if method == "constant" else second]
        assert len(actual) == 4
        for a, e in zip(actual, expected):
            assert a is e

    @pytest.mark.parametrize("method", ["mean", "mode", "ffill"])
    def test_python_dict_never_equal_group_key_groups_like_a_plain_key(self, method: str) -> None:
        values: _Cells = [1.0, None, 3.0, None, 5.0, None, 9.0, 2.0]
        plain: _Columns = {"a": values, "g": ["x", "x", "y", "y", "x", "x", "y", "x"]}
        odd: _Columns = {"a": values, "g": [_NEVER_EQUAL_KEY if k == "x" else k for k in plain["g"]]}
        expected = _impute(PythonDictMissingValueImputation, plain, method, ["a"], None, ["g"])
        actual = _impute(PythonDictMissingValueImputation, odd, method, ["a"], None, ["g"])
        assert not _differences(PythonDictMissingValueImputation, actual, expected)

    @pytest.mark.parametrize("method", ["constant", "ffill"])
    @pytest.mark.parametrize("scalar", [Decimal("sNaN"), Decimal("NaN")], ids=["snan", "nan"])
    def test_python_dict_decimal_nan_cells_are_missing(self, method: str, scalar: Decimal) -> None:
        cells: _Cells = [Decimal("1.5"), scalar, Decimal("3"), scalar]
        actual = _impute(PythonDictMissingValueImputation, {"a": cells}, method, ["a"], Decimal("0"), None)
        fill = Decimal("1.5") if method == "ffill" else Decimal("0")
        assert actual == [Decimal("1.5"), fill, Decimal("3"), Decimal("3") if method == "ffill" else fill]

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
    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPL_IDS)
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
