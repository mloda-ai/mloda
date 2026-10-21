from typing import Any
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pytest

from mloda.provider import FeatureSet
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
