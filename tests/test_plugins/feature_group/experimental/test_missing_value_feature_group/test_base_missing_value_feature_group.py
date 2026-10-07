from typing import Any

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys

from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda.provider import FeatureChainParser


class ConcreteMissingValueFeatureGroup(MissingValueFeatureGroup):
    """Minimal concrete implementation for testing base class methods."""

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        return set()

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        pass

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        return data

    @classmethod
    def _perform_imputation(
        cls,
        data: Any,
        imputation_method: str,
        in_features: list[str],
        constant_value: Any | None = None,
        group_by_features: list[str] | None = None,
    ) -> Any:
        return data


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
        feature_group = ConcreteMissingValueFeatureGroup()

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
        result = ConcreteMissingValueFeatureGroup().input_features(options, FeatureName("income__mean_imputed"))
        assert result is not None
        assert {str(f.name) for f in result} == expected
        assert len(result) == len(expected)

    @pytest.mark.parametrize(
        ("source", "source_name", "expected_group"),
        [
            pytest.param(
                Feature("income", Options(group={"scope": "a"})), "income", {"scope": "a"}, id="scoped_source"
            ),
            pytest.param(Feature("income"), "income", {}, id="plain_source"),
            pytest.param(
                Feature(
                    "income_sum",
                    Options(group={"aggregation_type": "sum", DefaultOptionKeys.in_features: "income"}),
                ),
                "income_sum",
                {"aggregation_type": "sum"},
                id="chained_source",
            ),
        ],
    )
    def test_group_by_features_carry_only_forwarded_group_options_of_source(
        self, source: Feature, source_name: str, expected_group: dict[str, Any]
    ) -> None:
        options = Options(
            group={"imputation_method": "mean"},
            context={DefaultOptionKeys.in_features: frozenset([source]), "group_by_features": ["region"]},
        )
        result = ConcreteMissingValueFeatureGroup().input_features(options, FeatureName("x"))
        assert result is not None
        by_name = {str(f.name): f for f in result}
        assert set(by_name) == {source_name, "region"}
        assert dict(by_name["region"].options.group) == expected_group
        assert DefaultOptionKeys.in_features not in by_name["region"].options.group
        if not expected_group:
            assert by_name["region"] == Feature("region")
