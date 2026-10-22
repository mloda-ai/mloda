"""
Tests for the TextCleaningFeatureGroup base class.
"""

from typing import Any
from unittest.mock import Mock, call

import pandas as pd
import pytest

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaning
from mloda_plugins.feature_group.experimental.text_cleaning.python_dict import PythonDictTextCleaning

IMPLEMENTATIONS = [PandasTextCleaning, PythonDictTextCleaning]
FRAMEWORKS = [PandasDataFrame, PythonDictFramework]
IMPL_DATA = [
    (PandasTextCleaning, lambda: pd.DataFrame({"b": ["x"], "a": ["y"]})),
    (PythonDictTextCleaning, lambda: {"b": ["x"], "a": ["y"]}),
]
IMPL_IDS = ["pandas", "python_dict"]


def _feature_set(name: str, *operations: str) -> FeatureSet:
    feature_set = FeatureSet()
    feature_set.add(Feature(FeatureName(name), Options({TextCleaningFeatureGroup.CLEANING_OPERATIONS: operations})))
    return feature_set


class TestTextCleaningFeatureGroupBase:
    """Tests for the TextCleaningFeatureGroup base class."""

    def test_match_feature_group_criteria_valid(self) -> None:
        """Test that valid feature names with cleaning operations supplied are accepted."""
        # Valid feature name; cleaning_operations is option-required on the name path.
        feature_name = "review__cleaned_text"
        options = Options(context={TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",)})

        # Test with string
        assert TextCleaningFeatureGroup.match_feature_group_criteria(feature_name, options)

        # Test with FeatureName
        feature_name_obj = FeatureName(feature_name)
        assert TextCleaningFeatureGroup.match_feature_group_criteria(feature_name_obj, options)

    def test_extract_operations_without_options_raises(self) -> None:
        """Absent cleaning operations must raise instead of silently copying the source column."""
        feature = Feature(FeatureName("text__cleaned_text"), Options())

        with pytest.raises(ValueError):
            TextCleaningFeatureGroup._extract_operations_and_source_feature(feature)

    def test_match_feature_group_criteria_invalid(self) -> None:
        """Test that invalid feature names are rejected."""
        options = Options()

        # Invalid feature names
        invalid_names = [
            "clean_text__review",  # Wrong prefix
            "cleaned_text_review",  # Missing double underscore
            "cleaned__text__review",  # Extra double underscore in wrong place
        ]

        for name in invalid_names:
            assert not TextCleaningFeatureGroup.match_feature_group_criteria(name, options)

    def test_input_features(self) -> None:
        """Test that the feature group correctly extracts source features."""
        feature_name = FeatureName("review__cleaned_text")
        options = Options()

        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        assert next(iter(input_features)).name == "review"

    def test_feature_chaining(self) -> None:
        """Test that the feature group works with chained features."""
        # Chained feature name (cleaned_text applied to an aggregated feature)
        feature_name = FeatureName("sum_aggr__sales__cleaned_text")
        options = Options()

        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        assert next(iter(input_features)).name == "sum_aggr__sales"

    @pytest.mark.parametrize(("impl", "make_data"), IMPL_DATA, ids=IMPL_IDS)
    def test_clean_source_check_is_strict_and_lists_sorted_columns(self, impl: Any, make_data: Any) -> None:
        with pytest.raises(ValueError) as exc_info:
            impl.calculate(TextCleaningFeatureGroup, make_data(), _feature_set("nope__cleaned_text", "normalize"), None)
        assert str(exc_info.value) == "Source features not found in data: ['nope']. Available columns: ['a', 'b']"

    @pytest.mark.parametrize(("impl", "make_data"), IMPL_DATA, ids=IMPL_IDS)
    def test_clean_rejects_an_unsupported_operation(self, impl: Any, make_data: Any) -> None:
        with pytest.raises(ValueError, match="Unsupported cleaning operation"):
            impl.calculate(TextCleaningFeatureGroup, make_data(), _feature_set("a__cleaned_text", "bogus"), None)

    def test_clean_applies_operations_in_order_through_apply(self) -> None:
        data = object()
        source_text = Mock(return_value="t0")
        apply = Mock(side_effect=["t1", "t2"])
        add_column = Mock(return_value="done")
        feature_set = _feature_set("a__cleaned_text", "normalize", "remove_urls")

        result = TextCleaningFeatureGroup.clean(
            data,
            feature_set,
            source_text=source_text,
            apply=apply,
            add_column=add_column,
            columns=Mock(return_value={"a"}),
        )

        source_text.assert_called_once_with(data, "a")
        assert apply.call_args_list == [call(data, "t0", "normalize"), call(data, "t1", "remove_urls")]
        add_column.assert_called_once_with(data, "a__cleaned_text", "t2")
        assert result == "done"


class TestTextCleaningFeatureChainParser:
    """Tests for the TextCleaningFeatureGroup's modernized configuration-based features."""

    def test_configuration_based_matching(self) -> None:
        """Test that configuration-based features are properly matched."""
        # Test configuration-based feature creation
        options = Options(
            context={
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords"),
                DefaultOptionKeys.in_features: "review",
            }
        )

        # Configuration-based features should match with placeholder names
        assert TextCleaningFeatureGroup.match_feature_group_criteria("placeholder", options)

        # Test with group/context separation - operations in context should match
        options_with_group = Options(
            group={"some_group_param": "value"},
            context={
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_punctuation"),
                DefaultOptionKeys.in_features: "description",
            },
        )
        assert TextCleaningFeatureGroup.match_feature_group_criteria("placeholder", options_with_group)

    def test_configuration_based_input_features(self) -> None:
        """Test that configuration-based features extract correct input features."""
        # Configuration-based feature with source feature in options
        options = Options(
            context={
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords"),
                DefaultOptionKeys.in_features: "review",
            }
        )

        feature_group = TextCleaningFeatureGroup()
        feature_name = FeatureName("placeholder")

        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        source_feature = next(iter(input_features))
        assert source_feature.name == "review"


_TEXTS: list[Any] = ["Hello, World!", None, float("nan"), "  Café   https://x.y  ", ""]


def _cleaned(impl: Any, data: Any, operation: str) -> list[Any]:
    source = impl._get_source_text(data, "t")
    return list(impl._apply_operation(data, source, operation))


class TestImplementationsConformToTheReferenceContract:
    """Null and NaN become an empty string before the operations, in every implementation."""

    def test_group_declares_no_subtypes(self) -> None:
        assert TextCleaningFeatureGroup.subtype_universe() == frozenset()

    @pytest.mark.parametrize("operation", sorted(TextCleaningFeatureGroup.SUPPORTED_OPERATIONS))
    def test_implementations_give_the_same_output(self, operation: str) -> None:
        if operation == "remove_stopwords":
            pytest.importorskip("nltk")
        pandas_out = _cleaned(PandasTextCleaning, pd.DataFrame({"t": _TEXTS}), operation)
        python_dict_out = _cleaned(PythonDictTextCleaning, {"t": list(_TEXTS)}, operation)
        assert python_dict_out == pandas_out

    @pytest.mark.parametrize(
        ("impl", "make_data"),
        [
            (PandasTextCleaning, lambda: pd.DataFrame({"t": _TEXTS})),
            (PythonDictTextCleaning, lambda: {"t": list(_TEXTS)}),
        ],
        ids=IMPL_IDS,
    )
    def test_null_and_nan_become_an_empty_string(self, impl: Any, make_data: Any) -> None:
        source = list(impl._get_source_text(make_data(), "t"))
        assert source[1] == ""
        assert source[2] == ""
