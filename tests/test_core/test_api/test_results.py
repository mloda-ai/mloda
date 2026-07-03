"""Tests for results_by_feature, mapping run_all result elements by their column names."""

from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from mloda.core.api.results import results_by_feature


class TestResultsByFeatureColumnExtraction:
    def test_columnar_dict_maps_keys_to_element(self) -> None:
        element = {"col_a": [1, 2], "col_b": [3, 4]}
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["col_a"] is element
        assert mapping["col_b"] is element

    def test_list_of_row_dicts_maps_first_row_keys_to_element(self) -> None:
        element = [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["col_a"] is element
        assert mapping["col_b"] is element

    def test_empty_list_element_contributes_no_columns(self) -> None:
        results: list[Any] = [[]]

        mapping = results_by_feature(results)

        assert mapping == {}

    def test_pyarrow_table_maps_column_names_to_element(self) -> None:
        element = pa.table({"col_a": [1, 2], "col_b": [3, 4]})
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["col_a"] is element
        assert mapping["col_b"] is element

    def test_pandas_dataframe_maps_columns_to_element(self) -> None:
        element = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["col_a"] is element
        assert mapping["col_b"] is element

    def test_polars_dataframe_maps_columns_to_element(self) -> None:
        element = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["col_a"] is element
        assert mapping["col_b"] is element


class TestResultsByFeatureMultiOutputNaming:
    def test_tilde_column_registers_base_name_and_full_name(self) -> None:
        element = pd.DataFrame({"heatmap~a": [1], "heatmap~b": [2]})
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["heatmap"] is element
        assert mapping["heatmap~a"] is element
        assert mapping["heatmap~b"] is element

    def test_base_name_uses_text_before_first_tilde_only(self) -> None:
        element = {"multi~x~y": [1]}
        results: list[Any] = [element]

        mapping = results_by_feature(results)

        assert mapping["multi"] is element
        assert mapping["multi~x~y"] is element
        assert "multi~x" not in mapping


class TestResultsByFeatureDuplicates:
    def test_first_occurrence_wins_across_elements(self) -> None:
        first = pd.DataFrame({"shared": [1]})
        second = {"shared": [2], "other": [3]}
        results: list[Any] = [first, second]

        mapping = results_by_feature(results)

        assert mapping["shared"] is first
        assert mapping["other"] is second

    def test_first_occurrence_wins_for_multi_output_base_name(self) -> None:
        first = {"heatmap~a": [1]}
        second = {"heatmap": [2]}
        results: list[Any] = [first, second]

        mapping = results_by_feature(results)

        assert mapping["heatmap"] is first
        assert mapping["heatmap~a"] is first


class TestResultsByFeatureErrors:
    def test_unsupported_element_type_raises_value_error_with_type_in_message(self) -> None:
        results: list[Any] = [42]

        with pytest.raises(ValueError, match="int"):
            results_by_feature(results)

    def test_absent_feature_name_raises_key_error(self) -> None:
        results: list[Any] = [{"col_a": [1]}]

        mapping = results_by_feature(results)

        with pytest.raises(KeyError):
            mapping["does_not_exist"]


class TestResultsByFeatureReturnType:
    def test_returns_plain_dict(self) -> None:
        results: list[Any] = [{"col_a": [1]}]

        mapping = results_by_feature(results)

        assert type(mapping) is dict

    def test_empty_results_list_returns_empty_dict(self) -> None:
        results: list[Any] = []

        mapping = results_by_feature(results)

        assert mapping == {}
        assert type(mapping) is dict
