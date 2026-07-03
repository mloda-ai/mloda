"""Tests for results_by_feature, _to_rows and _concat_frames (run_all result reading)."""

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.api.results import _concat_frames, _to_rows, results_by_feature

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class _SparkRowStub:
    """Mimics a pyspark Row: exposes asDict() only."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def asDict(self) -> dict[str, Any]:
        return dict(self._data)


class _SparkDataFrameStub:
    """Mimics a pyspark DataFrame: collect() returning Row-like objects plus a toPandas attribute."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def collect(self) -> list[_SparkRowStub]:
        return [_SparkRowStub(row) for row in self._rows]

    def toPandas(self) -> Any:
        return None


class _RelationStub:
    """Mimics a DuckDB/SQLite relation: df() only, no collect/to_pylist/to_dicts/to_dict."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def df(self) -> pd.DataFrame:
        return self._frame


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

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
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

    def test_list_of_scalars_raises_value_error_with_type_in_message(self) -> None:
        results: list[Any] = [[1, 2]]

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


class TestToRows:
    def test_pyarrow_table_converts_to_pylist_rows(self) -> None:
        table = pa.table({"col_a": [1, 2], "col_b": [3, 4]})

        rows = _to_rows(table)

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_dataframe_converts_to_dicts_rows(self) -> None:
        frame = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})

        rows = _to_rows(frame)

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_lazyframe_collects_to_rows(self) -> None:
        lazy = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).lazy()

        rows = _to_rows(lazy)

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_spark_like_object_collects_rows_via_as_dict(self) -> None:
        stub = _SparkDataFrameStub([{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}])

        rows = _to_rows(stub)

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_relation_like_object_converts_df_records(self) -> None:
        stub = _RelationStub(pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}))

        rows = _to_rows(stub)

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_list_input_returns_a_copy(self) -> None:
        source: list[dict[str, Any]] = [{"col_a": 1}, {"col_a": 2}]

        rows = _to_rows(source)

        assert rows == source
        assert rows is not source
        rows.append({"col_a": 3})
        assert source == [{"col_a": 1}, {"col_a": 2}]

    def test_ragged_columnar_dict_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _to_rows({"col_a": [1, 2], "col_b": [1]})

    def test_unsupported_type_raises_value_error_with_type_in_message(self) -> None:
        with pytest.raises(ValueError, match="int"):
            _to_rows(42)


class TestConcatFrames:
    def test_mixed_pandas_and_pyarrow_raises_value_error_naming_dropped_type(self) -> None:
        results: list[Any] = [pd.DataFrame({"col_a": [1, 2]}), pa.table({"col_b": [3, 4]})]

        with pytest.raises(ValueError) as exc_info:
            _concat_frames(results)

        assert "Table" in str(exc_info.value)
        assert "DataFrame" in str(exc_info.value)

    def test_mixed_pandas_and_columnar_dict_raises_value_error(self) -> None:
        results: list[Any] = [pd.DataFrame({"col_a": [1, 2]}), {"col_b": [3, 4]}]

        with pytest.raises(ValueError):
            _concat_frames(results)

    def test_overlapping_columns_pandas_keeps_first_frame_values(self) -> None:
        first = pd.DataFrame({"col_a": [1, 2], "shared": [10, 20]})
        second = pd.DataFrame({"shared": [99, 98], "col_b": [3, 4]})

        combined = _concat_frames([first, second])

        assert list(combined.columns).count("shared") == 1
        assert list(combined["shared"]) == [10, 20]
        assert list(combined["col_a"]) == [1, 2]
        assert list(combined["col_b"]) == [3, 4]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_overlapping_columns_polars_keeps_first_frame_values(self) -> None:
        first = pl.DataFrame({"col_a": [1, 2], "shared": [10, 20]})
        second = pl.DataFrame({"shared": [99, 98], "col_b": [3, 4]})

        combined = _concat_frames([first, second])

        assert combined.columns.count("shared") == 1
        assert combined["shared"].to_list() == [10, 20]
        assert combined["col_a"].to_list() == [1, 2]
        assert combined["col_b"].to_list() == [3, 4]

    def test_zero_row_frames_with_different_columns_concatenate(self) -> None:
        first = pd.DataFrame({"col_a": []})
        second = pd.DataFrame({"col_b": []})

        combined = _concat_frames([first, second])

        assert list(combined.columns) == ["col_a", "col_b"]
        assert len(combined) == 0

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_lazyframes_are_collected_and_concatenated(self) -> None:
        first = pl.DataFrame({"col_a": [1, 2]}).lazy()
        second = pl.DataFrame({"col_b": [3, 4]}).lazy()

        combined = _concat_frames([first, second])

        assert isinstance(combined, pl.DataFrame)
        assert combined.columns == ["col_a", "col_b"]
        assert combined["col_a"].to_list() == [1, 2]
        assert combined["col_b"].to_list() == [3, 4]
