"""Failing tests for the fluent ``Results`` list returned by run_all (issues #564/#568/#569).

``Results`` is a ``list`` subclass living in ``mloda.core.api.results`` (also exported
from ``mloda.user``). It replaces ``results_by_feature``/``run_one``/``run_all_as_dataframe``
with four accessors on the result list itself:

- ``get_one(name)``: the per-feature-group element containing a column (identity preserved)
- ``get_rows(name)``: that element converted to a flat list of row dicts
- ``get_values(name)``: one column as a plain Python list
- ``get_df()``: all elements horizontally concatenated into ONE pandas/polars DataFrame

The class does not exist yet; this module must fail at collection with ImportError.
"""

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.api.results import Results

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


class TestGetOneColumnExtraction:
    def test_columnar_dict_resolves_keys_to_element(self) -> None:
        element = {"col_a": [1, 2], "col_b": [3, 4]}
        results = Results([element])

        assert results.get_one("col_a") is element
        assert results.get_one("col_b") is element

    def test_list_of_row_dicts_resolves_first_row_keys_to_element(self) -> None:
        element = [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]
        results = Results([element])

        assert results.get_one("col_a") is element
        assert results.get_one("col_b") is element

    def test_empty_list_element_contributes_no_columns(self) -> None:
        results = Results([[]])

        with pytest.raises(ValueError):
            results.get_one("col_a")

    def test_pyarrow_table_resolves_column_names_to_element(self) -> None:
        element = pa.table({"col_a": [1, 2], "col_b": [3, 4]})
        results = Results([element])

        assert results.get_one("col_a") is element
        assert results.get_one("col_b") is element

    def test_pandas_dataframe_resolves_columns_to_element(self) -> None:
        element = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        results = Results([element])

        assert results.get_one("col_a") is element
        assert results.get_one("col_b") is element

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_dataframe_resolves_columns_to_element(self) -> None:
        element = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        results = Results([element])

        assert results.get_one("col_a") is element
        assert results.get_one("col_b") is element

    def test_named_lookup_picks_the_containing_element_among_several(self) -> None:
        first = {"col_a": [1]}
        second = {"col_b": [2]}
        results = Results([first, second])

        assert results.get_one("col_a") is first
        assert results.get_one("col_b") is second


class TestGetOneWithoutName:
    def test_single_element_is_returned_without_name(self) -> None:
        element = {"col_a": [1]}
        results = Results([element])

        assert results.get_one() is element

    def test_two_elements_without_name_raise_value_error_mentioning_count(self) -> None:
        results = Results([{"col_a": [1]}, {"col_b": [2]}])

        with pytest.raises(ValueError) as exc_info:
            results.get_one()

        assert "2" in str(exc_info.value)
        assert "get_one" in str(exc_info.value)

    def test_empty_results_without_name_raise_value_error(self) -> None:
        results = Results([])

        with pytest.raises(ValueError):
            results.get_one()


class TestGetOneMultiOutputNaming:
    def test_tilde_column_resolves_base_name_and_full_name(self) -> None:
        element = pd.DataFrame({"heatmap~a": [1], "heatmap~b": [2]})
        results = Results([element])

        assert results.get_one("heatmap") is element
        assert results.get_one("heatmap~a") is element
        assert results.get_one("heatmap~b") is element

    def test_base_name_uses_text_before_first_tilde_only(self) -> None:
        element = {"multi~x~y": [1]}
        results = Results([element])

        assert results.get_one("multi") is element
        assert results.get_one("multi~x~y") is element
        with pytest.raises(ValueError):
            results.get_one("multi~x")


class TestGetOneDuplicates:
    def test_first_occurrence_wins_across_elements(self) -> None:
        first = pd.DataFrame({"shared": [1]})
        second = {"shared": [2], "other": [3]}
        results = Results([first, second])

        assert results.get_one("shared") is first
        assert results.get_one("other") is second

    def test_first_occurrence_wins_for_multi_output_base_name(self) -> None:
        first = {"heatmap~a": [1]}
        second = {"heatmap": [2]}
        results = Results([first, second])

        assert results.get_one("heatmap") is first
        assert results.get_one("heatmap~a") is first


class TestGetOneErrors:
    def test_unsupported_element_type_raises_value_error_with_type_in_message(self) -> None:
        results = Results([42])

        with pytest.raises(ValueError, match="int"):
            results.get_one("col_a")

    def test_list_of_scalars_raises_value_error_with_type_in_message(self) -> None:
        results = Results([[1, 2]])

        with pytest.raises(ValueError, match="int"):
            results.get_one("col_a")

    def test_unknown_name_raises_value_error_listing_available_names(self) -> None:
        results = Results([{"col_a": [1], "col_b": [2]}])

        with pytest.raises(ValueError) as exc_info:
            results.get_one("does_not_exist")

        assert "col_a" in str(exc_info.value)
        assert "col_b" in str(exc_info.value)


class TestGetRows:
    def test_columnar_dict_zips_into_rows(self) -> None:
        results = Results([{"col_a": [1, 2], "col_b": [3, 4]}])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_pyarrow_table_converts_to_pylist_rows(self) -> None:
        results = Results([pa.table({"col_a": [1, 2], "col_b": [3, 4]})])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_pandas_dataframe_converts_to_records_rows(self) -> None:
        results = Results([pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_dataframe_converts_to_dicts_rows(self) -> None:
        results = Results([pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_lazyframe_collects_to_rows(self) -> None:
        results = Results([pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).lazy()])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_spark_like_object_collects_rows_via_as_dict(self) -> None:
        results = Results([_SparkDataFrameStub([{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}])])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_relation_like_object_converts_df_records(self) -> None:
        results = Results([_RelationStub(pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}))])

        rows = results.get_rows()

        assert rows == [{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]

    def test_list_element_returns_a_copy(self) -> None:
        source: list[dict[str, Any]] = [{"col_a": 1}, {"col_a": 2}]
        results = Results([source])

        rows = results.get_rows()

        assert rows == source
        assert rows is not source
        rows.append({"col_a": 3})
        assert source == [{"col_a": 1}, {"col_a": 2}]

    def test_named_lookup_converts_the_containing_element(self) -> None:
        results = Results([{"col_a": [1]}, {"col_b": [2]}])

        rows = results.get_rows("col_b")

        assert rows == [{"col_b": 2}]

    def test_ragged_columnar_dict_raises_value_error(self) -> None:
        results = Results([{"col_a": [1, 2], "col_b": [1]}])

        with pytest.raises(ValueError):
            results.get_rows()

    def test_unsupported_element_type_raises_value_error_with_type_in_message(self) -> None:
        results = Results([42])

        with pytest.raises(ValueError, match="int"):
            results.get_rows()


class TestGetValues:
    def test_pandas_dataframe_column_returns_plain_list(self) -> None:
        results = Results([pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})])

        values = results.get_values("col_a")

        assert values == [1, 2]
        assert type(values) is list

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_dataframe_column_returns_plain_list(self) -> None:
        results = Results([pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})])

        values = results.get_values("col_a")

        assert values == [1, 2]
        assert type(values) is list

    def test_pyarrow_table_column_returns_plain_list(self) -> None:
        results = Results([pa.table({"col_a": [1, 2], "col_b": [3, 4]})])

        values = results.get_values("col_a")

        assert values == [1, 2]
        assert type(values) is list

    def test_columnar_dict_column_returns_plain_list(self) -> None:
        results = Results([{"col_a": [1, 2], "col_b": [3, 4]}])

        values = results.get_values("col_b")

        assert values == [3, 4]
        assert type(values) is list

    def test_list_of_row_dicts_column_returns_plain_list(self) -> None:
        results = Results([[{"col_a": 1, "col_b": 3}, {"col_a": 2, "col_b": 4}]])

        values = results.get_values("col_a")

        assert values == [1, 2]
        assert type(values) is list

    def test_unknown_name_raises_value_error_listing_available_names(self) -> None:
        results = Results([{"col_a": [1], "col_b": [2]}])

        with pytest.raises(ValueError) as exc_info:
            results.get_values("does_not_exist")

        assert "col_a" in str(exc_info.value)
        assert "col_b" in str(exc_info.value)

    def test_multi_output_base_name_that_is_not_a_column_raises_value_error(self) -> None:
        results = Results([{"multi~x": [1], "multi~y": [2]}])

        with pytest.raises(ValueError):
            results.get_values("multi")


class TestGetDf:
    def test_mixed_pandas_and_pyarrow_raises_value_error_naming_dropped_type(self) -> None:
        results = Results([pd.DataFrame({"col_a": [1, 2]}), pa.table({"col_b": [3, 4]})])

        with pytest.raises(ValueError) as exc_info:
            results.get_df()

        assert "Table" in str(exc_info.value)
        assert "DataFrame" in str(exc_info.value)

    def test_mixed_pandas_and_columnar_dict_raises_value_error(self) -> None:
        results = Results([pd.DataFrame({"col_a": [1, 2]}), {"col_b": [3, 4]}])

        with pytest.raises(ValueError):
            results.get_df()

    def test_only_non_dataframe_elements_raise_value_error_mentioning_dataframe(self) -> None:
        results = Results([{"col_a": [1, 2]}])

        with pytest.raises(ValueError, match="DataFrame"):
            results.get_df()

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_mixed_pandas_and_polars_raises_value_error(self) -> None:
        results = Results([pd.DataFrame({"col_a": [1, 2]}), pl.DataFrame({"col_b": [3, 4]})])

        with pytest.raises(ValueError):
            results.get_df()

    def test_row_count_mismatch_raises_value_error_mentioning_row(self) -> None:
        results = Results([pd.DataFrame({"col_a": [1, 2, 3]}), pd.DataFrame({"col_b": [7, 8]})])

        with pytest.raises(ValueError, match="row"):
            results.get_df()

    def test_single_frame_is_returned_as_is(self) -> None:
        frame = pd.DataFrame({"col_a": [1, 2]})
        results = Results([frame])

        assert results.get_df() is frame

    def test_overlapping_columns_pandas_keeps_first_frame_values(self) -> None:
        first = pd.DataFrame({"col_a": [1, 2], "shared": [10, 20]})
        second = pd.DataFrame({"shared": [99, 98], "col_b": [3, 4]})
        results = Results([first, second])

        combined = results.get_df()

        assert list(combined.columns).count("shared") == 1
        assert list(combined["shared"]) == [10, 20]
        assert list(combined["col_a"]) == [1, 2]
        assert list(combined["col_b"]) == [3, 4]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_overlapping_columns_polars_keeps_first_frame_values(self) -> None:
        first = pl.DataFrame({"col_a": [1, 2], "shared": [10, 20]})
        second = pl.DataFrame({"shared": [99, 98], "col_b": [3, 4]})
        results = Results([first, second])

        combined = results.get_df()

        assert combined.columns.count("shared") == 1
        assert combined["shared"].to_list() == [10, 20]
        assert combined["col_a"].to_list() == [1, 2]
        assert combined["col_b"].to_list() == [3, 4]

    def test_zero_row_frames_with_different_columns_concatenate(self) -> None:
        first = pd.DataFrame({"col_a": []})
        second = pd.DataFrame({"col_b": []})
        results = Results([first, second])

        combined = results.get_df()

        assert list(combined.columns) == ["col_a", "col_b"]
        assert len(combined) == 0

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_lazyframes_are_collected_and_concatenated(self) -> None:
        first = pl.DataFrame({"col_a": [1, 2]}).lazy()
        second = pl.DataFrame({"col_b": [3, 4]}).lazy()
        results = Results([first, second])

        combined = results.get_df()

        assert isinstance(combined, pl.DataFrame)
        assert combined.columns == ["col_a", "col_b"]
        assert combined["col_a"].to_list() == [1, 2]
        assert combined["col_b"].to_list() == [3, 4]


class TestResultsIsAList:
    def test_results_is_instance_of_list(self) -> None:
        results = Results([{"col_a": [1]}])

        assert isinstance(results, list)

    def test_indexing_returns_the_original_elements(self) -> None:
        first = {"col_a": [1]}
        second = {"col_b": [2]}
        results = Results([first, second])

        assert results[0] is first
        assert results[1] is second

    def test_len_matches_element_count(self) -> None:
        assert len(Results([{"col_a": [1]}, {"col_b": [2]}])) == 2
        assert len(Results([])) == 0

    def test_iteration_yields_elements_in_order(self) -> None:
        first = {"col_a": [1]}
        second = {"col_b": [2]}
        results = Results([first, second])

        assert [element for element in results] == [first, second]

    def test_equality_with_plain_list(self) -> None:
        elements: list[Any] = [{"col_a": [1]}, {"col_b": [2]}]

        assert Results(elements) == elements
        assert Results([]) == []

    def test_results_is_exported_from_mloda_user(self) -> None:
        from mloda.user import Results as UserResults

        assert UserResults is Results
