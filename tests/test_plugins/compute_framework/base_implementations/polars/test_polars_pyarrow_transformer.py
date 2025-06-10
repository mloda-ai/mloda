import pytest
from mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer import (
    PolarsPyarrowTransformer,
)

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None


@pytest.mark.skipif(pl is None or pa is None, reason="Polars or PyArrow is not installed. Skipping this test.")
class TestPolarsPyarrowTransformer:
    def test_framework_types(self) -> None:
        """Test that the transformer returns the correct framework types."""
        assert PolarsPyarrowTransformer.framework() == pl.DataFrame
        assert PolarsPyarrowTransformer.other_framework() == pa.Table

    def test_check_imports(self) -> None:
        """Test that import checks work correctly."""
        assert PolarsPyarrowTransformer.check_imports() is True

    def test_polars_to_pyarrow_transformation(self) -> None:
        """Test transformation from Polars DataFrame to PyArrow Table."""
        # Create test data
        df = pl.DataFrame({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]})

        # Transform to PyArrow
        arrow_table = PolarsPyarrowTransformer.transform_fw_to_other_fw(df)

        # Verify it's a PyArrow Table
        assert isinstance(arrow_table, pa.Table)

        # Verify data integrity
        assert arrow_table.num_rows == 3
        assert arrow_table.num_columns == 3
        assert set(arrow_table.column_names) == {"int_col", "str_col", "float_col"}

    def test_pyarrow_to_polars_transformation(self) -> None:
        """Test transformation from PyArrow Table to Polars DataFrame."""
        # Create test data
        arrow_table = pa.table({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]})

        # Transform to Polars
        df = PolarsPyarrowTransformer.transform_other_fw_to_fw(arrow_table, None)

        # Verify it's a Polars DataFrame
        assert isinstance(df, pl.DataFrame)

        # Verify data integrity
        assert len(df) == 3
        assert len(df.columns) == 3
        assert set(df.columns) == {"int_col", "str_col", "float_col"}

    def test_roundtrip_transformation(self) -> None:
        """Test that data survives a roundtrip transformation."""
        # Create original data
        original_df = pl.DataFrame(
            {
                "int_col": [1, 2, 3, 4],
                "str_col": ["hello", "world", "test", "data"],
                "float_col": [1.1, 2.2, 3.3, 4.4],
                "bool_col": [True, False, True, False],
            }
        )

        # Roundtrip: Polars -> PyArrow -> Polars
        arrow_table = PolarsPyarrowTransformer.transform_fw_to_other_fw(original_df)
        restored_df = PolarsPyarrowTransformer.transform_other_fw_to_fw(arrow_table, None)

        # Verify data integrity
        assert restored_df.equals(original_df)

    def test_empty_dataframe_transformation(self) -> None:
        """Test transformation of empty DataFrames."""
        # Create empty DataFrame with schema
        empty_df = pl.DataFrame({"int_col": pl.Series([], dtype=pl.Int64), "str_col": pl.Series([], dtype=pl.Utf8)})

        # Transform to PyArrow and back
        arrow_table = PolarsPyarrowTransformer.transform_fw_to_other_fw(empty_df)
        restored_df = PolarsPyarrowTransformer.transform_other_fw_to_fw(arrow_table, None)

        # Verify structure is preserved
        assert len(restored_df) == 0
        assert set(restored_df.columns) == {"int_col", "str_col"}

    def test_null_values_transformation(self) -> None:
        """Test transformation of DataFrames with null values."""
        # Create DataFrame with null values
        df_with_nulls = pl.DataFrame(
            {"int_col": [1, None, 3], "str_col": ["a", None, "c"], "float_col": [1.1, 2.2, None]}
        )

        # Roundtrip transformation
        arrow_table = PolarsPyarrowTransformer.transform_fw_to_other_fw(df_with_nulls)
        restored_df = PolarsPyarrowTransformer.transform_other_fw_to_fw(arrow_table, None)

        # Verify null values are preserved
        assert restored_df.equals(df_with_nulls)

    def test_transform_method_with_correct_orientation(self) -> None:
        """Test the generic transform method with correct framework orientation."""
        # Create test data
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Test Polars -> PyArrow transformation
        arrow_result = PolarsPyarrowTransformer.transform(pl.DataFrame, pa.Table, df, None)
        assert isinstance(arrow_result, pa.Table)

        # Test PyArrow -> Polars transformation
        polars_result = PolarsPyarrowTransformer.transform(pa.Table, pl.DataFrame, arrow_result, None)
        assert isinstance(polars_result, pl.DataFrame)
        assert polars_result.equals(df)

    def test_transform_method_with_unsupported_frameworks(self) -> None:
        """Test that transform method raises error for unsupported frameworks."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError):
            PolarsPyarrowTransformer.transform(list, dict, df, None)
