import pytest
from typing import Any
from unittest.mock import patch

from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_pyarrow_transformer import (
    PolarsLazyPyarrowTransformer,
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


class TestPolarsLazyPyarrowTransformerAvailability:
    def test_framework_when_polars_available(self) -> None:
        """Test that framework() returns LazyFrame when polars is available."""
        if pl is not None:
            assert PolarsLazyPyarrowTransformer.framework() == pl.LazyFrame
        else:
            assert PolarsLazyPyarrowTransformer.framework() == NotImplementedError

    def test_other_framework_when_pyarrow_available(self) -> None:
        """Test that other_framework() returns Table when pyarrow is available."""
        if pa is not None:
            assert PolarsLazyPyarrowTransformer.other_framework() == pa.Table
        else:
            assert PolarsLazyPyarrowTransformer.other_framework() == NotImplementedError


@pytest.mark.skipif(pl is None or pa is None, reason="Polars or PyArrow is not installed. Skipping this test.")
class TestPolarsLazyPyarrowTransformer:
    if pl and pa:
        test_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        lazy_frame = pl.LazyFrame(test_data)
        arrow_table = pa.table(test_data)

    def test_framework(self) -> None:
        assert PolarsLazyPyarrowTransformer.framework() == pl.LazyFrame

    def test_other_framework(self) -> None:
        assert PolarsLazyPyarrowTransformer.other_framework() == pa.Table

    def test_transform_lazy_frame_to_arrow_table(self) -> None:
        """Test transformation from LazyFrame to PyArrow Table"""
        result = PolarsLazyPyarrowTransformer.transform_fw_to_other_fw(self.lazy_frame)

        # Result should be a PyArrow Table
        assert isinstance(result, pa.Table)

        # Verify data integrity
        expected_df = pl.DataFrame(self.test_data)
        result_df = pl.from_arrow(result)
        assert result_df.equals(expected_df)  # type: ignore

    def test_transform_arrow_table_to_lazy_frame(self) -> None:
        """Test transformation from PyArrow Table to LazyFrame"""
        result = PolarsLazyPyarrowTransformer.transform_other_fw_to_fw(self.arrow_table)

        # Result should be a LazyFrame
        assert isinstance(result, pl.LazyFrame)

        # Verify data integrity by collecting and comparing
        result_df = result.collect()
        expected_df = pl.DataFrame(self.test_data)
        assert result_df.equals(expected_df)

    def test_round_trip_transformation(self) -> None:
        """Test LazyFrame -> PyArrow -> LazyFrame round trip"""
        # LazyFrame to PyArrow
        arrow_result = PolarsLazyPyarrowTransformer.transform_fw_to_other_fw(self.lazy_frame)

        # PyArrow back to LazyFrame
        lazy_result = PolarsLazyPyarrowTransformer.transform_other_fw_to_fw(arrow_result)

        # Should be equivalent to original
        original_df = self.lazy_frame.collect()
        result_df = lazy_result.collect()
        assert result_df.equals(original_df)

    def test_transform_other_fw_to_fw_missing_polars(self) -> None:
        """Test error handling when polars is not available"""
        with patch(
            "mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_pyarrow_transformer.pl", None
        ):
            with pytest.raises(ImportError, match="Polars is not installed"):
                PolarsLazyPyarrowTransformer.transform_other_fw_to_fw(self.arrow_table)


@pytest.mark.skipif(pl is None or pa is None, reason="Polars or PyArrow is not installed. Skipping this test.")
class TestPolarsLazyPyarrowTransformerEquivalence:
    """Test that lazy transformer produces equivalent results to eager transformer"""

    if pl and pa:
        test_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}

    def test_lazy_vs_eager_arrow_conversion(self) -> None:
        """Test that lazy and eager transformers produce equivalent PyArrow results"""
        from mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer import (
            PolarsPyarrowTransformer,
        )

        # Create equivalent data
        df = pl.DataFrame(self.test_data)
        lazy_df = df.lazy()

        # Transform using both transformers
        eager_result = PolarsPyarrowTransformer.transform_fw_to_other_fw(df)
        lazy_result = PolarsLazyPyarrowTransformer.transform_fw_to_other_fw(lazy_df)

        # Results should be equivalent
        assert eager_result.equals(lazy_result)

    def test_lazy_vs_eager_polars_conversion(self) -> None:
        """Test that lazy and eager transformers produce equivalent Polars results"""
        from mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer import (
            PolarsPyarrowTransformer,
        )

        # Create PyArrow table
        arrow_table = pa.table(self.test_data)

        # Transform using both transformers
        eager_result = PolarsPyarrowTransformer.transform_other_fw_to_fw(arrow_table)
        lazy_result = PolarsLazyPyarrowTransformer.transform_other_fw_to_fw(arrow_table)

        # Collect lazy result and compare
        lazy_collected = lazy_result.collect()
        assert lazy_collected.equals(eager_result)
