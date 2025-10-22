import pytest
from typing import Any, Optional, Type
from unittest.mock import patch

from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_pyarrow_transformer import (
    PolarsLazyPyarrowTransformer,
)
from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import (
    TransformerTestBase,
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
class TestPolarsLazyPyarrowTransformer(TransformerTestBase):
    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the PolarsLazyPyarrowTransformer class."""
        return PolarsLazyPyarrowTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return pl.LazyFrame as the framework type."""
        return pl.LazyFrame

    def get_connection(self) -> Optional[Any]:
        """Polars Lazy doesn't require a connection."""
        return None

    def test_transform_other_fw_to_fw_missing_polars(self) -> None:
        """Test error handling when polars is not available"""
        test_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        arrow_table = pa.table(test_data)

        with patch(
            "mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_pyarrow_transformer.pl", None
        ):
            with pytest.raises(ImportError, match="Polars is not installed"):
                PolarsLazyPyarrowTransformer.transform_other_fw_to_fw(arrow_table)


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

        df = pl.DataFrame(self.test_data)
        lazy_df = df.lazy()

        eager_result = PolarsPyarrowTransformer.transform_fw_to_other_fw(df)
        lazy_result = PolarsLazyPyarrowTransformer.transform_fw_to_other_fw(lazy_df)

        assert eager_result.equals(lazy_result)

    def test_lazy_vs_eager_polars_conversion(self) -> None:
        """Test that lazy and eager transformers produce equivalent Polars results"""
        from mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer import (
            PolarsPyarrowTransformer,
        )

        arrow_table = pa.table(self.test_data)

        eager_result = PolarsPyarrowTransformer.transform_other_fw_to_fw(arrow_table, None)
        lazy_result = PolarsLazyPyarrowTransformer.transform_other_fw_to_fw(arrow_table)

        lazy_collected = lazy_result.collect()
        assert lazy_collected.equals(eager_result)
