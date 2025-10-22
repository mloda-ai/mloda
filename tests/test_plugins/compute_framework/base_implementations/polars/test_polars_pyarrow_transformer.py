from typing import Any, Optional, Type
import pytest

from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import TransformerTestBase
from mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer import (
    PolarsPyarrowTransformer,
)

try:
    import polars as pl
    import pyarrow as pa

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars or PyArrow not available")
class TestPolarsPyarrowTransformer(TransformerTestBase):
    """Test Polars PyArrow transformer using base test class."""

    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the Polars transformer class."""
        return PolarsPyarrowTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return Polars' framework type."""
        return pl.DataFrame

    def get_connection(self) -> Optional[Any]:
        """Return None as Polars doesn't need a connection."""
        return None

    def test_check_imports(self) -> None:
        """Test that import checks work correctly."""
        assert PolarsPyarrowTransformer.check_imports() is True

    def test_transform_method_with_correct_orientation(self) -> None:
        """Test the generic transform method with correct framework orientation."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        arrow_result = PolarsPyarrowTransformer.transform(pl.DataFrame, pa.Table, df, None)
        assert isinstance(arrow_result, pa.Table)

        polars_result = PolarsPyarrowTransformer.transform(pa.Table, pl.DataFrame, arrow_result, None)
        assert isinstance(polars_result, pl.DataFrame)
        assert polars_result.equals(df)

    def test_transform_method_with_unsupported_frameworks(self) -> None:
        """Test that transform method raises error for unsupported frameworks."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError):
            PolarsPyarrowTransformer.transform(list, dict, df, None)
