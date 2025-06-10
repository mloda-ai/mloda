from typing import Any, Optional

from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

try:
    import pyarrow as pa
except ImportError:
    pa = None


class PolarsLazyPyarrowTransformer(BaseTransformer):
    """
    Transformer for converting between Polars LazyFrame and PyArrow Table.

    This transformer handles bidirectional conversion between Polars LazyFrame
    and PyArrow Table data structures. For LazyFrame to PyArrow conversion,
    it collects the lazy frame first, then converts to PyArrow.
    """

    @classmethod
    def framework(cls) -> Any:
        if pl is None:
            return NotImplementedError
        return pl.LazyFrame

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        import polars as pl

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform a Polars LazyFrame to a PyArrow Table.

        This method collects the LazyFrame first, then uses Polars' native
        to_arrow() method for efficient conversion.
        """
        # Collect the lazy frame first, then convert to arrow
        collected_df = data.collect()
        return collected_df.to_arrow()

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform a PyArrow Table to a Polars LazyFrame.

        This method uses Polars' native from_arrow() method to create a DataFrame,
        then converts it to a LazyFrame.
        """
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        # Convert PyArrow to DataFrame, then make it lazy
        df = pl.from_arrow(data)
        return df.lazy()  # type: ignore
