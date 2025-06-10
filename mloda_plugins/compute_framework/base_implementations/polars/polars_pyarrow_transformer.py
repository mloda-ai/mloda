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


class PolarsPyarrowTransformer(BaseTransformer):
    """
    Transformer for converting between Polars DataFrame and PyArrow Table.

    This transformer handles bidirectional conversion between Polars DataFrame
    and PyArrow Table data structures, leveraging Polars' native PyArrow integration
    for efficient zero-copy operations where possible.
    """

    @classmethod
    def framework(cls) -> Any:
        if pl is None:
            return NotImplementedError
        return pl.DataFrame

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
        Transform a Polars DataFrame to a PyArrow Table.

        This method uses Polars' native to_arrow() method for efficient conversion.
        """
        return data.to_arrow()

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform a PyArrow Table to a Polars DataFrame.

        This method uses Polars' native from_arrow() method for efficient conversion.
        """
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.from_arrow(data)
