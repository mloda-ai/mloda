from typing import Any, Optional

from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


class PandasPyarrowTransformer(BaseTransformer):
    """
    Transformer for converting between Pandas DataFrame and PyArrow Table.

    This transformer handles bidirectional conversion between Pandas DataFrame
    and PyArrow Table data structures, ensuring proper data type handling and
    metadata management during the transformation process.
    """

    @classmethod
    def framework(cls) -> Any:
        if pd is None:
            return NotImplementedError
        return pd.DataFrame

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        import pandas as pd

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform a Pandas DataFrame to a PyArrow Table.

        This method converts a Pandas DataFrame to a PyArrow Table and
        removes the pandas-specific schema metadata to ensure clean conversion.
        """
        # drop pandas schema metadata
        pyarrow_table = pa.Table.from_pandas(data)
        schema = pyarrow_table.schema
        metadata = schema.metadata.copy() if schema.metadata else {}
        metadata.pop(b"pandas", None)
        new_schema = schema.with_metadata(metadata)
        return pa.Table.from_arrays(pyarrow_table.columns, schema=new_schema)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        return pa.Table.to_pandas(data)
