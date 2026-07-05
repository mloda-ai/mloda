from typing import Any, Optional

import pyarrow as pa
from pyarrow import csv as pyarrow_csv

from mloda.provider import BaseTransformer


class FileSourcePyArrowTransformer(BaseTransformer):
    """Materialize a ``FileSource`` descriptor into a ``pa.Table`` using PyArrow's CSV reader."""

    @classmethod
    def framework(cls) -> Any:
        from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

        return FileSource

    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        import mloda.core.abstract_plugins.components.input_data.file_source  # noqa: F401

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa  # noqa: F401

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        if data.format != "csv":
            raise ValueError(f"FileSourcePyArrowTransformer only supports the 'csv' format, got {data.format!r}.")
        return pyarrow_csv.read_csv(data.path).select(list(data.columns))

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        raise NotImplementedError
