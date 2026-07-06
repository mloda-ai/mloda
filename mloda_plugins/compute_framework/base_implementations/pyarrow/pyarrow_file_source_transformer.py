from typing import Any

import pyarrow as pa

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
        from pyarrow import csv as pyarrow_csv

        return pyarrow_csv.read_csv(
            data.path,
            convert_options=pyarrow_csv.ConvertOptions(include_columns=list(data.columns)),
        )

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
        raise NotImplementedError
