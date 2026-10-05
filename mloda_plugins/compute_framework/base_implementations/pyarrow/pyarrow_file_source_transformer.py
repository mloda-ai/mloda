from typing import Any

from mloda.core.optional_dependency import require
from mloda.provider import BaseTransformer

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

SUPPORTED_FORMATS = ("csv", "parquet", "json", "feather", "orc")


class FileSourcePyArrowTransformer(BaseTransformer):
    """Materialize a ``FileSource`` descriptor into a ``pa.Table`` using PyArrow readers."""

    @classmethod
    def framework(cls) -> Any:
        from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

        return FileSource

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        import mloda.core.abstract_plugins.components.input_data.file_source  # noqa: F401

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa  # noqa: F401

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        columns = list(data.columns)
        if data.format == "csv":
            from pyarrow import csv as pyarrow_csv

            return pyarrow_csv.read_csv(
                data.path,
                convert_options=pyarrow_csv.ConvertOptions(include_columns=columns),
            )
        if data.format == "parquet":
            pyarrow_parquet = require("pyarrow.parquet", "reading Parquet files")
            return pyarrow_parquet.read_table(data.path, columns=columns)
        if data.format == "json":
            pyarrow_json = require("pyarrow.json", "reading JSON files")
            result = pyarrow_json.read_json(
                data.path,
                parse_options=pyarrow_json.ParseOptions(explicit_schema=None, unexpected_field_behavior="error"),
            )
            return result.select(columns)
        if data.format == "feather":
            pyarrow_ipc = require("pyarrow.ipc", "reading Feather files")
            # Feather V2 is the Arrow IPC file format; ipc.open_file avoids the deprecated pyarrow.feather.read_table.
            with pyarrow_ipc.open_file(data.path) as reader:
                return reader.read_all().select(columns)
        if data.format == "orc":
            pyarrow_orc = require("pyarrow.orc", "reading ORC files")
            return pyarrow_orc.read_table(source=data.path, columns=columns).select(columns)
        raise ValueError(
            f"FileSourcePyArrowTransformer cannot read format {data.format!r}; supported: {', '.join(SUPPORTED_FORMATS)}. "
            "Register a loader for the framework with register_loader, or override load_neutral."
        )
