import csv
from typing import Any, Optional

from mloda.provider import BaseTransformer


def _coerce(value: str) -> Any:
    """Coerce a raw CSV cell to ``int``, else ``float``, else keep it as ``str``."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class FileSourceDictTransformer(BaseTransformer):
    """Materialize a ``FileSource`` descriptor into a columnar ``dict[str, list[Any]]``.

    Uses only the stdlib ``csv`` module, so a CSV can be read into PythonDict without pyarrow.
    """

    @classmethod
    def framework(cls) -> Any:
        from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

        return FileSource

    @classmethod
    def other_framework(cls) -> Any:
        return dict

    @classmethod
    def import_fw(cls) -> None:
        import mloda.core.abstract_plugins.components.input_data.file_source  # noqa: F401

    @classmethod
    def import_other_fw(cls) -> None:
        pass

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        if data.format != "csv":
            raise ValueError(f"FileSourceDictTransformer only supports the 'csv' format, got {data.format!r}.")

        with open(data.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            index: dict[str, int] = {}
            for name in data.columns:
                if name not in header:
                    raise ValueError(f"Column {name!r} not found in CSV header of {data.path}")
                index[name] = header.index(name)
            columns: dict[str, list[Any]] = {name: [] for name in data.columns}
            for row in reader:
                for name in data.columns:
                    columns[name].append(_coerce(row[index[name]]))
        return columns

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        raise NotImplementedError
