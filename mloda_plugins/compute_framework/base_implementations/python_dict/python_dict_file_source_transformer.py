import csv
import re
from typing import Any

from mloda.provider import BaseTransformer

_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_BOOL_MAP: dict[str, bool] = {
    "true": True,
    "false": False,
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
}


def _infer_column(cells: list[str | None]) -> list[Any]:
    """Infer a single column's type once (column-wise) and cast its cells.

    Empty-cell semantics match pyarrow's default CSV reader
    (``strings_can_be_null=False``): a missing cell in a numeric/bool column
    becomes ``None``, while a missing cell in a string column becomes ``""``
    (empty string). An all-empty column (no present cells) stays all ``None``.

    Null tokens (``NA``, ``NaN``, ``null``) inside an otherwise numeric column
    are not recognized: such a column stays string, whereas pyarrow parses
    those tokens as nulls.

    Integers beyond int64 range stay exact arbitrary-precision Python ints here,
    whereas pyarrow's CSV reader yields a lossy ``float64``.

    Both are deliberate divergences from pyarrow, tracked as a follow-up in
    issue #662.
    """
    present = [c for c in cells if c is not None]

    if not present:
        return list(cells)

    if all(_INT_RE.match(c) for c in present):
        return [None if c is None else int(c) for c in cells]

    if all(_FLOAT_RE.match(c) for c in present):
        return [None if c is None else float(c) for c in cells]

    if all(c in _BOOL_MAP for c in present):
        return [None if c is None else _BOOL_MAP[c] for c in cells]

    return ["" if c is None else c for c in cells]


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
            header = next(reader, [])

            index: dict[str, int] = {}
            for name in data.columns:
                occurrences = header.count(name)
                if occurrences > 1:
                    raise ValueError(
                        f"Duplicate column {name!r} in CSV header of {data.path}; cannot resolve which one to read."
                    )
                if occurrences == 0:
                    raise ValueError(f"Column {name!r} not found in CSV header of {data.path}")
                index[name] = header.index(name)

            raw: dict[str, list[str | None]] = {name: [] for name in data.columns}
            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"Ragged row {row_number} in {data.path}: expected {len(header)} columns, got {len(row)}."
                    )
                for name in data.columns:
                    cell = row[index[name]]
                    raw[name].append(None if cell == "" else cell)

        return {name: _infer_column(cells) for name, cells in raw.items()}
