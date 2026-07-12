import csv
import re
from typing import Any

from mloda.provider import BaseTransformer

#: A leading ``+`` is deliberately rejected: pyarrow's int64 parser refuses it and falls back to
#: double, so such a column must take the float branch (``_FLOAT_RE`` accepts ``+5``).
_INT_RE = re.compile(r"^-?[0-9]+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_BOOL_MAP: dict[str, bool] = {
    "true": True,
    "false": False,
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
}
#: pyarrow's default ``ConvertOptions.null_values`` minus ``""``: empty cells are already
#: mapped to ``None`` before inference.
_NULL_TOKENS: frozenset[str] = frozenset(
    {
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "n/a",
        "nan",
        "null",
    }
)
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
#: Any ``_INT_RE`` cell of at most this length (sign included) holds at most 18 significant digits,
#: so it always fits int64 and is far under the ``int(str)`` digit cap: ``int()`` needs no guard.
_INT_FAST_LEN = 18


def _as_int64(text: str) -> int | None:
    """The value of a ``_INT_RE`` cell if it fits signed int64, else ``None``.

    Never calls ``int()`` on an oversized digit string: CPython caps ``int(str)`` at
    ``sys.get_int_max_str_digits()`` (4300 by default) and LEADING ZEROS count toward that cap.
    Stripping the sign and the leading zeros first leaves at most 19 significant digits for anything
    that can fit int64 (the smallest 20-digit number already exceeds int64 max), so the surviving
    ``int()`` call is always within the cap.
    """
    negative = text[0] == "-"
    digits = (text[1:] if negative else text).lstrip("0")
    if len(digits) > 19:
        return None
    value = int(digits or "0")
    if negative:
        value = -value
    return value if _INT64_MIN <= value <= _INT64_MAX else None


def _infer_column(cells: list[str | None]) -> list[Any]:
    """Infer a single column's type once (column-wise) and cast its cells.

    Matches pyarrow's default CSV reader:

    - Empty cells (``None`` here) and null tokens become ``None`` in an int/float/bool column;
      in a string column an empty cell stays ``""`` and a null token stays literal text
      (``strings_can_be_null=False``). A column of only empty cells and/or null tokens is all-``None``.
    - The type is decided from the cells that are neither empty nor a null token.
    - An int column with any value outside signed int64 range degrades entirely to float
      (pyarrow's int64 -> double fallback); a value too large for float64 becomes ``inf``.
    - A leading ``+`` is not an int for pyarrow, so such a column degrades to float too.
    """
    typed = [c for c in cells if c is not None and c not in _NULL_TOKENS]

    if not typed:
        return [None] * len(cells)

    if all(_INT_RE.match(c) for c in typed):
        # Parse each cell once; the loop breaks on the first out-of-range value, which degrades the
        # whole column to float below. Short cells take the guard-free int() fast path.
        parsed: list[Any] = []
        for c in cells:
            if c is None or c in _NULL_TOKENS:
                parsed.append(None)
            elif len(c) <= _INT_FAST_LEN:
                parsed.append(int(c))
            else:
                value = _as_int64(c)
                if value is None:
                    break
                parsed.append(value)
        else:
            return parsed
        # Parse the float from the source string: float(int(...)) would raise on huge values.
        return [None if c is None or c in _NULL_TOKENS else float(c) for c in cells]

    if all(_FLOAT_RE.match(c) for c in typed):
        return [None if c is None or c in _NULL_TOKENS else float(c) for c in cells]

    if all(c in _BOOL_MAP for c in typed):
        return [None if c is None or c in _NULL_TOKENS else _BOOL_MAP[c] for c in cells]

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
