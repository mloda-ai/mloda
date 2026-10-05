"""Test-only stdlib file format ``.toyfmtfile`` on ReadFileFG: a comma-separated header plus comma-separated rows.

The sample listing reads only the first SAMPLE_SIZE_BYTES bytes (a prefix of the header). Listing calls are
recorded per class so tests can count reads. Names carry ``toyfmt`` so the group stays inert elsewhere.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.provider import ReadFileFG, SourceMatch
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

TOY_SUFFIX = ".toyfmtfile"
FOREIGN_SUFFIX = ".toyfmtforeign"
SAMPLE_SIZE = 64


def _cell(text: str) -> Any:
    return int(text) if text.lstrip("-").isdigit() else text


def read_toy_file(path: str, columns: tuple[str, ...]) -> dict[str, list[Any]]:
    """Columnar values of the requested columns."""
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        rows = [line.rstrip("\n").split(",") for line in handle if line.strip()]
    return {name: [_cell(row[header.index(name)]) for row in rows] for name in columns}


def write_toy_file(path: str | Path, columns: dict[str, list[Any]]) -> None:
    names = list(columns)
    rows = zip(*(columns[n] for n in names))
    Path(path).write_text(",".join(names) + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in rows))


def toy_match(path: str | Path) -> SourceMatch:
    return SourceMatch(source=os.path.abspath(path), access=str(path))


class ToyFileFormatFG(ReadFileFG):
    """Stdlib header format; the default FileSource neutral form, plus pyarrow and python-dict loaders."""

    SAMPLE_SIZE_BYTES: ClassVar[int] = SAMPLE_SIZE
    COLUMN_NAMES_CALLS: ClassVar[list[str]] = []
    SAMPLE_CALLS: ClassVar[list[str]] = []

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (TOY_SUFFIX,)

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        cls.COLUMN_NAMES_CALLS.append(str(path))
        with open(path, encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n")
        if not header:
            raise ValueError(f"{path} has an empty header")
        return header.split(",")

    @classmethod
    def sample_column_names(cls, path: str) -> Collection[str] | None:
        cls.SAMPLE_CALLS.append(str(path))
        with open(path, "rb") as handle:
            raw = handle.read(cls.SAMPLE_SIZE_BYTES)
        first = raw.decode("utf-8", errors="replace").split("\n")[0]
        names = first.split(",")
        if b"\n" not in raw and len(raw) == cls.SAMPLE_SIZE_BYTES:
            names = names[:-1]
        return [n for n in names if n]


def _load_pyarrow(group: Any, match: SourceMatch, features: Any) -> pa.Table:
    return pa.table(read_toy_file(match.access, tuple(sorted(features.get_all_names()))))


def _load_dict(group: Any, match: SourceMatch, features: Any) -> dict[str, list[Any]]:
    return read_toy_file(match.access, tuple(sorted(features.get_all_names())))


ToyFileFormatFG.register_loader(PyArrowTable, _load_pyarrow)
ToyFileFormatFG.register_loader(PythonDictFramework, _load_dict)


def toy_file_dac(path: str | Path, handle: str = "toyfmt_handle") -> DataAccessCollection:
    return DataAccessCollection(files={handle: str(path)})
