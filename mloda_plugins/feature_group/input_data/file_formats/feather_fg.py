"""Feather file format group: typed schema from the IPC footer, row counts and a PyArrow read."""

from collections.abc import Collection
from typing import Any

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.optional_dependency import require
from mloda.provider import ComputeFramework, ReadFileFG
from mloda.user import DataType
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    pyarrow_file_loader,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_suffixes import FEATHER_SUFFIXES


_read = pyarrow_file_loader("feather")


class FeatherFG(ReadFileFG):
    """Reads Feather (Arrow IPC) files into a pyarrow Table."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return FEATHER_SUFFIXES

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        return list(cls.describe_columns(SourceMatch(source=path, access=path)))

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        pyarrow_ipc = require("pyarrow.ipc", "reading Feather files")
        with pyarrow_ipc.open_file(match.access) as reader:
            return {field.name: DataType.from_arrow_type_safe(field.type) for field in reader.schema}

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return _read(match, features)

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        if cls._overrides_load_neutral(FeatherFG):
            return None
        pyarrow_dataset = require("pyarrow.dataset", "counting Feather rows")
        pyarrow_fs = require("pyarrow.fs", "counting Feather rows")
        # One local file, not a dataset: a directory or missing path raises OSError, no URI fetch.
        fragment = pyarrow_dataset.IpcFileFormat().make_fragment(match.access, filesystem=pyarrow_fs.LocalFileSystem())
        count: int = fragment.count_rows()
        return count


FeatherFG.register_loader(PyArrowTable, _read)
