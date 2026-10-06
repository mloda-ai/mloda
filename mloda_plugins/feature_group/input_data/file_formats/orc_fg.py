"""ORC file format group: typed schema from the footer, row counts and a PyArrow read."""

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
from mloda_plugins.feature_group.input_data.file_suffixes import ORC_SUFFIXES


_read = pyarrow_file_loader("orc")


class OrcFG(ReadFileFG):
    """Reads ORC files into a pyarrow Table."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return ORC_SUFFIXES

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        return list(cls.describe_columns(SourceMatch(source=path, access=path)))

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        pyarrow_orc = require("pyarrow.orc", "reading ORC files")
        return {
            field.name: DataType.from_arrow_type_safe(field.type) for field in pyarrow_orc.ORCFile(match.access).schema
        }

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return _read(cls, match, features)

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        if cls._overrides_load_neutral(OrcFG):
            return None
        pyarrow_orc = require("pyarrow.orc", "reading ORC files")
        nrows: int = pyarrow_orc.ORCFile(match.access).nrows
        return nrows


OrcFG.register_loader(PyArrowTable, _read)
