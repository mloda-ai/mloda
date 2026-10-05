"""Parquet file format group: typed schema from the footer, metadata row counts and a PyArrow read."""

from collections.abc import Collection
from typing import Any

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.optional_dependency import require
from mloda.provider import ComputeFramework, ReadFileFG
from mloda.user import DataType
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    FileSourcePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_suffixes import PARQUET_SUFFIXES


def _read(match: SourceMatch, features: Any) -> Any:
    source = FileSource(path=match.access, format="parquet", columns=tuple(sorted(features.get_all_names())))
    return FileSourcePyArrowTransformer.transform_fw_to_other_fw(source)


class ParquetFG(ReadFileFG):
    """Reads Parquet files into a pyarrow Table."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return PARQUET_SUFFIXES

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        return list(cls.describe_columns(SourceMatch(source=path, access=path)))

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        pyarrow_parquet = require("pyarrow.parquet", "reading Parquet files")
        with pyarrow_parquet.ParquetFile(match.access) as parquet_file:
            return {field.name: DataType.from_arrow_type_safe(field.type) for field in parquet_file.schema_arrow}

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return _read(match, features)

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        if cls._overrides_load_neutral(ParquetFG):
            return None
        pyarrow_parquet = require("pyarrow.parquet", "reading Parquet files")
        with pyarrow_parquet.ParquetFile(match.access) as parquet_file:
            num_rows: int = parquet_file.metadata.num_rows
        return num_rows


ParquetFG.register_loader(PyArrowTable, _read)
