"""CSV file format group: stdlib header discovery, a FileSource neutral form and a PyArrowTable loader."""

import csv
from collections.abc import Collection
from typing import Any

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.provider import ComputeFramework, ReadFileFG
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    FileSourcePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_suffixes import CSV_SUFFIXES


class CsvFG(ReadFileFG):
    """Reads CSV files; ``options={"CsvFG": path}`` points it at one file or folder."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return CSV_SUFFIXES

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return next(csv.reader(f), [])

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        # Other frameworks read CSV through pyarrow, whose row split can differ.
        if compute_framework.expected_data_framework() is not dict:
            return None
        if cls._loader_for(compute_framework) is not None:
            return None
        if cls._overrides_load_neutral(CsvFG):
            return None
        file_name = match.access
        with open(file_name, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = 0
            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"Ragged row {row_number} in {file_name}: expected {len(header)} columns, got {len(row)}."
                    )
                count += 1
            return count


def _load_pyarrow(match: SourceMatch, features: Any) -> Any:
    source = FileSource(path=match.access, format="csv", columns=tuple(sorted(features.get_all_names())))
    return FileSourcePyArrowTransformer.transform_fw_to_other_fw(source)


CsvFG.register_loader(PyArrowTable, _load_pyarrow)
