"""CSV file format group: stdlib header discovery, a FileSource neutral form and a PyArrowTable implementation."""

import csv
from collections.abc import Collection
from typing import Any

from mloda.core.abstract_plugins.implementation import loaded_implementations
from mloda.provider import ComputeFramework, FeatureSet, Implementation, is_match_abort, ReadFileFG, SourceMatch
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
    pyarrow_file_loader,
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
            try:
                return next(csv.reader(f), [])
            except csv.Error as exc:
                if is_match_abort(exc):
                    raise
                raise ValueError(str(exc)) from exc

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        # Runs without an engine: consults every loaded implementation, not the governed set.
        # Other frameworks read CSV through pyarrow, whose row split can differ.
        if compute_framework.expected_data_framework() is not dict:
            return None
        if cls.select_implementation(compute_framework, match, loaded_implementations()) is not None:
            return None
        if cls.overrides_load_neutral(CsvFG):
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


_read = pyarrow_file_loader("csv")


class CsvPyArrowTable(Implementation[CsvFG]):
    feature_group = CsvFG
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[CsvFG], data: SourceMatch, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return _read(group, data, features)
