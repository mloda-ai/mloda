"""JSON file format group: sampled and full column discovery with PyArrow, and a PyArrow read."""

import io
from collections.abc import Collection
from typing import Any

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.utils import is_match_abort
from mloda.core.optional_dependency import require
from mloda.provider import ReadFileFG
from mloda.user import DataType
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_suffixes import JSON_SUFFIXES


def _read(match: SourceMatch, features: Any) -> Any:
    pyarrow_json = require("pyarrow.json", "reading JSON files")
    result = pyarrow_json.read_json(
        match.access,
        parse_options=pyarrow_json.ParseOptions(explicit_schema=None, unexpected_field_behavior="error"),
    )
    return result.select(list(features.get_all_names()))


class JsonFG(ReadFileFG):
    """Reads line-delimited JSON files into a pyarrow Table."""

    SAMPLE_SIZE_BYTES = 65536

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return JSON_SUFFIXES

    @classmethod
    def sample_column_names(cls, path: str) -> Collection[str] | None:
        pyarrow_json = require("pyarrow.json", "reading JSON files")
        with open(path, "rb") as f:
            data = f.read(cls.SAMPLE_SIZE_BYTES + 1)
        if len(data) > cls.SAMPLE_SIZE_BYTES:
            cut = data[: cls.SAMPLE_SIZE_BYTES].rfind(b"\n")
            if cut < 0:
                return None
            data = data[:cut]
        try:
            table = pyarrow_json.read_json(io.BytesIO(data))
        except ValueError as exc:
            if is_match_abort(exc):
                raise
            return None
        names: list[str] = table.schema.names
        return names

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        return list(cls.describe_columns(SourceMatch(source=path, access=path)))

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        pyarrow_json = require("pyarrow.json", "reading JSON files")
        # block_size is a chunking granularity, not a row-count sample; this parses the whole file.
        table = pyarrow_json.read_json(match.access, read_options=pyarrow_json.ReadOptions(block_size=65536))
        return {field.name: DataType.from_arrow_type_safe(field.type) for field in table.schema}

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return _read(match, features)


JsonFG.register_loader(PyArrowTable, _read)
