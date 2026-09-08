from collections.abc import Sequence
from typing import Any
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseMergeEngine
from mloda.provider import BaseFilterEngine, BaseMaskEngine
from mloda.provider import OutputSchema
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_mask_engine import (
    PyArrowMaskEngine,
)

from mloda.user import FeatureName
from mloda.provider import ComputeFramework

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

try:
    import pandas as pd
except ImportError:
    pd = None


def arrow_schema_output_schema(schema: Any) -> OutputSchema | None:
    """Read a pyarrow Schema's names/types once and zip them, rather than calling schema.field()
    per name. Duplicate column names (e.g. an un-aliased join) collapse to the first occurrence.

    Shared with IcebergFramework, which reaches the same PyArrow interchange shape post-transform.
    """
    names = schema.names
    if not names:
        return None
    seen: dict[str, str] = {}
    for name, arrow_type in zip(names, schema.types):
        seen.setdefault(name, str(arrow_type))
    return tuple((name, seen[name]) for name in sorted(seen, key=str))


def arrow_schema_field_type(schema: Any, column_name: str) -> Any | None:
    """Return column_name's arrow type from its first occurrence; unlike schema.field(), never raises on duplicates."""
    for name, arrow_type in zip(schema.names, schema.types):
        if name == column_name:
            return arrow_type
    return None


class PyArrowTable(ComputeFramework):
    @staticmethod
    def is_available() -> bool:
        """Check if PyArrow is installed and available."""
        try:
            import pyarrow  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        return pa.Table

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PyArrowMergeEngine

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        return PyArrowFilterEngine

    @classmethod
    def mask_engine(cls) -> type[BaseMaskEngine]:
        return PyArrowMaskEngine

    def select_data_by_column_names(
        self,
        data: Any,
        selected_feature_names: Sequence[FeatureName],
        column_ordering: str | None = None,
        request_feature_order: list[str] | None = None,
    ) -> Any:
        column_names = set(data.schema.names)
        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering, request_feature_order=request_feature_order
        )
        return data.select([f for f in _selected_feature_names])

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.schema.names)

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        arrow_type = arrow_schema_field_type(data.schema, column_name)
        if arrow_type is None:
            return None
        return str(arrow_type)

    def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
        arrow_type = arrow_schema_field_type(data.schema, column_name)
        if arrow_type is None:
            return None
        return DataType.from_arrow_type_safe(arrow_type)

    def _output_schema(self, data: Any) -> OutputSchema | None:
        if isinstance(data, dict):
            return super()._output_schema(data)
        return arrow_schema_output_schema(data.schema)

    def transform(
        self,
        data: Any,
        feature_names: Sequence[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to table"""
            return pa.table(data)

        if isinstance(data, pa.ChunkedArray) or isinstance(data, pa.Array):
            """Added data: Add column to table"""
            if len(feature_names) == 1:
                return self.data.append_column(next(iter(feature_names)), data)
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")
