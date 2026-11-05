"""Conformance suite for ComputeFramework._output_schema: dict shape, empty and duplicate edges."""

from __future__ import annotations

from typing import Any

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require


class OutputSchemaSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the host's _output_schema; every available framework is served."""

    aspect = "output schema"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    def empty_schema_frame(self, adapter: FrameworkAdapter) -> Any:
        """A frame with no columns; the default goes through a zero-column Arrow table."""
        pa = require("pyarrow", "building Arrow test tables")
        return adapter.from_arrow(pa.table({}))

    def test_reads_dict_sorted_with_python_type_names(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({"b": ["x"], "a": [1], "c": [1.5]}) == (
            ("a", "int"),
            ("b", "str"),
            ("c", "float"),
        )

    def test_empty_schema_yields_none(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        assert framework_instance._output_schema(self.empty_schema_frame(adapter)) is None

    def test_duplicate_column_names_collapse_to_one_entry(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        frame = self.duplicate_column_frame(adapter, DataType.INT64, DataType.STRING)
        result = framework_instance._output_schema(frame)
        assert result is not None
        kept = [pair for pair in result if pair[0] == "a"]
        assert len(kept) == 1
        assert kept[0][1] == framework_instance._extract_column_dtype(frame, "a")
