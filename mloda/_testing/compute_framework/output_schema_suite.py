"""Conformance suite for ComputeFramework._output_schema: the dict interchange shape and the empty and duplicate edges."""

from __future__ import annotations

from typing import Any

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require
from mloda.user import ParallelizationMode

_SAMPLE_VALUES: dict[DataType, Any] = {
    DataType.INT64: 1,
    DataType.STRING: "x",
    DataType.DOUBLE: 1.5,
    DataType.BOOLEAN: True,
}


class OutputSchemaSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the host's _output_schema; every available framework is served."""

    aspect = "output schema"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    @pytest.fixture
    def framework_instance(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> ComputeFramework:
        instance = framework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        if adapter.connection is not None:
            instance.set_framework_connection_object(adapter.connection)
        return instance

    def empty_schema_frame(self, adapter: FrameworkAdapter) -> Any:
        """A frame with no columns; the default goes through a zero-column Arrow table."""
        pa = require("pyarrow", "building Arrow test tables")
        return adapter.from_arrow(pa.table({}))

    def duplicate_column_frame(self, adapter: FrameworkAdapter, first_type: DataType, second_type: DataType) -> Any:
        """A frame whose two columns are both named "a", typed first_type then second_type."""
        pa = require("pyarrow", "building Arrow test tables")
        table = pa.table(
            [
                pa.array([_SAMPLE_VALUES[first_type]], type=DataType.to_arrow_type(first_type)),
                pa.array([_SAMPLE_VALUES[second_type]], type=DataType.to_arrow_type(second_type)),
            ],
            names=["a", "b"],
        )
        return adapter.from_arrow(table.rename_columns(["a", "a"]))

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
        frame = self.duplicate_column_frame(adapter, DataType.INT64, DataType.INT64)
        result = framework_instance._output_schema(frame)
        assert result is not None
        assert len([pair for pair in result if pair[0] == "a"]) == 1
