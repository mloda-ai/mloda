"""Pure Python columns with one declared DataType each, validated on construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mloda._testing.value_fit import check_values_fit
from mloda.core.abstract_plugins.components.data_types import DataType


class TypedTestData:
    """Columnar test data whose values are checked against the declared types."""

    def __init__(self, columns: Mapping[str, tuple[DataType, Sequence[Any]]]) -> None:
        for name, (data_type, values) in columns.items():
            if not isinstance(data_type, DataType):
                raise ValueError(f"column '{name}': declared type {data_type!r} is not a DataType")
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"column '{name}': values must be a list or tuple")
        if len({len(values) for _, values in columns.values()}) > 1:
            raise ValueError("columns have unequal lengths")

        self._columns = {name: list(values) for name, (_, values) in columns.items()}
        self._types = {name: data_type for name, (data_type, _) in columns.items()}

        result = check_values_fit(self._columns, self._types)
        if not result.passed:
            raise ValueError("; ".join(result.failures))

    @property
    def types(self) -> dict[str, DataType]:
        return dict(self._types)

    @property
    def row_count(self) -> int:
        return len(next(iter(self._columns.values()), []))

    def to_python_dict(self) -> dict[str, list[Any]]:
        return {name: list(values) for name, values in self._columns.items()}
