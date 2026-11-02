"""Columnar test data whose values are validated against declared DataTypes."""

from collections.abc import Mapping, Sequence
from typing import Any

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.value_fit import check_values_fit


class TypedTestData:
    """Columns of values, each checked to fit its declared DataType."""

    def __init__(self, columns: Mapping[str, tuple[DataType, Sequence[Any]]]) -> None:
        problems: list[str] = []
        for name, (data_type, values) in columns.items():
            if not isinstance(data_type, DataType):
                problems.append(f"column '{name}' type must be a DataType, got {type(data_type).__name__}")
            if not isinstance(values, (list, tuple)):
                problems.append(f"column '{name}' values must be a list or tuple, got {type(values).__name__}")
        if problems:
            raise ValueError("; ".join(problems))

        lengths = {name: len(values) for name, (_, values) in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"columns have unequal lengths: {lengths}")

        data = {name: list(values) for name, (_, values) in columns.items()}
        self._types: dict[str, DataType] = {name: data_type for name, (data_type, _) in columns.items()}
        result = check_values_fit(data, self._types)
        if not result.passed:
            raise ValueError("values do not fit their declared types: " + "; ".join(result.failures))
        self._data = data
        self.row_count: int = next(iter(lengths.values()), 0)

    @property
    def types(self) -> dict[str, DataType]:
        """Return a fresh copy of the declared column types."""
        return dict(self._types)

    def to_python_dict(self) -> dict[str, list[Any]]:
        """Return fresh column lists."""
        return {name: list(values) for name, values in self._data.items()}
