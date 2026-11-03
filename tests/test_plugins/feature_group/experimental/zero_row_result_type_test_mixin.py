"""Shared framework adapters and helpers for family contract mixins (zero-row result typing).

Family mixins describe inputs as TypedTestData; the framework adapter converts them and reads results back.
feature_group_class is a feature group, or an Implementation for operation families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.framework_adapter import PandasAdapter as _PandasAdapter
from mloda._testing.framework_adapter import PolarsLazyAdapter as _PolarsLazyAdapter
from mloda._testing.framework_adapter import PyArrowAdapter as _PyArrowAdapter
from mloda._testing.framework_adapter import PythonDictAdapter as _PythonDictAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature

if TYPE_CHECKING:
    import pyarrow as pa


def edit_columns(
    data: TypedTestData,
    add: Mapping[str, tuple[DataType, Sequence[Any]]] | None = None,
    drop: Sequence[str] = (),
) -> TypedTestData:
    """A copy of data without the dropped columns and with the added ones (an added name replaces its column)."""
    values = data.to_python_dict()
    columns: dict[str, tuple[DataType, Sequence[Any]]] = {
        name: (data_type, values[name]) for name, data_type in data.types.items() if name not in drop
    }
    columns.update(add or {})
    return TypedTestData(columns)


class ZeroRowResultTypeTestMixin:
    """A calculate helper and the typed-result assertion over the class's framework adapter."""

    # Read by the family tests and by tests/test_plugins/test_mixin_consumer_coverage.py.
    feature_group_class: ClassVar[type[FeatureGroup] | type[Implementation[Any]]]
    adapter: ClassVar[FrameworkAdapter]

    def calculate(
        self, data: TypedTestData, *features: Feature, arrow_types: Mapping[str, pa.DataType] | None = None
    ) -> Any:
        feature_set = FeatureSet()
        for feature in features:
            feature_set.add(feature)
        frame = self.adapter.from_typed(data, arrow_types)
        if issubclass(self.feature_group_class, Implementation):
            return self.feature_group_class.calculate(self.feature_group_class.feature_group, frame, feature_set, None)
        return self.feature_group_class.calculate_feature(frame, feature_set)

    def assert_typed(self, result: Any, column: str, expected_rows: int) -> None:
        assert self.adapter.row_count(result) == expected_rows
        assert not self.adapter.lost_type(result, column), (
            f"{column} lost its type: {self.adapter.column_type(result, column)}"
        )


class PandasZeroRowAdapter(ZeroRowResultTypeTestMixin):
    adapter = _PandasAdapter()


class PyArrowZeroRowAdapter(ZeroRowResultTypeTestMixin):
    adapter = _PyArrowAdapter()


class PolarsLazyZeroRowAdapter(ZeroRowResultTypeTestMixin):
    adapter = _PolarsLazyAdapter()


class PythonDictZeroRowAdapter(ZeroRowResultTypeTestMixin):
    adapter = _PythonDictAdapter()
