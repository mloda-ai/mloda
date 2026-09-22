"""Shared contract: a feature group's result column keeps a concrete type on zero-row input.

Family mixins describe inputs as pyarrow Tables; a framework adapter converts them and reads row counts and
column types back, so each framework's hooks are written once, not once per family."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature
from mloda.user.python_dict import row_count as python_dict_row_count

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    POLARS_AVAILABLE = False


class ZeroRowResultTypeTestMixin:
    """Family-agnostic hooks, a calculate helper and the typed-result assertion."""

    # Read by the family tests and by tests/test_plugins/test_mixin_consumer_coverage.py.
    feature_group_class: ClassVar[type[FeatureGroup]]
    untyped_column_types: ClassVar[tuple[Any, ...]]

    @abstractmethod
    def from_arrow(self, table: pa.Table) -> Any:
        raise NotImplementedError

    @abstractmethod
    def row_count(self, result: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def column_type(self, result: Any, column: str) -> Any:
        raise NotImplementedError

    def calculate(self, table: pa.Table, feature: Feature) -> Any:
        feature_set = FeatureSet()
        feature_set.add(feature)
        return self.feature_group_class.calculate_feature(self.from_arrow(table), feature_set)

    def assert_typed(self, result: Any, column: str, expected_rows: int) -> None:
        assert self.row_count(result) == expected_rows
        column_type = self.column_type(result, column)
        assert column_type not in self.untyped_column_types, f"{column} lost its type: {column_type}"


class PandasZeroRowAdapter(ZeroRowResultTypeTestMixin):
    """pandas DataFrame containers; object dtype means the column lost its type."""

    untyped_column_types = (np.dtype(object),)

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pandas()

    def row_count(self, result: Any) -> int:
        return len(result)

    def column_type(self, result: Any, column: str) -> Any:
        return result[column].dtype


class PyArrowZeroRowAdapter(ZeroRowResultTypeTestMixin):
    """pyarrow Table containers; the null type means the column lost its type."""

    untyped_column_types = (pa.null(),)

    def from_arrow(self, table: pa.Table) -> Any:
        return table

    def row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def column_type(self, result: Any, column: str) -> Any:
        return result.column(column).type


class PolarsLazyZeroRowAdapter(ZeroRowResultTypeTestMixin):
    """polars LazyFrame containers, read after collect; Null and Object mean the column lost its type."""

    untyped_column_types = (pl.Null, pl.Object) if POLARS_AVAILABLE else ()

    def from_arrow(self, table: pa.Table) -> Any:
        return pl.LazyFrame(table)

    def row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def column_type(self, result: Any, column: str) -> Any:
        return result.collect()[column].dtype


class PythonDictZeroRowAdapter(ZeroRowResultTypeTestMixin):
    """Columnar dicts hold dtype-less lists, so the check reduces to: the result column exists with zero rows."""

    untyped_column_types = ()

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pydict()

    def row_count(self, result: Any) -> int:
        return python_dict_row_count(result)

    def column_type(self, result: Any, column: str) -> Any:
        return type(result[column])
