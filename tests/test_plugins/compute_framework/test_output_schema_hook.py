"""Tests for ComputeFramework._output_schema: sorted (column, dtype) pairs, the dict interchange
shape on every framework, degradation to None on failure, and that lazy frames (polars/duckdb/sqlite)
are never materialized to build the schema.
"""

import sqlite3
import types
from typing import Any, cast
from unittest.mock import Mock

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import FeatureGroup
from mloda.user import Feature, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_plugins.compute_framework.base_implementations.dict_interchange_output_schema_test_mixin import (
    DictInterchangeOutputSchemaTestMixin,
)

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from pyspark.sql.types import StructField, StructType
    from pyspark.sql.types import DoubleType as SparkDoubleType
    from pyspark.sql.types import IntegerType as SparkIntegerType
    from pyspark.sql.types import StringType as SparkStringType
except ImportError:
    StructField = None
    StructType = None
    SparkDoubleType = None
    SparkIntegerType = None
    SparkStringType = None

try:
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.types import DoubleType as IcebergDoubleType
    from pyiceberg.types import LongType as IcebergLongType
    from pyiceberg.types import NestedField
    from pyiceberg.types import StringType as IcebergStringType
    from pyiceberg.types import StructType as IcebergStructType
except ImportError:
    IcebergSchema = None  # type: ignore[assignment, misc]
    PyIcebergTable = None  # type: ignore[assignment, misc]
    IcebergDoubleType = None  # type: ignore[assignment, misc]
    IcebergLongType = None  # type: ignore[assignment, misc]
    NestedField = None  # type: ignore[assignment, misc]
    IcebergStringType = None  # type: ignore[assignment, misc]
    IcebergStructType = None  # type: ignore[assignment, misc]

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


class TestPythonDictOutputSchema:
    def test_sorted_columns_with_dtypes(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema({"b": ["x", "y"], "a": [1, 2]}) == (("a", "int"), ("b", "str"))

    def test_all_none_column_yields_none_dtype(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema({"a": [None, None]}) == (("a", None),)

    def test_no_columns_yields_none(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema({}) is None

    def test_non_native_shape_yields_none(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema([1, 2, 3]) is None


class TestPythonDictRowWiseOutputSchemaSinglePass:
    """Perf regression: the row-wise list[dict] shape must resolve dtypes in a single pass
    over the rows, not by rebuilding a full per-column values list for every column."""

    def test_output_schema_does_not_call_column_values_for_list_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_if_list(data: Any, column_name: str) -> list[Any]:
            if isinstance(data, list):
                raise AssertionError("_column_values must not be called by _output_schema for row-wise list input")
            return data.get(column_name, [])  # type: ignore[no-any-return]

        monkeypatch.setattr(PythonDictFramework, "_column_values", staticmethod(_raise_if_list))

        rows = [{"b": "x", "a": 1}, {"b": "y", "a": 2}]
        assert PythonDictFramework()._output_schema(rows) == (("a", "int"), ("b", "str"))

    def test_non_string_keys_are_stringified(self) -> None:
        """MINOR: the row-wise branch's final tuple emits (name, ...) instead of (str(name), ...),
        so a non-str key (e.g. int 1) leaks through unstringified, unlike the base class."""
        assert PythonDictFramework()._output_schema([{1: "x", "a": 2}]) == (("1", "str"), ("a", "int"))


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowOutputSchema:
    def test_sorted_columns_with_arrow_dtypes(self) -> None:
        table = pa.table({"b": ["x"], "a": [1]})
        assert PyArrowTable()._output_schema(table) == (("a", "int64"), ("b", "string"))

    def test_dict_interchange_shape(self) -> None:
        assert PyArrowTable()._output_schema({"b": ["x"], "a": [1]}) == (("a", "int"), ("b", "str"))

    def test_duplicate_column_names_collapse_to_one_entry(self) -> None:
        table = pa.table({"a": [1], "b": [2]}).rename_columns(["a", "a"])
        result = PyArrowTable()._output_schema(table)
        assert result is not None
        a_entries = [pair for pair in result if pair[0] == "a"]
        assert len(a_entries) == 1

    def test_empty_schema_yields_none(self) -> None:
        assert PyArrowTable()._output_schema(pa.table({})) is None

    def test_output_schema_reads_schema_once_and_never_calls_field(self) -> None:
        """Perf regression: schema must be read once total, and schema.field() (the O(columns)
        lookup the issue names) must never be called. pa.Table/pa.Schema are immutable
        C-extension types and cannot be monkeypatched directly, so a lightweight stand-in wraps
        schema property reads and forbids field() instead.
        """

        class _FieldForbiddenSchema:
            def __init__(self, real_schema: Any) -> None:
                self.names = real_schema.names
                self.types = real_schema.types

            def field(self, name: Any) -> Any:
                raise AssertionError("schema.field() must not be called by _output_schema")

        class _SchemaAccessCountingTable:
            def __init__(self, real_table: Any) -> None:
                self._schema = _FieldForbiddenSchema(real_table.schema)
                self.schema_reads = 0

            @property
            def schema(self) -> Any:
                self.schema_reads += 1
                return self._schema

        wrapper = _SchemaAccessCountingTable(pa.table({"c": ["x"], "b": [1], "a": [1.5]}))
        assert PyArrowTable()._output_schema(wrapper) == (("a", "double"), ("b", "int64"), ("c", "string"))
        assert wrapper.schema_reads == 1


@pytest.mark.skipif(StructType is None, reason="PySpark is not installed. Skipping this test.")
class TestSparkOutputSchema:
    """SparkFramework._output_schema reads schema.fields once; a real SparkSession/JVM is not
    needed since StructType/StructField construct without one."""

    def test_sorted_columns_with_spark_dtypes(self) -> None:
        struct = StructType([StructField("b", SparkStringType()), StructField("a", SparkIntegerType())])
        data = types.SimpleNamespace(schema=struct)
        assert SparkFramework()._output_schema(data) == (("a", "IntegerType()"), ("b", "StringType()"))

    def test_dict_interchange_shape(self) -> None:
        assert SparkFramework()._output_schema({"b": ["x"], "a": [1]}) == (("a", "int"), ("b", "str"))

    def test_empty_schema_yields_none(self) -> None:
        data = types.SimpleNamespace(schema=StructType([]))
        assert SparkFramework()._output_schema(data) is None

    def test_duplicate_column_names_collapse_to_one_entry(self) -> None:
        struct = StructType([StructField("a", SparkIntegerType()), StructField("a", SparkStringType())])
        data = types.SimpleNamespace(schema=struct)
        result = SparkFramework()._output_schema(data)
        assert result is not None
        a_entries = [pair for pair in result if pair[0] == "a"]
        assert len(a_entries) == 1

    def test_output_schema_does_not_call_dunder_getitem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Perf regression: StructType.__getitem__(name) does a linear scan over fields, so
        _output_schema must never index the schema by name."""

        def _raise_if_called(self: Any, key: Any) -> Any:
            raise AssertionError("StructType.__getitem__ must not be called by _output_schema")

        monkeypatch.setattr(StructType, "__getitem__", _raise_if_called)

        struct = StructType(
            [
                StructField("c", SparkStringType()),
                StructField("b", SparkIntegerType()),
                StructField("a", SparkDoubleType()),
            ]
        )
        data = types.SimpleNamespace(schema=struct)
        assert SparkFramework()._output_schema(data) == (
            ("a", "DoubleType()"),
            ("b", "IntegerType()"),
            ("c", "StringType()"),
        )


@pytest.mark.skipif(IcebergSchema is None or pa is None, reason="PyIceberg or PyArrow is not installed.")
class TestIcebergOutputSchema:
    """IcebergFramework._output_schema reads schema.column_names once (nested fields included,
    e.g. "b.c") for a native IcebergTable, then schema.find_field(name) per name (a cached O(1)
    lookup, not a rebuild); or delegates to arrow_schema_output_schema for the PyArrow interchange
    shape reached after transform()."""

    @staticmethod
    def _mock_table(schema: Any) -> Any:
        mock_table = Mock(spec=PyIcebergTable)
        mock_table.schema.return_value = schema
        return mock_table

    def test_sorted_columns_with_iceberg_types(self) -> None:
        schema = IcebergSchema(NestedField(1, "b", IcebergStringType()), NestedField(2, "a", IcebergLongType()))
        assert IcebergFramework()._output_schema(self._mock_table(schema)) == (("a", "long"), ("b", "string"))

    def test_dict_interchange_shape(self) -> None:
        assert IcebergFramework()._output_schema({"b": ["x"], "a": [1]}) == (("a", "int"), ("b", "str"))

    def test_empty_schema_yields_none(self) -> None:
        assert IcebergFramework()._output_schema(self._mock_table(IcebergSchema())) is None

    def test_nested_struct_columns_use_dotted_paths(self) -> None:
        """schema.column_names (unlike top-level schema.fields) includes nested leaves as dotted
        paths, matching what the pre-existing _extract_column_names/_extract_column_dtype path
        already reported for a native Iceberg table."""
        schema = IcebergSchema(
            NestedField(1, "a", IcebergLongType()),
            NestedField(
                2,
                "b",
                IcebergStructType(NestedField(3, "c", IcebergStringType()), NestedField(4, "d", IcebergLongType())),
            ),
        )
        result = IcebergFramework()._output_schema(self._mock_table(schema))
        assert result is not None
        names = dict(result)
        assert names["a"] == "long"
        assert names["b.c"] == "string"
        assert names["b.d"] == "long"
        assert "b" in names

    def test_pyarrow_interchange_shape(self) -> None:
        """After transform(), a plain PyArrow table (not yet an Iceberg table) is a valid shape."""
        table = pa.table({"b": ["x"], "a": [1]})
        assert IcebergFramework()._output_schema(table) == (("a", "int64"), ("b", "string"))

    def test_pyarrow_interchange_duplicate_columns_collapse_to_one_entry(self) -> None:
        table = pa.table({"a": [1], "b": [2]}).rename_columns(["a", "a"])
        result = IcebergFramework()._output_schema(table)
        assert result is not None
        a_entries = [pair for pair in result if pair[0] == "a"]
        assert len(a_entries) == 1

    def test_output_schema_calls_schema_method_once(self) -> None:
        """Perf regression: Table.schema() must be called once total, not once per column."""
        schema = IcebergSchema(
            NestedField(1, "c", IcebergStringType()),
            NestedField(2, "b", IcebergLongType()),
            NestedField(3, "a", IcebergDoubleType()),
        )
        mock_table = self._mock_table(schema)
        assert IcebergFramework()._output_schema(mock_table) == (("a", "double"), ("b", "long"), ("c", "string"))
        assert mock_table.schema.call_count == 1

    def test_output_schema_reads_column_names_property_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Perf regression: schema.column_names rebuilds its list on every read, so _output_schema
        must read it once total, not once per column."""
        call_count = 0
        original_get_column_names = cast(property, IcebergSchema.__dict__["column_names"]).fget
        assert original_get_column_names is not None

        def _counting_column_names(self: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original_get_column_names(self)

        monkeypatch.setattr(IcebergSchema, "column_names", property(_counting_column_names))

        schema = IcebergSchema(
            NestedField(1, "c", IcebergStringType()),
            NestedField(2, "b", IcebergLongType()),
            NestedField(3, "a", IcebergDoubleType()),
        )
        assert IcebergFramework()._output_schema(self._mock_table(schema)) == (
            ("a", "double"),
            ("b", "long"),
            ("c", "string"),
        )
        assert call_count == 1


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsEagerOutputSchema:
    def test_sorted_columns_with_polars_dtypes(self) -> None:
        df = pl.DataFrame({"b": ["x"], "a": [1]})
        assert PolarsDataFrame()._output_schema(df) == (("a", "Int64"), ("b", "String"))

    def test_dict_interchange_shape(self) -> None:
        assert PolarsDataFrame()._output_schema({"b": ["x"], "a": [1]}) == (("a", "int"), ("b", "str"))

    def test_empty_schema_yields_none(self) -> None:
        assert PolarsDataFrame()._output_schema(pl.DataFrame()) is None

    def test_output_schema_does_not_index_columns_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Perf regression: data[column_name] and `in data.columns` are each O(columns) per call,
        so _output_schema must read `.schema` instead of indexing per column."""

        def _raise_if_called(self: Any, key: Any) -> Any:
            raise AssertionError("DataFrame.__getitem__ must not be called by _output_schema")

        monkeypatch.setattr(pl.DataFrame, "__getitem__", _raise_if_called)

        df = pl.DataFrame({"c": ["x"], "b": [1], "a": [1.5]})
        assert PolarsDataFrame()._output_schema(df) == (("a", "Float64"), ("b", "Int64"), ("c", "String"))

    def test_output_schema_reads_schema_property_at_most_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0
        original_get_schema = cast(property, pl.DataFrame.__dict__["schema"]).fget
        assert original_get_schema is not None

        def _counting_schema(self: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original_get_schema(self)

        monkeypatch.setattr(pl.DataFrame, "schema", property(_counting_schema))

        df = pl.DataFrame({"c": ["x"], "b": [1], "a": [1.5]})
        assert PolarsDataFrame()._output_schema(df) == (("a", "Float64"), ("b", "Int64"), ("c", "String"))
        assert call_count == 1


class TestBareComputeFrameworkDictInterchangeOutputSchema(DictInterchangeOutputSchemaTestMixin):
    """Shared mixin plus the dict-interchange edge cases, covered once here since that path is
    framework-agnostic (pandas/pyarrow/duckdb each cover the shared smoke test in their own file)."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return ComputeFramework()

    def test_empty_dict_yields_none(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({}) is None

    def test_all_none_column_yields_none_dtype(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({"a": [None, None]}) == (("a", None),)

    def test_scalar_column_value_yields_none_dtype(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({"a": 1}) == (("a", None),)

    def test_non_string_keys_are_stringified_and_sorted_by_string_form(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({1: [1], "a": [2]}) == (("1", "int"), ("a", "int"))


class _NamesOnlyFramework(ComputeFramework):
    """Overrides only _extract_column_names; dtype extraction stays the base no-op (returns None)."""

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.keys())


class TestDefaultDtypeFrameworkOutputSchema:
    def test_dtype_defaults_to_none_when_unoverridden(self) -> None:
        fw = _NamesOnlyFramework()
        data = types.MappingProxyType({"b": [1], "a": [2]})
        assert fw._output_schema(data) == (("a", None), ("b", None))


class TestBaseComputeFrameworkOutputSchemaRaises:
    """Non-dict data reaches the base _extract_column_names, which raises, so _output_schema raises too."""

    def test_raises_not_implemented_error(self) -> None:
        with pytest.raises(NotImplementedError):
            ComputeFramework()._output_schema([1, 2])


class _RaisingColumnADtypeFramework(ComputeFramework):
    """Names come back fixed as {"a", "b"}; dtype extraction raises for "a" but works for "b"."""

    def _extract_column_names(self, data: Any) -> set[str]:
        return {"a", "b"}

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        if column_name == "a":
            raise RuntimeError("cannot read dtype for column 'a'")
        return "int"


class TestBaseComputeFrameworkGenericLoopDegradesColumnToNone:
    """Coverage restoration: exercises ComputeFramework._output_schema's generic per-column
    safe_field loop directly, independent of PythonDictFramework's own single-pass override.
    This loop still runs for every framework that does not override _output_schema."""

    def test_raising_extract_column_dtype_degrades_only_that_column(self) -> None:
        fw = _RaisingColumnADtypeFramework()
        # A non-dict-shaped sentinel so this hits the generic loop, not the dict shortcut.
        assert fw._output_schema(object()) == (("a", None), ("b", "int"))


class _ContextCapturingExtender(Extender):
    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


class _HookCapturingExtender(Extender):
    def __init__(self, hook: ExtenderHook, priority: int = 100) -> None:
        self._hook = hook
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {self._hook}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


def _build_feature_set() -> FeatureSet:
    return FeatureSet([Feature("my_feature")])


def _build_framework(extenders: set[Extender]) -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender=extenders)


class _OutputSchemaFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"b": ["x", "y"], "a": [1, 2]}


class TestOutputSchemaEndToEnd:
    """Regression: output_schema through the real hook wiring reflects sorted columns/dtypes; rows_out unaffected."""

    def test_output_schema_and_rows_out_both_populate(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = _build_framework({extender})

        cfw.run_calculate_feature(_OutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema == (("a", "int"), ("b", "str"))
        assert captured.rows_out == 2


class _BaseFrameworkOutputSchemaFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return [1, 2]


class TestBaseComputeFrameworkOutputSchemaEndToEnd:
    """Regression: output_schema on the base ComputeFramework degrades to None (_extract_column_names raises)."""

    def test_output_schema_degrades_to_none(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = ComputeFramework(
            mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender={extender}
        )

        cfw.run_calculate_feature(_BaseFrameworkOutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema is None
        assert captured.status == "success"


class _HostileColumnARow(dict[str, Any]):
    """Access to column 'a' always raises; 'b' stays readable. Poisons row access directly
    since single-pass _output_schema no longer calls _extract_column_dtype for this shape."""

    def get(self, key: Any, default: Any = None) -> Any:
        if key == "a":
            raise RuntimeError("corrupted row: cannot read column 'a'")
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        if key == "a":
            raise RuntimeError("corrupted row: cannot read column 'a'")
        return super().__getitem__(key)


class _RowWiseOutputSchemaFeatureGroup(FeatureGroup):
    """Root feature group returning the row-wise list[dict] shape PythonDictFramework accepts before transform."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return [_HostileColumnARow(b="x", a=1), _HostileColumnARow(b="y", a=2)]


class TestDtypeFailureDegradesColumnToNone:
    def test_dtype_raising_degrades_only_that_columns_dtype(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = _build_framework({extender})

        result = cfw.run_calculate_feature(_RowWiseOutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema == (("a", None), ("b", "str"))
        assert captured.status == "success"
        # dict.__eq__ does not invoke the overridden get/__getitem__, so this equality check is
        # unaffected by the poisoned column and confirms the schema-read failure did not corrupt data.
        assert result == [{"b": "x", "a": 1}, {"b": "y", "a": 2}]


class TestValidateHooksOutputSchema:
    def test_validate_output_feature_output_schema_reflects_native_data(self) -> None:
        """VALIDATE_OUTPUT_FEATURE reads self.data (the finalized native shape) for output_schema, mirroring rows_in."""
        feature_set = _build_feature_set()
        extender = _HookCapturingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        cfw = _build_framework({extender})
        cfw.data = {"col": [1, 2, 3]}
        cfw.set_column_names()

        cfw.run_validate_output_features(_OutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema == (("col", "int"),)

    def test_validate_input_feature_leaves_output_schema_none(self) -> None:
        feature_set = _build_feature_set()
        extender = _HookCapturingExtender(ExtenderHook.VALIDATE_INPUT_FEATURE)
        cfw = _build_framework({extender})
        cfw.data = {"col": [1, 2, 3]}

        cfw.run_validate_input_features(_OutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema is None


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyOutputSchemaStaysLazy:
    def test_output_schema_does_not_collect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_if_called(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("LazyFrame.collect must not be called by _output_schema")

        monkeypatch.setattr(pl.LazyFrame, "collect", _raise_if_called)

        lazy_frame = pl.LazyFrame({"b": ["x"], "a": [1]})

        assert PolarsLazyDataFrame()._output_schema(lazy_frame) == (("a", "Int64"), ("b", "String"))

    def test_output_schema_calls_collect_schema_at_most_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Perf regression: collect_schema() must run once total, not once per column."""
        call_count = 0
        original_collect_schema = pl.LazyFrame.collect_schema

        def _counting_collect_schema(self: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original_collect_schema(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect_schema", _counting_collect_schema)

        lazy_frame = pl.LazyFrame({"c": ["x"], "b": [1], "a": [1.5]})

        assert PolarsLazyDataFrame()._output_schema(lazy_frame) == (("a", "Float64"), ("b", "Int64"), ("c", "String"))
        assert call_count == 1


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyDictInterchangeOutputSchema(DictInterchangeOutputSchemaTestMixin):
    """Test PolarsLazyDataFrame._output_schema on the dict interchange shape using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PolarsLazyDataFrame()


class _IndexForbiddenList(list[str]):
    """Stand-in for data.columns whose .index() raises, since the builtin list type itself
    cannot be monkeypatched."""

    def index(self, *args: Any, **kwargs: Any) -> int:
        raise AssertionError("list.index must not be called by _output_schema")


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB/PyArrow is not installed.")
class TestDuckDBOutputSchemaStaysLazy:
    """DuckDBFramework._output_schema must never trigger the relation's expensive count_star() query."""

    def test_output_schema_does_not_call_dunder_len(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict({"b": ["x"], "a": [1]})
        relation = DuckdbRelation.from_arrow(conn, arrow_table)

        def _raise_if_called(self: Any) -> int:
            raise AssertionError("DuckdbRelation.__len__ must not be called by _output_schema")

        monkeypatch.setattr(DuckdbRelation, "__len__", _raise_if_called)

        assert DuckDBFramework()._output_schema(relation) == (("a", "BIGINT"), ("b", "VARCHAR"))

    def test_output_schema_does_not_call_list_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Perf regression: data.columns.index(name) is O(columns) per call, quadratic overall."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict({"c": ["x"], "b": [1], "a": [1.5]})
        relation = DuckdbRelation.from_arrow(conn, arrow_table)

        original_get_columns = cast(property, DuckdbRelation.__dict__["columns"]).fget
        assert original_get_columns is not None

        def _index_forbidden_columns(self: Any) -> Any:
            return _IndexForbiddenList(original_get_columns(self))

        monkeypatch.setattr(DuckdbRelation, "columns", property(_index_forbidden_columns))

        assert DuckDBFramework()._output_schema(relation) == (("a", "DOUBLE"), ("b", "BIGINT"), ("c", "VARCHAR"))

    def test_output_schema_reads_columns_and_types_properties_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stronger perf regression than test_output_schema_does_not_call_list_index: counts total
        reads of columns/types instead of only forbidding .index(), catching any once-per-column read."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict({"c": ["x"], "b": [1], "a": [1.5]})
        relation = DuckdbRelation.from_arrow(conn, arrow_table)

        original_get_columns = cast(property, DuckdbRelation.__dict__["columns"]).fget
        original_get_types = cast(property, DuckdbRelation.__dict__["types"]).fget
        assert original_get_columns is not None
        assert original_get_types is not None

        columns_read_count = 0
        types_read_count = 0

        def _counting_columns(self: Any) -> Any:
            nonlocal columns_read_count
            columns_read_count += 1
            return original_get_columns(self)

        def _counting_types(self: Any) -> Any:
            nonlocal types_read_count
            types_read_count += 1
            return original_get_types(self)

        monkeypatch.setattr(DuckdbRelation, "columns", property(_counting_columns))
        monkeypatch.setattr(DuckdbRelation, "types", property(_counting_types))

        assert DuckDBFramework()._output_schema(relation) == (("a", "DOUBLE"), ("b", "BIGINT"), ("c", "VARCHAR"))
        assert columns_read_count == 1
        assert types_read_count == 1

    def test_validate_output_feature_hook_does_not_call_dunder_len(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression at the hook-dispatch level: VALIDATE_OUTPUT_FEATURE must read output_schema
        without materializing the relation, not just when _output_schema is unit-tested directly."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict({"b": ["x"], "a": [1]})
        relation = DuckdbRelation.from_arrow(conn, arrow_table)

        def _raise_if_called(self: Any) -> int:
            raise AssertionError("DuckdbRelation.__len__ must not be called by _output_schema")

        monkeypatch.setattr(DuckdbRelation, "__len__", _raise_if_called)

        feature_set = _build_feature_set()
        extender = _HookCapturingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        cfw = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender={extender})
        cfw.data = relation
        cfw.set_column_names()

        cfw.run_validate_output_features(_OutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema == (("a", "BIGINT"), ("b", "VARCHAR"))


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB/PyArrow is not installed.")
class TestDuckDBOutputSchemaDuplicateColumns:
    """Duplicate column names (e.g. an un-aliased join) collapse to one entry, first occurrence wins."""

    def test_duplicate_column_names_collapse_to_one_entry(self) -> None:
        conn = duckdb.connect()
        relation = DuckdbRelation(conn, conn.sql("select 1 as a, 2 as b, 3 as a"))

        result = DuckDBFramework()._output_schema(relation)

        assert result is not None
        a_entries = [pair for pair in result if pair[0] == "a"]
        assert len(a_entries) == 1


class _PandasOutputSchemaFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"a": [1]}


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasOutputSchema:
    def test_sorted_columns_with_pandas_dtypes(self) -> None:
        df = pd.DataFrame({"b": [1.5], "a": [1]})
        assert PandasDataFrame()._output_schema(df) == (("a", "int64"), ("b", "float64"))

    def test_non_string_labels_are_stringified_and_sorted_by_string_form(self) -> None:
        df = pd.DataFrame({0: [1], "b": [1.5]})
        assert PandasDataFrame()._output_schema(df) == (("0", "int64"), ("b", "float64"))

    def test_dict_result_reports_python_type_names(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender={extender})

        cfw.run_calculate_feature(_PandasOutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema == (("a", "int"),)
        assert captured.status == "success"


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestOutputSchemaCalculateVsValidateOutput:
    """FEATURE_GROUP_CALCULATE_FEATURE sees the raw FG return value's schema,
    VALIDATE_OUTPUT_FEATURE sees the finalized native-shape schema after transform."""

    def test_calculate_and_validate_output_hooks_see_different_schemas(self) -> None:
        feature_set = _build_feature_set()
        calculate_extender = _HookCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        validate_output_extender = _HookCapturingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        cfw = PandasDataFrame(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            function_extender={calculate_extender, validate_output_extender},
        )

        cfw.run_calculation(_PandasOutputSchemaFeatureGroup, feature_set, location=None)

        calculate_captured = calculate_extender.captured
        assert calculate_captured is not None
        assert calculate_captured.output_schema == (("a", "int"),)

        validate_output_captured = validate_output_extender.captured
        assert validate_output_captured is not None
        assert validate_output_captured.output_schema == (("a", "int64"),)


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteOutputSchemaStaysLazy:
    """SqliteFramework._output_schema reads propagated hints or PRAGMA affinity and must never scan or count rows."""

    @staticmethod
    def _forbid_row_scan(monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_len(self: Any) -> int:
            raise AssertionError("SqliteRelation.__len__ must not be called by _output_schema")

        def _raise_infer(self: Any, type_hints: Any) -> Any:
            raise AssertionError(
                "SqliteRelation._types_with_inferred_unknown_hints must not be called by _output_schema"
            )

        monkeypatch.setattr(SqliteRelation, "__len__", _raise_len)
        monkeypatch.setattr(SqliteRelation, "_types_with_inferred_unknown_hints", _raise_infer)

    def test_fully_resolved_hints_report_every_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = sqlite3.connect(":memory:")
        relation = SqliteRelation.from_arrow(conn, pa.table({"b": ["x"], "a": [1]}))
        self._forbid_row_scan(monkeypatch)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        assert SqliteFramework()._output_schema(relation) == (("a", "int64"), ("b", "string"))
        assert not [s for s in statements if s.startswith("SELECT *") and "LIMIT 0" not in s]

    def test_unresolved_hint_reports_none_without_scanning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = sqlite3.connect(":memory:")
        relation = SqliteRelation.from_arrow(conn, pa.table({"b": ["x"], "a": [1]}))
        derived = relation.select(_raw_sql="*, a * 2 AS c")
        self._forbid_row_scan(monkeypatch)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        assert SqliteFramework()._output_schema(derived) == (("a", "int64"), ("b", "string"), ("c", None))
        assert not [s for s in statements if s.startswith("SELECT *") and "LIMIT 0" not in s]

    def test_relation_without_cached_hints_reports_affinity_types_without_scanning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = sqlite3.connect(":memory:")
        relation = SqliteRelation.from_arrow(conn, pa.table({"b": ["x"], "a": [1]}))
        bare = SqliteRelation(conn, relation.table_name)
        assert bare.type_hints is None
        self._forbid_row_scan(monkeypatch)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        assert SqliteFramework()._output_schema(bare) == (("a", "int64"), ("b", "string"))
        assert not [s for s in statements if s.startswith("SELECT *") and "LIMIT 0" not in s]
