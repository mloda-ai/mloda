"""Tests for ComputeFramework._output_schema: sorted (column, dtype) pairs, the dict interchange
shape on every framework, degradation to None on failure, and that lazy frames (polars/duckdb/sqlite)
are never materialized to build the schema.
"""

import sqlite3
import types
from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import FeatureGroup
from mloda.user import Feature, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

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

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
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


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowOutputSchema:
    def test_sorted_columns_with_arrow_dtypes(self) -> None:
        table = pa.table({"b": ["x"], "a": [1]})
        assert PyArrowTable()._output_schema(table) == (("a", "int64"), ("b", "string"))


class DictInterchangeOutputSchemaMixin:
    """Shared _output_schema tests for the dict interchange shape, identical on every framework since it
    bypasses the framework's own extraction and goes straight through the module-level dict reader."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        raise NotImplementedError

    def test_reads_dict_sorted_with_python_type_names(self, framework_instance: Any) -> None:
        assert framework_instance._output_schema({"b": ["x"], "a": [1], "c": [1.5]}) == (
            ("a", "int"),
            ("b", "str"),
            ("c", "float"),
        )


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDictInterchangeOutputSchema(DictInterchangeOutputSchemaMixin):
    @pytest.fixture
    def framework_instance(self) -> Any:
        return PandasDataFrame()


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowDictInterchangeOutputSchema(DictInterchangeOutputSchemaMixin):
    @pytest.fixture
    def framework_instance(self) -> Any:
        return PyArrowTable()


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB/PyArrow is not installed.")
class TestDuckDBDictInterchangeOutputSchema(DictInterchangeOutputSchemaMixin):
    @pytest.fixture
    def framework_instance(self) -> Any:
        return DuckDBFramework()


class TestBareComputeFrameworkDictInterchangeOutputSchema(DictInterchangeOutputSchemaMixin):
    """Also covers the dict-interchange edge cases once, since that path is framework-agnostic."""

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


class _RaisingDtypeFramework(PythonDictFramework):
    """PythonDictFramework whose _extract_column_dtype always raises, to exercise output_schema degradation."""

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        raise RuntimeError("dtype boom")


class _RowWiseOutputSchemaFeatureGroup(FeatureGroup):
    """Root feature group returning the row-wise list[dict] shape PythonDictFramework accepts before transform."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return [{"b": "x", "a": 1}, {"b": "y", "a": 2}]


class TestDtypeFailureDegradesColumnToNone:
    def test_dtype_raising_degrades_only_that_columns_dtype(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = _RaisingDtypeFramework(
            mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender={extender}
        )

        result = cfw.run_calculate_feature(_RowWiseOutputSchemaFeatureGroup, feature_set)

        assert result == [{"b": "x", "a": 1}, {"b": "y", "a": 2}]
        captured = extender.captured
        assert captured is not None
        assert captured.output_schema == (("a", None), ("b", None))
        assert captured.status == "success"


class TestValidateHooksLeaveOutputSchemaNone:
    def test_validate_output_feature_leaves_output_schema_none(self) -> None:
        feature_set = _build_feature_set()
        extender = _HookCapturingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        cfw = _build_framework({extender})
        cfw.data = {"col": [1, 2, 3]}
        cfw.set_column_names()

        cfw.run_validate_output_features(_OutputSchemaFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.output_schema is None

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
