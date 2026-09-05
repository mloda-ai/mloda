"""Tests for ComputeFramework._output_schema: sorted (column, dtype) pairs, degradation to None
on failure, and that lazy frames (polars/duckdb) are never materialized to build the schema.
"""

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

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class TestPythonDictOutputSchema:
    """PythonDictFramework._output_schema pairs sorted column names with a best-effort dtype."""

    def test_sorted_columns_with_dtypes(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema({"b": ["x", "y"], "a": [1, 2]}) == (("a", "int"), ("b", "str"))

    def test_all_none_column_yields_none_dtype(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema({"a": [None, None]}) == (("a", None),)

    def test_empty_dict_yields_empty_schema(self) -> None:
        fw = PythonDictFramework()
        assert fw._output_schema({}) == ()


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowOutputSchema:
    """PyArrowTable._output_schema reads the arrow schema's native type names."""

    def test_sorted_columns_with_arrow_dtypes(self) -> None:
        table = pa.table({"b": ["x"], "a": [1]})
        assert PyArrowTable()._output_schema(table) == (("a", "int64"), ("b", "string"))


class _NamesOnlyFramework(ComputeFramework):
    """Overrides only _extract_column_names; dtype extraction stays the base no-op (returns None)."""

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.keys())


class TestDefaultDtypeFrameworkOutputSchema:
    """A framework overriding only _extract_column_names gets None dtypes from the base class."""

    def test_dtype_defaults_to_none_when_unoverridden(self) -> None:
        fw = _NamesOnlyFramework()
        assert fw._output_schema({"b": [1], "a": [2]}) == (("a", None), ("b", None))


class TestBaseComputeFrameworkOutputSchemaRaises:
    """The base ComputeFramework's _extract_column_names raises, so _output_schema raises too."""

    def test_raises_not_implemented_error(self) -> None:
        with pytest.raises(NotImplementedError):
            ComputeFramework()._output_schema({"a": [1]})


class _ContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

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
    """Calls func like a real extender for a caller-chosen hook, then reads HookContext.current()."""

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
    """Root feature group returning a dict with mixed dtypes."""

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
    """Root feature group returning a dict, run on the base (unoverridden) ComputeFramework."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"a": [1, 2], "b": [3, 4]}


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


class TestDtypeFailureDegradesToNone:
    """A dtype-extraction failure inside _output_schema must degrade output_schema to None, not break the call."""

    def test_dtype_raising_degrades_output_schema_to_none(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender()
        cfw = _RaisingDtypeFramework(
            mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender={extender}
        )

        result = cfw.run_calculate_feature(_OutputSchemaFeatureGroup, feature_set)

        assert result == {"b": ["x", "y"], "a": [1, 2]}
        captured = extender.captured
        assert captured is not None
        assert captured.output_schema is None
        assert captured.status == "success"


class TestValidateHooksLeaveOutputSchemaNone:
    """VALIDATE_INPUT_FEATURE and VALIDATE_OUTPUT_FEATURE hooks never populate output_schema."""

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
    """PolarsLazyDataFrame._output_schema must never call LazyFrame.collect()."""

    def test_output_schema_does_not_collect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_if_called(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("LazyFrame.collect must not be called by _output_schema")

        monkeypatch.setattr(pl.LazyFrame, "collect", _raise_if_called)

        lazy_frame = pl.LazyFrame({"b": ["x"], "a": [1]})

        assert PolarsLazyDataFrame()._output_schema(lazy_frame) == (("a", "Int64"), ("b", "String"))


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed.")
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
