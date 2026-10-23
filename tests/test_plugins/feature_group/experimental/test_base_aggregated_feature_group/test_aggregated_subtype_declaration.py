"""Pinning tests for the SubtypeDeclaration dogfooding of AggregatedFeatureGroup (issue #639, cycle 2)."""

import inspect
import math
from abc import abstractmethod
from collections.abc import Callable
import statistics
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.provider import SubtypeDeclaration
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.polars_lazy import PolarsLazyAggregation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.feature_group.experimental.reference_contract import same


AGGREGATION_UNIVERSE = frozenset(AggregatedFeatureGroup.AGGREGATION_TYPES)


class AbstractOperationBase(OperationFeatureGroup):
    """A test-local abstract family base."""

    @classmethod
    @abstractmethod
    def _family_hook(cls) -> str:
        """Abstract hook that makes this base abstract."""


class TestAggregatedSubtypeDeclaration:
    """AggregatedFeatureGroup declares its aggregation types via SUBTYPES."""

    def test_declaration_key_is_aggregation_type(self) -> None:
        declaration = AggregatedFeatureGroup.SUBTYPES
        assert declaration is not None
        assert isinstance(declaration, SubtypeDeclaration)
        assert declaration.key == AggregatedFeatureGroup.AGGREGATION_TYPE
        assert declaration.key == "aggregation_type"

    def test_universe_matches_declared_aggregation_types(self) -> None:
        assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE


class TestAggregatedResolveSubtype:
    """resolve_subtype resolves aggregation types from the name and the options path."""

    def test_resolves_from_feature_name(self) -> None:
        assert AggregatedFeatureGroup.resolve_subtype("sales__sum_aggr", Options()) == "sum"

    def test_resolves_from_a_chained_feature_name(self) -> None:
        assert AggregatedFeatureGroup.resolve_subtype("s__mean_imputed__sum_aggr", Options()) == "sum"

    def test_resolves_from_options_path(self) -> None:
        options = Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "max"})
        assert AggregatedFeatureGroup.resolve_subtype("aggr_config_placeholder", options) == "max"

    def test_unrelated_name_resolves_to_none(self) -> None:
        # Anchor on the declaration so this test is red until SUBTYPES lands.
        assert AggregatedFeatureGroup.SUBTYPES is not None
        assert AggregatedFeatureGroup.resolve_subtype("sales_total", Options()) is None


class TestAggregatedSupportedSubtypes:
    """supported_subtypes defaults to the full universe on the concrete group."""

    def test_default_is_full_universe_for_any_framework(self) -> None:
        assert AggregatedFeatureGroup.supported_subtypes(PandasDataFrame) == AGGREGATION_UNIVERSE
        assert AggregatedFeatureGroup.supported_subtypes(PythonDictFramework) == AGGREGATION_UNIVERSE


class TestAggregatedBehaviorNeutral:
    """The declaration is behavior-neutral: every aggregation type stays supported on PandasDataFrame."""

    def test_every_aggregation_type_stays_supported_on_pandas(self) -> None:
        # Anchor on the declared universe so this test is red until SUBTYPES lands.
        assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE
        for aggregation_type in AggregatedFeatureGroup.AGGREGATION_TYPES:
            supported = AggregatedFeatureGroup.supports_compute_framework(
                f"sales__{aggregation_type}_aggr", Options(), PandasDataFrame
            )
            assert supported is True, f"'{aggregation_type}' must stay supported on PandasDataFrame"


class TestAggregatedSubtypeSupportMatrix:
    """The group is concrete and its matrix comes from the implementations, one entry per served framework."""

    def test_group_declares_the_universe(self) -> None:
        assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE

    def test_abstract_family_base_has_an_empty_matrix(self) -> None:
        assert inspect.isabstract(AbstractOperationBase)
        assert AbstractOperationBase.subtype_support_matrix() == {}

    def test_matrix_gives_every_served_framework_the_full_universe(self) -> None:
        matrix = AggregatedFeatureGroup.subtype_support_matrix()
        assert matrix, "the concrete group must have a non-empty support matrix"
        declared = {cfw.get_class_name() for cfw in AggregatedFeatureGroup.compute_framework_definition()}
        assert set(matrix) == declared
        assert {PandasDataFrame.get_class_name(), PyArrowTable.get_class_name()} <= set(matrix)
        unserved = {f.get_class_name() for f in (PythonDictFramework, DuckDBFramework, SqliteFramework)}
        assert not unserved & set(matrix)
        assert all(subtypes == AGGREGATION_UNIVERSE for subtypes in matrix.values())


_NAN = float("nan")

# Single column: four non-null values (one NaN) and one null. Multi column: every row keeps two or more
# non-null cells so std and var are defined; rows mix clean cells, NaN and null.
_SINGLE_COLUMN: dict[str, list[float | None]] = {"a": [1.0, 2.0, _NAN, None, 3.0]}
_MULTI_COLUMNS: dict[str, list[float | None]] = {
    "a": [1.0, _NAN, None, 8.0, None],
    "b": [2.0, 4.0, 6.0, _NAN, _NAN],
    "c": [3.0, 5.0, 7.0, None, 9.0],
}

_DIVERGENT_DECLARATIONS: dict[type[Implementation[Any]], dict[str, str]] = {
    PandasAggregation: {
        "sum": "skips NaN in sum",
        "avg": "skips NaN in avg",
        "mean": "skips NaN in mean",
        "std": "skips NaN in std",
        "var": "skips NaN in var",
        "count": "does not count NaN",
    },
    PolarsLazyAggregation: {
        "median": "sorts NaN as the largest value in median",
        "max": "propagates NaN in multi-column max",
    },
    PyArrowAggregation: {},
}


def _reference(aggregation_type: str, cells: list[float | None]) -> float | int:
    """The reference contract: null is skipped, NaN propagates in sum/avg/mean/std/var, count counts NaN."""
    present = [cell for cell in cells if cell is not None]
    clean = [cell for cell in present if not math.isnan(cell)]
    has_nan = len(clean) != len(present)
    if aggregation_type == "count":
        return len(present)
    if aggregation_type in ("min", "max", "median"):
        reducers: dict[str, Callable[[list[float]], float]] = {"min": min, "max": max, "median": statistics.median}
        return reducers[aggregation_type](clean)
    if has_nan:
        return _NAN
    if aggregation_type == "sum":
        return sum(clean)
    if aggregation_type in ("avg", "mean"):
        return statistics.fmean(clean)
    return statistics.stdev(clean) if aggregation_type == "std" else statistics.variance(clean)


def _reference_result(aggregation_type: str, columns: dict[str, list[float | None]]) -> list[float | int]:
    names = list(columns)
    if len(names) == 1:
        return [_reference(aggregation_type, columns[names[0]])]
    row_count = len(columns[names[0]])
    return [_reference(aggregation_type, [columns[name][row] for name in names]) for row in range(row_count)]


def _pandas_result(aggregation_type: str, columns: dict[str, list[float | None]]) -> list[Any]:
    # A pandas float column cannot tell null from NaN, so both cells become NaN.
    frame = pd.DataFrame({k: [np.nan if v is None else v for v in vs] for k, vs in columns.items()})
    result = PandasAggregation._perform_aggregation(frame, aggregation_type, list(columns))
    return list(result) if hasattr(result, "__iter__") else [result]


def _pyarrow_result(aggregation_type: str, columns: dict[str, list[float | None]]) -> list[Any]:
    table = pa.table({k: pa.array(vs, type=pa.float64()) for k, vs in columns.items()})
    result = PyArrowAggregation._perform_aggregation(table, aggregation_type, list(columns))
    return [result.as_py()] if isinstance(result, pa.Scalar) else list(result.tolist())


def _polars_result(aggregation_type: str, columns: dict[str, list[float | None]]) -> list[Any]:
    pl = pytest.importorskip("polars")
    frame = pl.LazyFrame({k: pl.Series(k, vs, dtype=pl.Float64) for k, vs in columns.items()})
    expression = PolarsLazyAggregation._perform_aggregation(frame, aggregation_type, list(columns))
    return list(frame.select(expression.alias("result")).collect()["result"].to_list())


_RESULT_OF: dict[type[Implementation[Any]], Callable[[str, dict[str, list[float | None]]], list[Any]]] = {
    PandasAggregation: _pandas_result,
    PyArrowAggregation: _pyarrow_result,
    PolarsLazyAggregation: _polars_result,
}


class TestImplementationsConformToTheReferenceContract:
    """An implementation differs from the reference exactly where it declares a divergence."""

    @pytest.mark.parametrize("aggregation_type", sorted(AGGREGATION_UNIVERSE))
    @pytest.mark.parametrize("implementation", list(_RESULT_OF), ids=lambda impl: impl.__name__)
    def test_result_differs_from_the_reference_exactly_when_declared(
        self, implementation: type[Implementation[Any]], aggregation_type: str
    ) -> None:
        assert dict(implementation.divergent) == _DIVERGENT_DECLARATIONS[implementation]
        shapes = {"single": _SINGLE_COLUMN, "multi": _MULTI_COLUMNS}
        differing: list[str] = []
        for shape, columns in shapes.items():
            # Polars rejects a multi-column median at run time; unsupported_multi_column_aggregations pins that.
            if implementation is PolarsLazyAggregation and shape == "multi" and aggregation_type == "median":
                continue
            actual = _RESULT_OF[implementation](aggregation_type, columns)
            expected = _reference_result(aggregation_type, columns)
            assert len(actual) == len(expected), f"{shape}: {actual} vs {expected}"
            if not all(same(a, e) for a, e in zip(actual, expected)):
                differing.append(f"{shape}: {actual} != reference {expected}")
        assert bool(differing) == (aggregation_type in implementation.divergent), differing
