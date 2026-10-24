"""Pins each operation family's group/implementation contract and that plugins import through mloda.provider."""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.code_closure import closure_parts
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.provider import DefaultOptionKeys, FeatureSet
from mloda.user import Feature, Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow import pyarrow_type_semantics
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.polars_lazy import PolarsLazyAggregation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregation
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClustering
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecasting
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueImputation
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow import PyArrowMissingValueImputation
from mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict import (
    PythonDictMissingValueImputation,
)
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import PandasDimensionalityReduction
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistance
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentrality
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncoding
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import PandasSklearnPipeline
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScaling
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaning
from mloda_plugins.feature_group.experimental.text_cleaning.python_dict import PythonDictTextCleaning
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindow

from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import (
    PandasZeroRowAdapter,
    PolarsLazyZeroRowAdapter,
    PyArrowZeroRowAdapter,
    PythonDictZeroRowAdapter,
    ZeroRowResultTypeTestMixin,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group" / "experimental"
assert SCAN_ROOT.exists(), f"experimental plugin root not found: {SCAN_ROOT}"

# The core package a plugin module must never import from: the facade is the authoring surface.
CORE_CHAINER_PACKAGE = "mloda.core.abstract_plugins.components.feature_chainer"

# Lower bounds only, so a broken glob or a wrong root cannot make the widened sweeps pass vacuously.
MIN_EXPERIMENTAL_MODULES = 40


def _module_name_of(path: Path) -> str:
    """Dotted module name of a plugin file, derived from the scan root so no path is hard-coded."""
    relative = path.relative_to(SCAN_ROOT.parents[2])
    return ".".join(relative.with_suffix("").parts)


def _imported_modules(path: Path) -> set[str]:
    """Every module name the import statements of one file reference."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _family_declarations() -> dict[Path, type[Any]]:
    """The family base of every directory under the scan root that owns a base.py, keyed by directory."""
    declarations: dict[Path, type[Any]] = {}
    for path in sorted(SCAN_ROOT.rglob("base.py")):
        families = _mixin_subclasses_defined_in(_module_name_of(path))
        assert len(families) == 1, f"{path.relative_to(SCAN_ROOT)} defines {len(families)} mixin classes, expected 1"
        declarations[path.parent] = families[0]
    return declarations


def _mixin_subclasses_defined_in(module_name: str) -> list[type[Any]]:
    """The FeatureChainParserMixin subclasses a module defines itself, ignoring imported ones."""
    module = importlib.import_module(module_name)
    return [
        member
        for _name, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, FeatureChainParserMixin) and member.__module__ == module_name
    ]


def test_experimental_plugins_import_the_chainer_through_the_facade() -> None:
    """Plugin modules import the chainer API through mloda.provider, never the core package."""
    offenders: list[str] = []
    visited = 0
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        visited += 1
        deep = sorted(
            module
            for module in _imported_modules(path)
            if module == CORE_CHAINER_PACKAGE or module.startswith(f"{CORE_CHAINER_PACKAGE}.")
        )
        if deep:
            offenders.append(f"{path.relative_to(SCAN_ROOT)} imports {deep}")
    assert offenders == [], f"plugin modules must import from mloda.provider, not the core path: {offenders}"
    assert visited >= MIN_EXPERIMENTAL_MODULES, (
        f"sweep visited only {visited} modules under {SCAN_ROOT}: the sweep root is wrong or the glob broke"
    )


def _perform(name: str) -> Any:
    return lambda impl: getattr(impl, name)


def _kwargs(**extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "add_column": _perform("_add_result_to_data"),
        "columns": lambda impl: impl.compute_framework.extract_column_names,
    }
    spec.update(extra)
    return spec


_ABSENT_OLD_HOOKS = ("_get_available_columns", "_check_source_features_exist")

# (group, implementations, loop method name, feature name, expected keyword callables per implementation)
OPERATION_FAMILIES: dict[str, tuple[type[Any], list[type[Any]], str, str, list[dict[str, Any]]]] = {
    "aggregation": (
        AggregatedFeatureGroup,
        [PandasAggregation, PyArrowAggregation, PolarsLazyAggregation],
        "aggregate",
        "sales__sum_aggr",
        [_kwargs(perform=_perform("_perform_aggregation"))] * 3,
    ),
    "time_window": (
        TimeWindowFeatureGroup,
        [PandasTimeWindow, PyArrowTimeWindow],
        "window",
        "temperature__avg_3_day_window",
        [
            _kwargs(
                perform=_perform("_perform_window_operation"),
                column_semantics=lambda _impl: pandas_type_semantics.column_semantics,
            ),
            _kwargs(
                perform=_perform("_perform_window_operation"),
                column_semantics=lambda _impl: pyarrow_type_semantics.column_semantics,
            ),
        ],
    ),
    "clustering": (
        ClusteringFeatureGroup,
        [PandasClustering],
        "cluster",
        "a__cluster_kmeans_2",
        [
            _kwargs(
                perform=_perform("_perform_clustering"),
                perform_with_probabilities=_perform("_perform_clustering_with_probabilities"),
            )
        ],
    ),
    "forecasting": (
        ForecastingFeatureGroup,
        [PandasForecasting],
        "forecast",
        "sales__linear_forecast_7day",
        [
            _kwargs(
                perform=_perform("_perform_forecasting"),
                perform_with_confidence=_perform("_perform_forecasting_with_confidence"),
                column_semantics=lambda _impl: pandas_type_semantics.column_semantics,
            )
        ],
    ),
    "dimensionality_reduction": (
        DimensionalityReductionFeatureGroup,
        [PandasDimensionalityReduction],
        "reduce",
        "a,b,c__pca_2d",
        [_kwargs(perform=_perform("_perform_reduction"))],
    ),
    "node_centrality": (
        NodeCentralityFeatureGroup,
        [PandasNodeCentrality],
        "centrality",
        "user__degree_centrality",
        [_kwargs(perform=_perform("_calculate_centrality"))],
    ),
    "missing_value": (
        MissingValueFeatureGroup,
        [PandasMissingValueImputation, PyArrowMissingValueImputation, PythonDictMissingValueImputation],
        "impute",
        "income__mean_imputed",
        [_kwargs(perform=_perform("_perform_imputation"))] * 3,
    ),
    "text_cleaning": (
        TextCleaningFeatureGroup,
        [PandasTextCleaning, PythonDictTextCleaning],
        "clean",
        "text__cleaned_text",
        [_kwargs(source_text=_perform("_get_source_text"), apply=_perform("_apply_operation"))] * 2,
    ),
    "geo_distance": (
        GeoDistanceFeatureGroup,
        [PandasGeoDistance],
        "measure",
        "a&b__haversine_distance",
        [_kwargs(distance=_perform("_calculate_distance"))],
    ),
    "scaling": (
        ScalingFeatureGroup,
        [PandasScaling],
        "scale",
        "income__standard_scaled",
        [_kwargs(training_data=_perform("_extract_training_data"), apply=_perform("_apply_scaler"))],
    ),
    "encoding": (
        EncodingFeatureGroup,
        [PandasEncoding],
        "encode",
        "category__onehot_encoded",
        [_kwargs(training_data=_perform("_extract_training_data"), apply=_perform("_apply_encoder"))],
    ),
    "sklearn_pipeline": (
        SklearnPipelineFeatureGroup,
        [PandasSklearnPipeline],
        "run_pipeline",
        "a,b__sklearn_pipeline_scaling",
        [_kwargs(training_data=_perform("_extract_training_data"), apply=_perform("_apply_pipeline"))],
    ),
}

OPERATION_GROUPS = [(family[0]) for family in OPERATION_FAMILIES.values()]
GROUP_IDS = list(OPERATION_FAMILIES)
IMPLEMENTATION_CASES = [
    pytest.param(name, impl, id=f"{name}-{impl.__name__}")
    for name, family in OPERATION_FAMILIES.items()
    for impl in family[1]
]
DELEGATION_CASES = [
    pytest.param(name, impl, family[4][index], id=f"{name}-{impl.__name__}")
    for name, family in OPERATION_FAMILIES.items()
    for index, impl in enumerate(family[1])
]


@pytest.mark.parametrize("group", OPERATION_GROUPS, ids=GROUP_IDS)
def test_operation_group_is_concrete(group: type[Any]) -> None:
    assert not inspect.isabstract(group)


def test_operation_family_bases_under_the_scan_root_are_exactly_the_table_groups() -> None:
    discovered = {family for family in _family_declarations().values() if issubclass(family, OperationFeatureGroup)}
    assert discovered == set(OPERATION_GROUPS)


@pytest.mark.parametrize(("name", "impl"), IMPLEMENTATION_CASES)
def test_divergent_keys_are_subtypes_of_the_group(name: str, impl: type[Any]) -> None:
    group = OPERATION_FAMILIES[name][0]
    assert set(impl.divergent) <= set(group.subtype_universe())


@pytest.mark.parametrize(("name", "impl"), IMPLEMENTATION_CASES)
def test_implementation_serves_its_group(name: str, impl: type[Any]) -> None:
    group = OPERATION_FAMILIES[name][0]
    assert impl.__bases__ == (Implementation,)
    assert impl.feature_group is group
    assert not inspect.isabstract(impl)
    for hook in _ABSENT_OLD_HOOKS:
        assert not hasattr(impl, hook)
    assert impl.compute_framework in group.compute_framework_definition()
    group_part = f"{group.__module__}:{group.__name__}:"
    assert any(part.startswith(group_part) for part in closure_parts(impl, include_dependencies=False))


@pytest.mark.parametrize(("name", "impl", "expected"), DELEGATION_CASES)
def test_calculate_delegates_to_the_group_it_is_handed(name: str, impl: type[Any], expected: dict[str, Any]) -> None:
    _group, _impls, loop, feature_name, _specs = OPERATION_FAMILIES[name]
    group = Mock()
    data = {"a": [1]}
    feature_set = FeatureSet()
    feature_set.add(Feature(feature_name))

    result = impl.calculate(group, data, feature_set, None)

    getattr(group, loop).assert_called_once_with(
        data, feature_set, **{key: resolve(impl) for key, resolve in expected.items()}
    )
    assert result is getattr(group, loop).return_value


_ROWS = 12
_DAYS = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(_ROWS)]
_NUMBERS = [float(i + 1) for i in range(_ROWS)]
_GEO_POINTS = [(50.0 + i * 0.1, 8.0 + i * 0.1) for i in range(_ROWS)]
_NUMERIC_COLUMNS: dict[str, list[Any]] = {
    "a": _NUMBERS,
    "b": [float((i * 7) % 5) for i in range(_ROWS)],
    "c": [float((i * 3) % 4) for i in range(_ROWS)],
}


def _numeric_table() -> pa.Table:
    return pa.table(_NUMERIC_COLUMNS)


def _time_table() -> pa.Table:
    return pa.table({"temperature": _NUMBERS, "sales": _NUMBERS, "reference_time": _DAYS, "ts": _DAYS})


def _node_table() -> pa.Table:
    users = [f"u{i % 4}" for i in range(_ROWS)]
    targets = [f"u{(i + 1) % 4}" for i in range(_ROWS)]
    return pa.table({"user": users, "source": users, "target": targets, "w": _NUMBERS})


def _missing_value_table() -> pa.Table:
    income: list[Any] = [None if i % 4 == 0 else v for i, v in enumerate(_NUMBERS)]
    return pa.table({"income": income, "region": ["n" if i % 2 else "s" for i in range(_ROWS)]})


# One frame builder per family (a superset of columns) and the option variants that change the declared inputs.
LOOP_INPUT_CASES: dict[str, tuple[Callable[[], pa.Table], list[Options]]] = {
    "aggregation": (lambda: pa.table({"sales": _NUMBERS}), [Options()]),
    "time_window": (
        _time_table,
        [Options(), Options({DefaultOptionKeys.reference_time: "ts"})],
    ),
    "clustering": (_numeric_table, [Options()]),
    "forecasting": (
        _time_table,
        [Options(), Options({DefaultOptionKeys.reference_time: "ts"})],
    ),
    "dimensionality_reduction": (_numeric_table, [Options()]),
    "node_centrality": (_node_table, [Options(), Options({"weight_column": "w"})]),
    "missing_value": (_missing_value_table, [Options(), Options({"group_by_features": ["region"]})]),
    "text_cleaning": (
        lambda: pa.table({"text": ["Hello, The World!"] * _ROWS}),
        [Options({"cleaning_operations": ("normalize",)})],
    ),
    "geo_distance": (lambda: pa.table({"a": _GEO_POINTS, "b": list(reversed(_GEO_POINTS))}), [Options()]),
    "scaling": (lambda: pa.table({"income": _NUMBERS}), [Options()]),
    "encoding": (lambda: pa.table({"category": ["x", "y", "z"] * (_ROWS // 3)}), [Options()]),
    "sklearn_pipeline": (_numeric_table, [Options()]),
}

_ADAPTERS: dict[type[ComputeFramework], ZeroRowResultTypeTestMixin] = {
    PandasDataFrame: PandasZeroRowAdapter(),
    PyArrowTable: PyArrowZeroRowAdapter(),
    PolarsLazyDataFrame: PolarsLazyZeroRowAdapter(),
    PythonDictFramework: PythonDictZeroRowAdapter(),
}

LOOP_INPUT_PARAMS = [
    pytest.param(name, impl, options, id=f"{name}-{impl.__name__}-{index}")
    for name, family in OPERATION_FAMILIES.items()
    for impl in family[1]
    for index, options in enumerate(LOOP_INPUT_CASES.get(name, (None, []))[1])
]


def test_loop_input_cases_cover_every_operation_family() -> None:
    assert set(LOOP_INPUT_CASES) == set(OPERATION_FAMILIES)
    assert all(variants for _builder, variants in LOOP_INPUT_CASES.values())


@pytest.mark.parametrize(("name", "impl", "options"), LOOP_INPUT_PARAMS)
def test_loop_runs_on_declared_inputs_only(name: str, impl: type[Any], options: Options) -> None:
    group = OPERATION_FAMILIES[name][0]
    feature = Feature(OPERATION_FAMILIES[name][3], options)
    declared = group.declared_input_features(feature.options, feature.name)
    assert declared is not None
    table = LOOP_INPUT_CASES[name][0]().select(sorted(str(f.name) for f in declared))
    data = _ADAPTERS[impl.compute_framework].from_arrow(table)
    feature_set = FeatureSet()
    feature_set.add(feature)
    feature_set.materialize_option_defaults(group)

    result = impl.calculate(group, data, feature_set, None)

    columns = {str(c) for c in impl.compute_framework.extract_column_names(result)}
    output = str(feature.name)
    assert any(c == output or c.startswith(f"{output}~") for c in columns), columns
