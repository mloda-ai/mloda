"""Pins the three column-wise framework hooks to FeatureChainParserMixin and to every concrete family.

FeatureChainParserMixin declares the hooks as non-abstract classmethods whose body raises
NotImplementedError. That is deliberate: an @abstractmethod on a non-ABC class is silently
unenforced, so it would guarantee nothing. These tests are the enforcement instead. They check
both ends of the contract: the raising default exists on the mixin, and no shipped family
(pandas, PyArrow, Polars, python dict) falls through to it.
"""

from __future__ import annotations

import ast
from abc import ABC, ABCMeta
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.user.python_dict import row_count
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.polars_lazy import (
    PolarsLazyAggregatedFeatureGroup,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow import PyArrowMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict import (
    PythonDictMissingValueFeatureGroup,
)
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import PandasSklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.python_dict import PythonDictTextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup
from tests.test_plugins.feature_group.experimental.columnwise_hooks_test_mixin import (
    ADD_HOOK,
    CHECK_HOOK,
    DISCOVERY_HOOK,
    ColumnDiscoveryHooksTestMixin,
    ColumnwiseHooksTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_check_source_features_signature import (
    STRICT_PANDAS_CLASSES,
    TOLERANT_PANDAS_CLASSES,
)

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None  # type: ignore

HOOK_NAMES: tuple[str, ...] = (DISCOVERY_HOOK, CHECK_HOOK, ADD_HOOK)

# Anchor the scan root to the repo layout via __file__, not the cwd: a cwd-relative root makes the
# rglob loop empty and the sweep pass vacuously. This file sits four parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group" / "experimental"
assert SCAN_ROOT.exists(), f"scan root not found; check the parents index for the repo root: {SCAN_ROOT}"

# Lower bound only: a new experimental family adds a base.py and must not fail this sweep.
MIN_BASE_MODULES = 12
PANDAS_CONSUMER_COUNT = 12

# Single source of truth for the tolerant/strict split: a strict group raises when ANY source name is
# missing, a tolerant one only when NONE of them exists. Every strict fixture below reads this map,
# and test_strictness_map_agrees_with_signature_lists pins it against the signature suite's lists.
STRICTNESS: dict[type[Any], bool] = {
    PandasAggregatedFeatureGroup: False,
    PandasClusteringFeatureGroup: False,
    PandasForecastingFeatureGroup: False,
    PandasTimeWindowFeatureGroup: False,
    PandasMissingValueFeatureGroup: True,
    PandasDimensionalityReductionFeatureGroup: True,
    PandasGeoDistanceFeatureGroup: True,
    PandasNodeCentralityFeatureGroup: True,
    PandasEncodingFeatureGroup: True,
    PandasSklearnPipelineFeatureGroup: True,
    PandasScalingFeatureGroup: True,
    PandasTextCleaningFeatureGroup: True,
    PolarsLazyAggregatedFeatureGroup: False,
    PyArrowAggregatedFeatureGroup: False,
    PyArrowTimeWindowFeatureGroup: False,
    PyArrowMissingValueFeatureGroup: True,
    PythonDictMissingValueFeatureGroup: True,
    PythonDictTextCleaningFeatureGroup: True,
}


def sample_frame() -> pd.DataFrame:
    """Return the two-column frame every pandas hook consumer works on."""
    return pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})


def sample_table() -> pa.Table:
    """Return the two-column table every PyArrow hook consumer works on."""
    return pa.table({"col_a": [1, 2], "col_b": [3, 4]})


def sample_columnar_dict() -> dict[str, list[Any]]:
    """Return the two-column columnar dict every python-dict hook consumer works on."""
    return {"col_a": ["one", "two"], "col_b": ["three", "four"]}


class HooklessConsumer(FeatureChainParserMixin):
    """Inherits FeatureChainParserMixin and overrides no hook, so every hook call must raise."""


HOOK_INVOCATIONS: list[tuple[str, Callable[[Any, Any], Any]]] = [
    (DISCOVERY_HOOK, lambda hook, data: hook(data)),
    (CHECK_HOOK, lambda hook, data: hook(data, ["col_a"])),
    (ADD_HOOK, lambda hook, data: hook(data, "hook_result", [1, 2])),
]


class StrictnessFromMap:
    """Supplies the strict fixture from STRICTNESS, so no consumer restates the split."""

    @pytest.fixture
    def strict(self, plugin_class: Any) -> bool:
        return STRICTNESS[plugin_class]


class PyArrowColumns:
    """Reads column names off a PyArrow Table."""

    def column_names(self, data: Any) -> list[str]:
        return [str(name) for name in data.schema.names]


class PythonDictColumns:
    """Reads column names off a columnar ``dict[str, list]``, and writes row-aligned lists."""

    def column_names(self, data: Any) -> list[str]:
        return [str(name) for name in data]

    def make_result(self, sample_data: Any) -> Any:
        """Return a row-aligned list, the only shape the columnar dict writer accepts."""
        return [str(index) for index in range(row_count(sample_data))]


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_feature_chain_parser_mixin_declares_hook(hook_name: str) -> None:
    """The mixin owns all three hook declarations, and each one is callable."""
    hook = getattr(FeatureChainParserMixin, hook_name, None)
    assert hook is not None, f"FeatureChainParserMixin does not declare {hook_name}"
    assert callable(hook), f"FeatureChainParserMixin.{hook_name} is not callable"


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_hook_is_not_abstract(hook_name: str) -> None:
    """The hooks must not be decorated @abstractmethod: on a non-ABC class that enforces nothing."""
    hook = getattr(FeatureChainParserMixin, hook_name, None)
    assert hook is not None, f"FeatureChainParserMixin does not declare {hook_name}"
    assert getattr(hook, "__isabstractmethod__", False) is False, f"{hook_name} is marked abstract"


def test_feature_chain_parser_mixin_is_not_an_abc() -> None:
    """The mixin stays a plain class, which is why the hooks raise instead of being abstract."""
    assert not issubclass(FeatureChainParserMixin, ABC)
    assert not isinstance(FeatureChainParserMixin, ABCMeta)


@pytest.mark.parametrize(("hook_name", "invoke"), HOOK_INVOCATIONS, ids=[name for name, _ in HOOK_INVOCATIONS])
def test_unimplemented_hook_raises_not_implemented(hook_name: str, invoke: Callable[[Any, Any], Any]) -> None:
    """A subclass that does not implement a hook fails loudly, naming the class and the hook."""
    hook = getattr(HooklessConsumer, hook_name, None)
    assert hook is not None, f"FeatureChainParserMixin does not declare {hook_name}"

    with pytest.raises(NotImplementedError) as exc_info:
        invoke(hook, sample_frame())

    message = str(exc_info.value)
    assert HooklessConsumer.__name__ in message, f"{hook_name} error omits the class name: {message}"
    assert hook_name in message, f"{hook_name} error omits the hook name: {message}"


def test_strictness_map_agrees_with_signature_lists() -> None:
    """The strict/tolerant split here must not drift from the lists the signature suite drives."""
    from_lists: dict[type[Any], bool] = {cls: True for cls in STRICT_PANDAS_CLASSES}
    from_lists.update({cls: False for cls in TOLERANT_PANDAS_CLASSES})
    assert len(from_lists) == PANDAS_CONSUMER_COUNT, (
        f"signature lists cover {len(from_lists)} pandas classes, expected {PANDAS_CONSUMER_COUNT}"
    )
    mismatches = sorted(cls.__name__ for cls, strict in from_lists.items() if STRICTNESS.get(cls) is not strict)
    assert mismatches == [], f"STRICTNESS disagrees with test_check_source_features_signature for: {mismatches}"


@pytest.mark.parametrize("plugin_class", list(STRICTNESS), ids=[cls.__name__ for cls in STRICTNESS])
def test_class_declares_its_source_feature_strictness(plugin_class: type[Any]) -> None:
    """The table and the code attribute must not drift, so no class silently flips policy during the migration."""
    assert plugin_class.STRICT_SOURCE_FEATURES is STRICTNESS[plugin_class], (
        f"{plugin_class.__name__}.STRICT_SOURCE_FEATURES is {plugin_class.STRICT_SOURCE_FEATURES!r}, "
        f"STRICTNESS says {STRICTNESS[plugin_class]!r}"
    )


def test_no_base_module_declares_a_hook_in_source() -> None:
    """Static sweep: no base.py under the experimental tree re-adds a hook declaration to a class body."""
    offenders: list[str] = []
    visited = 0
    for path in sorted(SCAN_ROOT.rglob("base.py")):
        visited += 1
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            declared = sorted(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in HOOK_NAMES
            )
            if declared:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}::{node.name} declares {declared}")
    assert offenders == [], f"base modules must inherit the hook declarations, found: {offenders}"
    assert visited >= MIN_BASE_MODULES, (
        f"sweep visited only {visited} base.py files under {SCAN_ROOT}: the sweep root is wrong or the glob broke"
    )


class TestPandasAggregatedHooks(StrictnessFromMap, ColumnDiscoveryHooksTestMixin):
    """Aggregated pandas group: tolerant check, plus column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasAggregatedFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasClusteringHooks(StrictnessFromMap, ColumnDiscoveryHooksTestMixin):
    """Clustering pandas group: tolerant check, plus column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasClusteringFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasMissingValueHooks(StrictnessFromMap, ColumnDiscoveryHooksTestMixin):
    """Missing value pandas group: strict check, plus column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasMissingValueFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasForecastingHooks(StrictnessFromMap, ColumnDiscoveryHooksTestMixin):
    """Forecasting pandas group: tolerant check, plus column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasForecastingFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasTimeWindowHooks(StrictnessFromMap, ColumnDiscoveryHooksTestMixin):
    """Time window pandas group: tolerant check, plus column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasTimeWindowFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasDimensionalityReductionHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Dimensionality reduction pandas group: strict check, and one result expands to per-dimension columns."""

    result_feature_name = "col_a__pca_2d"

    def make_result(self, sample_data: Any) -> Any:
        """Return a two-dimensional reduction result with one row per data row."""
        rows = len(sample_data)
        return np.arange(rows * 2, dtype=float).reshape(rows, 2)

    def expected_result_columns(self) -> set[str]:
        return {f"{self.result_feature_name}~dim1", f"{self.result_feature_name}~dim2"}

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasDimensionalityReductionFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasGeoDistanceHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Geo distance pandas group: strict check."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasGeoDistanceFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasNodeCentralityHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Node centrality pandas group: strict check."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasNodeCentralityFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasEncodingHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Encoding pandas group: strict check."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasEncodingFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasSklearnPipelineHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Sklearn pipeline pandas group: strict check."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasSklearnPipelineFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasScalingHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Scaling pandas group: strict check."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasScalingFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPandasTextCleaningHooks(StrictnessFromMap, ColumnwiseHooksTestMixin):
    """Text cleaning pandas group: strict check."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PandasTextCleaningFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_frame()


class TestPolarsLazyAggregatedHooks(StrictnessFromMap, ColumnDiscoveryHooksTestMixin):
    """Aggregated Polars lazy group: tolerant check, plus column discovery on the lazy schema."""

    def column_names(self, data: Any) -> list[str]:
        return [str(name) for name in data.collect_schema().names()]

    def make_result(self, sample_data: Any) -> Any:
        """Return a Polars expression, the only shape the lazy writer accepts."""
        return pl.col("col_a")

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PolarsLazyAggregatedFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        return pl.LazyFrame({"col_a": [1, 2], "col_b": [3, 4]})


class TestPyArrowAggregatedHooks(StrictnessFromMap, PyArrowColumns, ColumnDiscoveryHooksTestMixin):
    """Aggregated PyArrow group: tolerant check, plus column discovery."""

    def make_result(self, sample_data: Any) -> Any:
        """Return a scalar, which the aggregated writer broadcasts over the rows."""
        return 1

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PyArrowAggregatedFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_table()


class TestPyArrowMissingValueHooks(StrictnessFromMap, PyArrowColumns, ColumnDiscoveryHooksTestMixin):
    """Missing value PyArrow group: strict check, plus column discovery."""

    def make_result(self, sample_data: Any) -> Any:
        """Return a row-aligned PyArrow array, the shape append_column takes."""
        return pa.array(range(sample_data.num_rows))

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PyArrowMissingValueFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_table()


class TestPyArrowTimeWindowHooks(StrictnessFromMap, PyArrowColumns, ColumnDiscoveryHooksTestMixin):
    """Time window PyArrow group: tolerant check, plus column discovery."""

    def make_result(self, sample_data: Any) -> Any:
        """Return a row-aligned PyArrow array, the shape append_column takes."""
        return pa.array(range(sample_data.num_rows))

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PyArrowTimeWindowFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_table()


class TestPythonDictMissingValueHooks(StrictnessFromMap, PythonDictColumns, ColumnDiscoveryHooksTestMixin):
    """Missing value python-dict group: strict check, plus column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PythonDictMissingValueFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_columnar_dict()


class TestPythonDictTextCleaningHooks(StrictnessFromMap, PythonDictColumns, ColumnwiseHooksTestMixin):
    """Text cleaning python-dict group: strict check, no column discovery."""

    @pytest.fixture
    def plugin_class(self) -> Any:
        return PythonDictTextCleaningFeatureGroup

    @pytest.fixture
    def sample_data(self) -> Any:
        return sample_columnar_dict()
