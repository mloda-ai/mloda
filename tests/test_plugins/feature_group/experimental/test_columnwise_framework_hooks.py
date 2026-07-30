"""Pins that the twelve experimental bases inherit their framework hooks from the shared mixins (os-017)."""

import ast
from pathlib import Path

import pytest

from mloda.provider import FeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.columnwise_framework_hooks import (
    ColumnDiscoveryFrameworkHooks,
    ColumnwiseFrameworkHooks,
)
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

PAIR_HOOKS: frozenset[str] = frozenset({"_check_source_features_exist", "_add_result_to_data"})
DISCOVERY_HOOK = "_get_available_columns"
ALL_HOOKS: frozenset[str] = PAIR_HOOKS | {DISCOVERY_HOOK}

# Anchor the scan root to the repo layout via __file__, not the cwd: a cwd-relative root makes the
# rglob loop empty and the sweep pass vacuously. This file sits four parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group" / "experimental"
assert SCAN_ROOT.exists(), f"scan root not found; check the parents index for the repo root: {SCAN_ROOT}"

EXPECTED_BASE_MODULES = 12

# Bases that resolve column names against the data before computing, so they need the discovery hook.
DISCOVERY_BASES: list[type[FeatureGroup]] = [
    AggregatedFeatureGroup,
    ClusteringFeatureGroup,
    MissingValueFeatureGroup,
    ForecastingFeatureGroup,
    TimeWindowFeatureGroup,
]
assert len(DISCOVERY_BASES) == 5, f"DISCOVERY_BASES lists {len(DISCOVERY_BASES)} bases, expected 5"

# Bases that only check and write columns, so the discovery hook must stay out of their abstract surface.
PAIR_ONLY_BASES: list[type[FeatureGroup]] = [
    DimensionalityReductionFeatureGroup,
    GeoDistanceFeatureGroup,
    NodeCentralityFeatureGroup,
    EncodingFeatureGroup,
    SklearnPipelineFeatureGroup,
    ScalingFeatureGroup,
    TextCleaningFeatureGroup,
]
assert len(PAIR_ONLY_BASES) == 7, f"PAIR_ONLY_BASES lists {len(PAIR_ONLY_BASES)} bases, expected 7"

ALL_BASES: list[type[FeatureGroup]] = DISCOVERY_BASES + PAIR_ONLY_BASES
assert len(ALL_BASES) == EXPECTED_BASE_MODULES, f"ALL_BASES lists {len(ALL_BASES)} bases, expected 12"

EXPECTED_ABSTRACT_HOOKS = [pytest.param(base, ALL_HOOKS, id=base.__name__) for base in DISCOVERY_BASES] + [
    pytest.param(base, PAIR_HOOKS, id=base.__name__) for base in PAIR_ONLY_BASES
]


@pytest.mark.parametrize("base", ALL_BASES, ids=lambda base: base.__name__)
def test_every_base_inherits_columnwise_hooks(base: type[FeatureGroup]) -> None:
    """All twelve bases take the check/add pair from ColumnwiseFrameworkHooks."""
    assert issubclass(base, ColumnwiseFrameworkHooks), f"{base.__name__} does not inherit ColumnwiseFrameworkHooks"


@pytest.mark.parametrize("base", DISCOVERY_BASES, ids=lambda base: base.__name__)
def test_discovery_bases_inherit_discovery_hooks(base: type[FeatureGroup]) -> None:
    """The five column-resolving bases take all three hooks from ColumnDiscoveryFrameworkHooks."""
    assert issubclass(base, ColumnDiscoveryFrameworkHooks), (
        f"{base.__name__} does not inherit ColumnDiscoveryFrameworkHooks"
    )


@pytest.mark.parametrize("base", PAIR_ONLY_BASES, ids=lambda base: base.__name__)
def test_pair_only_bases_are_not_discovery_hooks(base: type[FeatureGroup]) -> None:
    """The seven pair-only bases must not be widened to ColumnDiscoveryFrameworkHooks."""
    assert not issubclass(base, ColumnDiscoveryFrameworkHooks), (
        f"{base.__name__} must not inherit ColumnDiscoveryFrameworkHooks"
    )


@pytest.mark.parametrize("base", ALL_BASES, ids=lambda base: base.__name__)
def test_base_does_not_redeclare_hooks(base: type[FeatureGroup]) -> None:
    """No base redeclares a hook in its own body, the declaration lives only in the mixin."""
    redeclared = sorted(name for name in ALL_HOOKS if name in vars(base))
    assert redeclared == [], f"{base.__name__} still declares {redeclared} itself instead of inheriting them"


@pytest.mark.parametrize(("base", "expected"), EXPECTED_ABSTRACT_HOOKS)
def test_abstract_hook_surface_is_unchanged(base: type[FeatureGroup], expected: frozenset[str]) -> None:
    """Each base keeps exactly the hooks it had as abstract, so no family silently gains one."""
    actual = ALL_HOOKS & set(base.__abstractmethods__)
    assert actual == set(expected), f"{base.__name__} abstract hooks are {sorted(actual)}, expected {sorted(expected)}"


def test_columnwise_hooks_is_not_a_feature_group() -> None:
    """ColumnwiseFrameworkHooks must stay out of the FeatureGroup tree so plugin collection is unaffected."""
    assert not issubclass(ColumnwiseFrameworkHooks, FeatureGroup)


def test_column_discovery_hooks_is_not_a_feature_group() -> None:
    """ColumnDiscoveryFrameworkHooks must stay out of the FeatureGroup tree so plugin collection is unaffected."""
    assert not issubclass(ColumnDiscoveryFrameworkHooks, FeatureGroup)


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
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in ALL_HOOKS
            )
            if declared:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}::{node.name} declares {declared}")
    assert offenders == [], f"base modules must inherit the hooks from the mixins, found: {offenders}"
    assert visited == EXPECTED_BASE_MODULES, (
        f"sweep visited {visited} base.py files under {SCAN_ROOT}, expected {EXPECTED_BASE_MODULES}"
    )


def test_mixins_declare_the_hooks_abstractly() -> None:
    """The mixins own the declarations: the pair on the base mixin, the discovery hook on the subclass."""
    assert PAIR_HOOKS <= set(ColumnwiseFrameworkHooks.__abstractmethods__)
    assert DISCOVERY_HOOK not in ColumnwiseFrameworkHooks.__abstractmethods__
    assert ALL_HOOKS <= set(ColumnDiscoveryFrameworkHooks.__abstractmethods__)
    assert issubclass(ColumnDiscoveryFrameworkHooks, ColumnwiseFrameworkHooks)
