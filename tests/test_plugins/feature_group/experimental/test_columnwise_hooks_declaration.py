"""Pins the DECLARATION of the column-wise hook contract: which family needs which hook.

The three hooks live on FeatureChainParserMixin as raising defaults, so a downstream author who
writes a new compute-framework implementation gets no signal until the run reaches the hook. The
declaration closes that gap: every family base states the hooks its own calculate_feature calls in
``REQUIRED_COLUMNWISE_HOOKS``, and the class-definition guard reads it.

A declaration that drifts from the code is worse than none, so the sweep at the end of this module
compares each declaration against a static scan of the hooks the base module actually calls.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards import (
    warn_missing_columnwise_hooks,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    COLUMN_DISCOVERY_HOOKS,
    COLUMNWISE_HOOKS,
    FeatureChainParserMixin,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
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
from tests.test_plugins.feature_group.experimental.columnwise_hooks_test_mixin import (
    ADD_HOOK,
    CHECK_HOOK,
    DISCOVERY_HOOK,
    resolved_hook,
)
from tests.test_plugins.feature_group.experimental.test_columnwise_hooks_contract import (
    HOOK_NAMES,
    MIN_BASE_MODULES,
    SCAN_ROOT,
    STRICTNESS,
)

AUTHOR_GUARDS_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"

# The expected declaration per family base, kept as a table so a changed declaration is a visible diff.
# A family that resolves column names against the data needs the discovery hook on top of the pair.
FAMILY_REQUIREMENTS: dict[type[Any], frozenset[str]] = {
    AggregatedFeatureGroup: COLUMN_DISCOVERY_HOOKS,
    ClusteringFeatureGroup: COLUMN_DISCOVERY_HOOKS,
    MissingValueFeatureGroup: COLUMN_DISCOVERY_HOOKS,
    ForecastingFeatureGroup: COLUMN_DISCOVERY_HOOKS,
    TimeWindowFeatureGroup: COLUMN_DISCOVERY_HOOKS,
    DimensionalityReductionFeatureGroup: COLUMNWISE_HOOKS,
    GeoDistanceFeatureGroup: COLUMNWISE_HOOKS,
    NodeCentralityFeatureGroup: COLUMNWISE_HOOKS,
    EncodingFeatureGroup: COLUMNWISE_HOOKS,
    SklearnPipelineFeatureGroup: COLUMNWISE_HOOKS,
    ScalingFeatureGroup: COLUMNWISE_HOOKS,
    TextCleaningFeatureGroup: COLUMNWISE_HOOKS,
}

FAMILY_IDS = [family.__name__ for family in FAMILY_REQUIREMENTS]


def _module_name_of(path: Path) -> str:
    """Dotted module name of a plugin file, derived from the scan root so no path is hard-coded."""
    relative = path.relative_to(SCAN_ROOT.parents[2])
    return ".".join(relative.with_suffix("").parts)


def _hooks_called_in(path: Path) -> set[str]:
    """The three hooks a module calls on ``cls`` / ``self``, anywhere in the module."""
    called: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in HOOK_NAMES:
            continue
        if isinstance(func.value, ast.Name) and func.value.id in ("cls", "self"):
            called.add(func.attr)
    return called


def _mixin_subclasses_defined_in(module_name: str) -> list[type[Any]]:
    """The FeatureChainParserMixin subclasses a module defines itself, ignoring imported ones."""
    module = importlib.import_module(module_name)
    return [
        member
        for _name, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, FeatureChainParserMixin) and member.__module__ == module_name
    ]


def test_columnwise_hooks_constant_names_the_two_write_hooks() -> None:
    """COLUMNWISE_HOOKS is the pair every column-wise family needs: the check and the writer."""
    assert COLUMNWISE_HOOKS == frozenset({CHECK_HOOK, ADD_HOOK})
    assert isinstance(COLUMNWISE_HOOKS, frozenset)


def test_column_discovery_hooks_constant_adds_the_discovery_hook() -> None:
    """COLUMN_DISCOVERY_HOOKS is the pair plus the discovery hook, so the two constants cannot drift apart."""
    assert COLUMN_DISCOVERY_HOOKS == COLUMNWISE_HOOKS | {DISCOVERY_HOOK}
    assert isinstance(COLUMN_DISCOVERY_HOOKS, frozenset)


def test_mixin_requires_no_hook_by_default() -> None:
    """The mixin itself declares no requirement, so an ordinary parse-only feature group is unaffected."""
    assert FeatureChainParserMixin.REQUIRED_COLUMNWISE_HOOKS == frozenset()


@pytest.mark.parametrize(("family", "expected"), list(FAMILY_REQUIREMENTS.items()), ids=FAMILY_IDS)
def test_family_base_declares_its_required_hooks(family: type[Any], expected: frozenset[str]) -> None:
    """Each family base states the hooks its calculate_feature calls."""
    assert family.REQUIRED_COLUMNWISE_HOOKS == expected


@pytest.mark.parametrize("family", list(FAMILY_REQUIREMENTS), ids=FAMILY_IDS)
def test_family_base_owns_its_declaration(family: type[Any]) -> None:
    """The declaration sits in the family's own class body, not inherited from a sibling or the mixin."""
    assert "REQUIRED_COLUMNWISE_HOOKS" in family.__dict__


def test_declaration_matches_the_hooks_each_base_module_calls() -> None:
    """Anti-drift sweep: a base's declaration must equal the hooks its own module calls on cls.

    This is the test that stops the declaration going stale: adding a ``cls._get_available_columns``
    call to a base without widening its declaration fails here, as does declaring a hook nothing calls.
    """
    mismatches: list[str] = []
    visited = 0
    for path in sorted(SCAN_ROOT.rglob("base.py")):
        visited += 1
        called = _hooks_called_in(path)
        families = _mixin_subclasses_defined_in(_module_name_of(path))
        if len(families) != 1:
            mismatches.append(f"{path.relative_to(SCAN_ROOT)} defines {len(families)} mixin classes, expected 1")
            continue
        declared = set(families[0].REQUIRED_COLUMNWISE_HOOKS)
        if declared != called:
            mismatches.append(
                f"{families[0].__name__} declares {sorted(declared)} but its module calls {sorted(called)}"
            )
    assert mismatches == [], f"REQUIRED_COLUMNWISE_HOOKS drifted from the code: {mismatches}"
    assert visited >= MIN_BASE_MODULES, (
        f"sweep visited only {visited} base.py files under {SCAN_ROOT}: the sweep root is wrong or the glob broke"
    )


@pytest.mark.parametrize("plugin_class", list(STRICTNESS), ids=[cls.__name__ for cls in STRICTNESS])
def test_shipped_family_class_declares_a_requirement_it_implements(plugin_class: type[Any]) -> None:
    """Precondition for the silence sweep: every shipped class is framework-bound and carries a requirement."""
    assert "compute_framework_rule" in plugin_class.__dict__, (
        f"{plugin_class.__name__} is not framework-bound, so the guard would skip it vacuously"
    )
    assert plugin_class.REQUIRED_COLUMNWISE_HOOKS, f"{plugin_class.__name__} inherits an empty requirement"


@pytest.mark.parametrize("plugin_class", list(STRICTNESS), ids=[cls.__name__ for cls in STRICTNESS])
def test_shipped_family_class_reports_no_missing_hook(
    plugin_class: type[Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The new guard is quiet on the current tree: no shipped class leaves a required hook unimplemented."""
    default_hooks = {hook: resolved_hook(FeatureChainParserMixin, hook) for hook in HOOK_NAMES}
    missing = sorted(
        hook
        for hook in plugin_class.REQUIRED_COLUMNWISE_HOOKS
        if resolved_hook(plugin_class, hook) is default_hooks[hook]
    )
    assert missing == [], f"{plugin_class.__name__} inherits the raising default for {missing}"

    with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):
        warn_missing_columnwise_hooks(plugin_class)

    warnings = [record.getMessage() for record in caplog.records if record.name == AUTHOR_GUARDS_LOGGER]
    assert warnings == [], f"the guard warns about the shipped {plugin_class.__name__}: {warnings}"
