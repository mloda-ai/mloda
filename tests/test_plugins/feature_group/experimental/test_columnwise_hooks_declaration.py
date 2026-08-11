"""Pins the DECLARATION of the column-wise hook contract: which family needs which hook.

The three hooks live on FeatureChainParserMixin as raising defaults, so a downstream author who
writes a new compute-framework implementation gets no signal until the run reaches the hook. The
declaration closes that gap: every family base states the hooks its own calculate_feature calls in
``REQUIRED_COLUMNWISE_HOOKS``, and ``missing_columnwise_hooks`` reads it back in a test.

A declaration that drifts from the code is worse than none, so the sweep at the end of this module
compares each declaration against a static scan of the hooks the base module actually calls.

The contract is also an AUTHORING surface: a plugin author must reach it through ``mloda.provider``,
the documented facade, without reading core source. The facade tests and the import sweep below pin
that, so no family base is allowed to reach into the deep core path again.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import mloda.provider
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    COLUMN_DISCOVERY_HOOKS,
    COLUMNWISE_HOOKS,
    FeatureChainParserMixin,
    missing_columnwise_hooks,
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

MIXIN_MODULE = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin"

# The core package a plugin module must never import from: the facade is the authoring surface.
CORE_CHAINER_PACKAGE = "mloda.core.abstract_plugins.components.feature_chainer"

# The contract a plugin author needs: the two declarations and the reader over them.
CONTRACT_SYMBOLS = (
    "COLUMNWISE_HOOKS",
    "COLUMN_DISCOVERY_HOOKS",
    "missing_columnwise_hooks",
)

# Lower bounds only, so a broken glob or a wrong root cannot make the widened sweeps pass vacuously.
MIN_EXPERIMENTAL_MODULES = 40
# The structural invariant: every one of the 12 family bases calls its own hooks. A framework module
# that calls one too is above this floor, so sharing the hooks out of the tree cannot break the sweep.
MIN_HOOK_CALLING_MODULES = 12

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


def _owning_family(path: Path, declarations: dict[Path, type[Any]]) -> type[Any] | None:
    """The family base of the nearest ancestor directory owning a base.py, or None for a family-less module."""
    for directory in path.parents:
        family = declarations.get(directory)
        if family is not None:
            return family
        if directory == SCAN_ROOT:
            break
    return None


def _mixin_subclasses_defined_in(module_name: str) -> list[type[Any]]:
    """The FeatureChainParserMixin subclasses a module defines itself, ignoring imported ones."""
    module = importlib.import_module(module_name)
    return [
        member
        for _name, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, FeatureChainParserMixin) and member.__module__ == module_name
    ]


@pytest.mark.parametrize("symbol", CONTRACT_SYMBOLS)
def test_provider_facade_lists_the_contract_symbol(symbol: str) -> None:
    """mloda.provider is the documented plugin-author facade, so the contract must be part of its API."""
    assert symbol in mloda.provider.__all__, f"mloda.provider must list '{symbol}' in __all__"


@pytest.mark.parametrize("symbol", CONTRACT_SYMBOLS)
def test_provider_facade_symbol_is_the_core_object(symbol: str) -> None:
    """The facade re-exports, it does not redefine: a copy would drift from the core declaration."""
    core = importlib.import_module(MIXIN_MODULE)
    assert hasattr(core, symbol), f"{MIXIN_MODULE} must define '{symbol}'"
    assert hasattr(mloda.provider, symbol), f"mloda.provider must expose '{symbol}'"
    assert getattr(mloda.provider, symbol) is getattr(core, symbol), (
        f"mloda.provider.{symbol} must be the identical object from {MIXIN_MODULE}"
    )


def test_experimental_plugins_reach_the_contract_through_the_facade() -> None:
    """No experimental plugin module imports the feature_chainer core package directly.

    Anti-regression sweep: before the declaration landed this count was zero, and a plugin author who
    can only find the contract behind a six-segment private path has not been given a contract.
    """
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


def test_no_family_module_calls_a_hook_its_base_does_not_declare() -> None:
    """Widened anti-drift sweep: a hook called ANYWHERE in a family must appear in that family's declaration.

    The base-module sweep above only sees base.py, but the hooks are called from framework modules too
    (missing_value/python_dict.py already calls the discovery hook). A framework module reaching for a
    hook its family base never declared is exactly the drift the declaration is supposed to prevent.
    """
    declarations = _family_declarations()
    offenders: list[str] = []
    visited = 0
    calling = 0
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        visited += 1
        called = _hooks_called_in(path)
        if not called:
            continue
        calling += 1
        family = _owning_family(path, declarations)
        if family is None:
            offenders.append(f"{path.relative_to(SCAN_ROOT)} calls {sorted(called)} but belongs to no family base")
            continue
        undeclared = sorted(called - set(family.REQUIRED_COLUMNWISE_HOOKS))
        if undeclared:
            offenders.append(
                f"{path.relative_to(SCAN_ROOT)} calls {undeclared}, which {family.__name__} does not declare"
            )
    assert offenders == [], f"REQUIRED_COLUMNWISE_HOOKS drifted from the code: {offenders}"
    assert visited >= MIN_EXPERIMENTAL_MODULES, (
        f"sweep visited only {visited} modules under {SCAN_ROOT}: the sweep root is wrong or the glob broke"
    )
    assert calling >= MIN_HOOK_CALLING_MODULES, (
        f"sweep found only {calling} hook-calling modules under {SCAN_ROOT}: the call detection broke"
    )


@pytest.mark.parametrize("plugin_class", list(STRICTNESS), ids=[cls.__name__ for cls in STRICTNESS])
def test_shipped_family_class_declares_a_requirement_it_implements(plugin_class: type[Any]) -> None:
    """Precondition for the sweep below: every shipped class is framework-bound and carries a requirement."""
    assert "compute_framework_rule" in plugin_class.__dict__, (
        f"{plugin_class.__name__} is not framework-bound, so it is no implementation to check"
    )
    assert plugin_class.REQUIRED_COLUMNWISE_HOOKS, f"{plugin_class.__name__} inherits an empty requirement"


@pytest.mark.parametrize("plugin_class", list(STRICTNESS), ids=[cls.__name__ for cls in STRICTNESS])
def test_shipped_family_class_reports_no_missing_hook(plugin_class: type[Any]) -> None:
    """This is the check a plugin repo runs: no shipped class leaves a declared hook unimplemented.

    The hand-rolled comparison is an independent oracle: it must agree with the exported reader, so a
    reader that silently reports nothing cannot pass this sweep.
    """
    default_hooks = {hook: resolved_hook(FeatureChainParserMixin, hook) for hook in HOOK_NAMES}
    expected = sorted(
        hook
        for hook in plugin_class.REQUIRED_COLUMNWISE_HOOKS
        if resolved_hook(plugin_class, hook) is default_hooks[hook]
    )
    assert expected == [], f"{plugin_class.__name__} inherits the raising default for {expected}"
    assert missing_columnwise_hooks(plugin_class) == expected, (
        f"missing_columnwise_hooks disagrees with the direct hook comparison for {plugin_class.__name__}"
    )
