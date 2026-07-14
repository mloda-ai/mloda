"""Paired engine/diagnostic tests for issue #722 (environment-level), flipped to the Stage 4 target.

Stage 3b rewired resolve_feature onto a standalone environment built by
build_resolution_environment with all available compute frameworks; Stage 4 adds the
compute_frameworks parameter that restricts that environment to a run's exact framework
set. The div-8 debug test below asserts the TARGET CONTRACT and FAILS until Stage 4 lands.

Index (divergence number in issue #722 -> test names):
- #2 (unavailable-framework honesty):
    test_engine_unavailable_only_framework_raises_no_feature_groups (TARGET CONTRACT)
    test_resolve_feature_unavailable_only_framework_reports_not_runnable (TARGET CONTRACT)
- #8 (run framework set, TARGET CONTRACT via the Stage 4 compute_frameworks parameter):
    test_engine_resolution_limited_to_run_framework_set (TARGET CONTRACT)
    test_resolve_feature_compute_frameworks_parameter_limits_run_set (TARGET CONTRACT)
- #9 (registry strict mode now shared):
    test_engine_strict_mode_with_empty_registry_raises (TARGET CONTRACT)
    test_resolve_feature_applies_collector_strict_mode (TARGET CONTRACT)
- #13 (redefinition conflict, shared environment outcome):
    test_engine_redefinition_conflict_raises (TARGET CONTRACT)
    test_resolve_feature_redefinition_conflict_reports_error_with_candidates (TARGET CONTRACT)
    test_get_feature_group_docs_redefinition_conflict_degrades_silently (docs surface, unchanged)
"""

from __future__ import annotations

import linecache
import sys
import textwrap
from typing import Any, Iterator, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.api.plugin_docs import get_feature_group_docs, resolve_feature
from mloda.core.prepare.accessible_plugins import (
    PreFilterPlugins,
    RedefinitionConflictError,
    dedup_feature_group_subclasses,
)
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722C)
# ---------------------------------------------------------------------------


class CfwUnavailable722C(ComputeFramework):
    """Framework whose backing dependency is never installed."""

    @staticmethod
    def is_available() -> bool:
        return False


class CfwEnabled722C(ComputeFramework):
    """Available framework, the one enabled for the run."""


class CfwOther722C(ComputeFramework):
    """Available framework, NOT enabled for the run."""


class CfwStrictRun722C(ComputeFramework):
    """Available framework carrying the strict-mode run."""


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722c_* names)
# ---------------------------------------------------------------------------


class ProbeUnavailable722C(FeatureGroup):
    """Declares ONLY the unavailable framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwUnavailable722C}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == "probe722c_unavailable"


class ProbeTwoFw722C(FeatureGroup):
    """Declares two available frameworks; the run enables only one of them."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwEnabled722C, CfwOther722C}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == "probe722c_twofw"


class ProbeStrict722C(FeatureGroup):
    """Concrete probe that is never registered in the injected registry."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwStrictRun722C}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == "probe722c_strict"


# ---------------------------------------------------------------------------
# Jupyter-cell simulation helpers for divergence #13
# (construction pattern mirrors tests/test_core/test_prepare/test_feature_group_dedup.py)
# ---------------------------------------------------------------------------

# Keeps exec'd classes alive across assertions, like IPython's Out[N] history. Leaking them for the process
# lifetime is safe: the autouse cleanup unbinds them from __main__, so they are no longer live-in-module and
# dedup preserves instead of re-raising, and they only match the unique name probe722c_redef.
_REF_STORE: list[Any] = []


def _exec_fg_in_main(class_name: str, body: str, cell_label: str) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into __main__ with linecache-backed source (Jupyter cell simulation)."""
    main_mod = sys.modules["__main__"]
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    exec(compile(src, filename, "exec"), main_mod.__dict__)  # nosec B102
    return cast(type[FeatureGroup], main_mod.__dict__[class_name])


def _make_fg_source(class_name: str, feature_name: str, extra_body: str = "") -> str:
    return f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_

class {class_name}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{feature_name}"}}
{extra_body}
"""


def _make_conflicting_pair(qualname: str, feature_name: str) -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Two classes sharing (module, qualname) with DIFFERENT source; v2 is live in __main__."""
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(qualname, feature_name, extra_body="    def extra_method(self):\n        return 722\n")
    v1 = _exec_fg_in_main(qualname, src_v1, f"cell-{qualname}-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, f"cell-{qualname}-v2")
    _REF_STORE.extend([v1, v2])
    return v1, v2


@pytest.fixture(autouse=True)
def _cleanup_main_module_attrs() -> Iterator[None]:
    """Snapshot __main__ attrs and restore after each test (xdist safety, mirrors test_feature_group_dedup)."""
    main_mod = sys.modules["__main__"]
    snapshot = set(main_mod.__dict__.keys())
    yield
    for key in set(main_mod.__dict__.keys()) - snapshot:
        main_mod.__dict__.pop(key, None)


# ---------------------------------------------------------------------------
# Divergence #2: a feature whose ONLY framework is unavailable
# ---------------------------------------------------------------------------


def test_engine_unavailable_only_framework_raises_no_feature_groups() -> None:
    """TARGET CONTRACT: the engine environment drops unavailable frameworks and resolution fails."""
    assert CfwUnavailable722C.is_available() is False  # premise guard

    collector = PluginCollector.enabled_feature_groups({ProbeUnavailable722C})
    accessible = PreFilterPlugins(
        compute_frameworks={CfwUnavailable722C}, plugin_collector=collector
    ).get_accessible_plugins()

    # TARGET CONTRACT: get_cfw_subclasses drops unavailable frameworks; the probe maps to an EMPTY set.
    assert accessible == {ProbeUnavailable722C: set()}

    # TARGET CONTRACT: the engine refuses to resolve the feature.
    with pytest.raises(ValueError, match="No feature groups found"):
        IdentifyFeatureGroupClass(
            feature=Feature("probe722c_unavailable"),
            accessible_plugins=accessible,
            links=None,
        )


def test_resolve_feature_unavailable_only_framework_reports_not_runnable() -> None:
    """resolve_feature is honest about a feature the engine can never run."""
    assert CfwUnavailable722C.is_available() is False  # premise guard

    collector = PluginCollector.enabled_feature_groups({ProbeUnavailable722C})
    result = resolve_feature("probe722c_unavailable", plugin_collector=collector)

    # TARGET CONTRACT (#2): the standalone environment drops unavailable frameworks exactly like
    # the engine's, so the probe has no accessible framework and the feature is reported as not
    # runnable on any available framework instead of silently runnable.
    assert result.feature_group is None
    assert result.error is not None
    assert result.supported_compute_frameworks == []


# ---------------------------------------------------------------------------
# Divergence #8: the run's compute-framework set
# ---------------------------------------------------------------------------


def test_engine_resolution_limited_to_run_framework_set() -> None:
    """TARGET CONTRACT: the engine intersects declared frameworks with the run's set."""
    assert CfwEnabled722C.is_available() is True  # premise guard
    assert CfwOther722C.is_available() is True  # premise guard

    collector = PluginCollector.enabled_feature_groups({ProbeTwoFw722C})
    accessible = PreFilterPlugins(
        compute_frameworks={CfwEnabled722C}, plugin_collector=collector
    ).get_accessible_plugins()

    # TARGET CONTRACT: only the run-enabled framework survives the environment build.
    assert accessible == {ProbeTwoFw722C: {CfwEnabled722C}}

    identifier = IdentifyFeatureGroupClass(
        feature=Feature("probe722c_twofw"),
        accessible_plugins=accessible,
        links=None,
    )
    resolved, frameworks = identifier.get()
    assert resolved is ProbeTwoFw722C
    # TARGET CONTRACT: per-feature resolution carries only the run's framework set.
    assert frameworks == {CfwEnabled722C}


def test_resolve_feature_compute_frameworks_parameter_limits_run_set() -> None:
    """The compute_frameworks parameter restricts the standalone environment to the run's set."""
    collector = PluginCollector.enabled_feature_groups({ProbeTwoFw722C})

    # Premise guard: without a run set the standalone default reports every installed
    # declared framework, exactly like an all-frameworks engine environment.
    unrestricted = resolve_feature("probe722c_twofw", plugin_collector=collector)
    assert unrestricted.feature_group is ProbeTwoFw722C
    assert unrestricted.supported_compute_frameworks == ["CfwEnabled722C", "CfwOther722C"]

    result = resolve_feature("probe722c_twofw", compute_frameworks={CfwEnabled722C}, plugin_collector=collector)

    assert result.feature_group is ProbeTwoFw722C
    assert result.error is None
    # TARGET CONTRACT (#8, Stage 4): compute_frameworks restricts the standalone environment
    # exactly like mlodaAPI(compute_frameworks=...) restricts the engine's run set, so only
    # the enabled framework is reported.
    assert result.supported_compute_frameworks == ["CfwEnabled722C"]
    assert result.unsupported_compute_frameworks == []


# ---------------------------------------------------------------------------
# Divergence #9: registry strict mode
# ---------------------------------------------------------------------------


def _strict_collector_with_empty_registry() -> PluginCollector:
    """Enable only the strict probe, strict mode on, EMPTY injected registry (global registry untouched)."""
    collector = PluginCollector.enabled_feature_groups({ProbeStrict722C})
    return collector.set_strict_mode("strict").set_registry(PluginRegistry())


def test_engine_strict_mode_with_empty_registry_raises() -> None:
    """TARGET CONTRACT: strict mode drops the unregistered concrete probe and the environment build fails."""
    collector = _strict_collector_with_empty_registry()

    with pytest.raises(ValueError, match="Strict mode filtered out all FeatureGroups"):
        PreFilterPlugins(compute_frameworks={CfwStrictRun722C}, plugin_collector=collector)


def test_resolve_feature_applies_collector_strict_mode() -> None:
    """resolve_feature honors the collector's strict mode through the shared environment build."""
    collector = _strict_collector_with_empty_registry()
    result = resolve_feature("probe722c_strict", plugin_collector=collector)

    # TARGET CONTRACT (#9): the standalone environment applies strict mode exactly like the
    # engine's; with an empty registry the build fails and the environment error is returned.
    assert result.feature_group is None
    assert result.error is not None
    assert "Strict mode filtered out all FeatureGroups" in result.error


# ---------------------------------------------------------------------------
# Divergence #13: one redefinition conflict, three different semantics
# ---------------------------------------------------------------------------


def test_engine_redefinition_conflict_raises() -> None:
    """Engine semantic: dedup raises RedefinitionConflictError (a ValueError) and construction dies."""
    v1, v2 = _make_conflicting_pair("ProbeRedefDedup722C", "probe722c_redef")

    # TARGET CONTRACT (#13): the engine environment build keeps raising on a genuine conflict;
    # both paths project the same environment build outcome, the engine as a raise.
    with pytest.raises(RedefinitionConflictError) as exc_info:
        dedup_feature_group_subclasses({v1, v2}, allow_redefinition=False)
    assert "redefined with different source code" in str(exc_info.value)
    assert isinstance(exc_info.value, ValueError)

    # Engine environment construction hits the same conflict.
    collector = PluginCollector.enabled_feature_groups({v1, v2})
    with pytest.raises(ValueError, match="redefined with different source code"):
        PreFilterPlugins(compute_frameworks={CfwEnabled722C}, plugin_collector=collector)


def test_resolve_feature_redefinition_conflict_reports_error_with_candidates() -> None:
    """Debug semantic: same conflict degrades to an error string with the conflicting candidates."""
    v1, v2 = _make_conflicting_pair("ProbeRedefResolve722C", "probe722c_redef")
    collector = PluginCollector.enabled_feature_groups({v1, v2})

    result = resolve_feature("probe722c_redef", plugin_collector=collector)

    # TARGET CONTRACT (#13): both paths project the same environment build outcome; the debug
    # path returns it as an error field with the scope- and name-matching conflicting candidates.
    assert result.feature_group is None
    assert result.error is not None
    assert "redefined with different source code" in result.error
    assert set(result.candidates) == {v1, v2}


def test_get_feature_group_docs_redefinition_conflict_degrades_silently() -> None:
    """Docs semantic: same conflict is silently collapsed to the live class; the call does not raise."""
    v1, v2 = _make_conflicting_pair("ProbeRedefDocs722C", "probe722c_redef")
    collector = PluginCollector.enabled_feature_groups({v1, v2})

    # Premise guard: dedup raises on this fixture, proving the degrade path below handles a genuine conflict.
    with pytest.raises(RedefinitionConflictError):
        dedup_feature_group_subclasses({v1, v2}, allow_redefinition=False)

    # DOCS SURFACE (#13, unchanged by Stage 3b): get_feature_group_docs deliberately degrades a
    # conflict to the live class; only the engine and resolve_feature share the environment outcome.
    result = get_feature_group_docs(plugin_collector=collector)
    assert isinstance(result, list)
