"""Tests for tri-state strict mode on ComputeFrameworks (issue #526, cycle 5).

Contract: PreFilterPlugins._set_compute_frameworks mirrors the FeatureGroup
tri-state. "off" keeps today's behavior (intersection with available
subclasses). "strict" keeps only registered (or abstract) ComputeFramework
classes, raising a ValueError that names ComputeFrameworks and how to
register when every concrete framework from the pre-filter set is dropped.
"warn" filters nothing but emits ONE aggregated WARNING per process per
unregistered concrete framework, keyed module:qualname and deduplicated via
the existing _warned_unregistered set (cleared per test by conftest).
Registry lookups go through the injected registry (registry_for), not a
hard-coded PluginRegistry.default().

All tests are written to FAIL until the feature exists. Assertions are
membership-only on this module's local doubles, so the tests stay xdist
parallel-safe.
"""

import inspect
import logging
from abc import abstractmethod

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.prepare.accessible_plugins import PreFilterPlugins


class _CfwStrictObserverFG(FeatureGroup):
    """Local observer: compute_framework_rule is None, so its accessible mapping
    value equals exactly the set of ComputeFrameworks that survived filtering."""


class _CfwStrictRegistered(ComputeFramework):
    """Local concrete framework that tests register explicitly."""


class _CfwStrictUnregisteredA(ComputeFramework):
    """Local concrete framework that is never registered."""


class _CfwStrictUnregisteredB(ComputeFramework):
    """Second never-registered concrete framework, for aggregation tests."""


class _CfwStrictInjectedOnly(ComputeFramework):
    """Local concrete framework registered ONLY in a fresh injected registry."""


class _CfwStrictAbstractInfra(ComputeFramework):
    """Abstract local framework; strict and warn must treat it as infrastructure."""

    @abstractmethod
    def _cfw_strict_abstract_infra_hook(self) -> None: ...


def _qualid(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _cfw_records(caplog: pytest.LogCaptureFixture, cls: type[ComputeFramework]) -> list[str]:
    """Messages from the accessible_plugins logger that name the given framework."""
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "mloda.core.prepare.accessible_plugins" and _qualid(cls) in rec.getMessage()
    ]


class TestCfwStrictModeOff:
    def test_off_mode_keeps_unregistered_cfw_regardless_of_injected_registry(self) -> None:
        """Off mode never filters by registration, even with an (empty) injected registry."""
        collector = PluginCollector().set_strict_mode("off").set_registry(PluginRegistry())
        accessible = PreFilterPlugins({_CfwStrictUnregisteredA}, collector).get_accessible_plugins()
        assert _CfwStrictUnregisteredA in accessible[_CfwStrictObserverFG]


class TestCfwStrictModeStrict:
    def test_strict_keeps_registered_and_drops_unregistered_cfws(self) -> None:
        register_plugin(_CfwStrictObserverFG)
        register_plugin(_CfwStrictRegistered)
        collector = PluginCollector().set_strict_mode("strict")

        accessible = PreFilterPlugins({_CfwStrictRegistered, _CfwStrictUnregisteredA}, collector)
        surviving = accessible.get_accessible_plugins()[_CfwStrictObserverFG]

        assert _CfwStrictRegistered in surviving
        assert _CfwStrictUnregisteredA not in surviving, (
            "strict mode must drop concrete ComputeFrameworks that are not registered"
        )

    def test_strict_keeps_unregistered_abstract_cfws(self) -> None:
        assert inspect.isabstract(_CfwStrictAbstractInfra), "sanity: the local double is abstract"

        register_plugin(_CfwStrictObserverFG)
        register_plugin(_CfwStrictRegistered)
        collector = PluginCollector().set_strict_mode("strict")

        cfws: set[type[ComputeFramework]] = {_CfwStrictRegistered, _CfwStrictAbstractInfra, _CfwStrictUnregisteredA}
        surviving = PreFilterPlugins(cfws, collector).get_accessible_plugins()[_CfwStrictObserverFG]

        assert _CfwStrictAbstractInfra in surviving, (
            "abstract ComputeFrameworks are infrastructure and must survive strict mode unregistered"
        )
        assert _CfwStrictUnregisteredA not in surviving, (
            "sanity: strict mode still drops concrete unregistered ComputeFrameworks"
        )

    def test_strict_dropping_all_concrete_cfws_raises_with_fix_hint(self) -> None:
        register_plugin(_CfwStrictObserverFG)
        collector = PluginCollector().set_strict_mode("strict")

        with pytest.raises(ValueError) as exc_info:
            PreFilterPlugins({_CfwStrictUnregisteredA, _CfwStrictUnregisteredB}, collector)

        message = str(exc_info.value)
        assert "ComputeFramework" in message, "the empty-result error must name ComputeFrameworks"
        assert "register" in message, "the empty-result error must say how to fix it (registration)"

    def test_strict_consults_injected_registry_for_cfws(self) -> None:
        """Frameworks registered only in the injected registry survive; the default registry is ignored."""
        fresh = PluginRegistry()
        fresh.register(_CfwStrictObserverFG)
        fresh.register(_CfwStrictInjectedOnly)
        register_plugin(_CfwStrictRegistered)  # default registry only: must NOT count

        collector = PluginCollector().set_strict_mode("strict").set_registry(fresh)
        cfws: set[type[ComputeFramework]] = {_CfwStrictInjectedOnly, _CfwStrictRegistered}
        surviving = PreFilterPlugins(cfws, collector).get_accessible_plugins()[_CfwStrictObserverFG]

        assert _CfwStrictInjectedOnly in surviving
        assert _CfwStrictRegistered not in surviving, (
            "a ComputeFramework registered only in the default registry must be dropped "
            "when a different registry is injected"
        )


class TestCfwStrictModeWarn:
    def test_warn_keeps_unregistered_cfw_and_logs_qualified_name(self, caplog: pytest.LogCaptureFixture) -> None:
        register_plugin(_CfwStrictRegistered)
        collector = PluginCollector().set_strict_mode("warn")

        cfws: set[type[ComputeFramework]] = {_CfwStrictRegistered, _CfwStrictAbstractInfra, _CfwStrictUnregisteredA}
        with caplog.at_level(logging.WARNING):
            surviving = PreFilterPlugins(cfws, collector).get_accessible_plugins()[_CfwStrictObserverFG]

        assert _CfwStrictUnregisteredA in surviving, "warn mode must not filter ComputeFrameworks"
        assert _cfw_records(caplog, _CfwStrictUnregisteredA), (
            "warn mode must log a WARNING naming the unregistered framework as module:qualname"
        )
        assert not _cfw_records(caplog, _CfwStrictRegistered), "warn mode must not name registered ComputeFrameworks"
        assert not _cfw_records(caplog, _CfwStrictAbstractInfra), (
            "warn mode must not name abstract infrastructure ComputeFrameworks"
        )

    def test_warn_aggregates_per_construction_and_once_per_process(self, caplog: pytest.LogCaptureFixture) -> None:
        """First construction emits ONE aggregated record naming both unregistered
        frameworks; a second construction stays silent (per-process dedup via
        _warned_unregistered, cleared between tests by conftest)."""
        collector = PluginCollector().set_strict_mode("warn")
        cfws: set[type[ComputeFramework]] = {_CfwStrictUnregisteredA, _CfwStrictUnregisteredB}

        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(cfws, collector)
            PreFilterPlugins(cfws, collector)

        records = [
            rec.getMessage()
            for rec in caplog.records
            if rec.name == "mloda.core.prepare.accessible_plugins"
            and (
                _qualid(_CfwStrictUnregisteredA) in rec.getMessage()
                or _qualid(_CfwStrictUnregisteredB) in rec.getMessage()
            )
        ]
        assert len(records) == 1, (
            f"warn mode must emit one aggregated record per process for unregistered frameworks, got {len(records)}"
        )
        assert _qualid(_CfwStrictUnregisteredA) in records[0]
        assert _qualid(_CfwStrictUnregisteredB) in records[0]
