"""Pytest fixture helpers for isolating the process-global plugin registry."""

from __future__ import annotations

import sys
from collections.abc import Generator, Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry


@pytest.fixture
def isolated_plugin_registry() -> Iterator[PluginRegistry]:
    """Snapshot the default registry and policy, yield it, and restore both on teardown."""
    registry = PluginRegistry.default()
    snapshot = registry.snapshot()
    policy = registry.policy
    yield registry
    registry.restore(snapshot)
    registry.set_policy(policy)
    registry._policy_warned_keys.clear()


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Generator[None, Any, Any]:
    """Fail collection of a conformance suite host that skips contract tests without a declared cause."""
    report = yield
    suite_module = sys.modules.get("mloda._testing.suite")
    if suite_module is None or not isinstance(collector, pytest.Class):
        return report
    cls = collector.obj
    if suite_module.ConformanceSuite not in cls.__mro__ or cls._suite_base:
        return report
    inherited = [m.name for node in collector.listchain()[:-1] for m in node.own_markers]
    if not (suite_module.host_has_skip_mark(cls) or suite_module.has_skip_mark_names(inherited)):
        return report
    undeclared = suite_module.undeclared_contract_tests(cls, True)
    if undeclared:
        report.outcome = "failed"
        report.longrepr = (
            f"{cls.__name__}: skip, skipif or xfail marks require a declared cause (switched_off) for: "
            + ", ".join(undeclared)
        )
        report.result = []
    return report
