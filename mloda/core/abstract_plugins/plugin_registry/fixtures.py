"""Pytest fixture helpers for isolating the process-global plugin registry."""

from __future__ import annotations

import sys
from collections.abc import Generator, Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
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
    if suite_module is None or report.outcome != "passed" or not isinstance(collector, pytest.Class):
        return report
    cls = collector.obj
    if not is_nominal_subclass(cls, suite_module.ConformanceSuite) or cls._suite_base:
        return report
    marked = suite_module.has_skip_mark_names({m.name for m in collector.iter_markers()})
    undeclared = set(suite_module.marked_contract_tests(cls))
    if marked:
        undeclared.update(suite_module.undeclared_contract_tests(cls, True))
    if undeclared:
        sources = [
            node.name
            for node in collector.listchain()
            if suite_module.has_skip_mark_names(m.name for m in node.own_markers)
        ]
        where = f" (marks on: {', '.join(sources)})" if sources else ""
        report.outcome = "failed"
        report.longrepr = (
            f"{cls.__name__}: skip, skipif or xfail marks require a declared cause (switched_off) for: "
            + ", ".join(sorted(undeclared))
            + where
        )
        report.result = []
    return report
