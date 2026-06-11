"""Pytest fixture helpers for isolating the process-global plugin registry."""

from __future__ import annotations

from collections.abc import Iterator

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
