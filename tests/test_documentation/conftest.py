"""Docs-scoped, deliberate plugin-registry seeding (issue #583)."""

import pytest

from mloda.user import PluginLoader


@pytest.fixture(scope="session", autouse=True)
def seed_registry_for_docs() -> None:
    """Seed PluginRegistry.default() before the documentation tests run (replaces the old import-time call)."""
    PluginLoader.all()
