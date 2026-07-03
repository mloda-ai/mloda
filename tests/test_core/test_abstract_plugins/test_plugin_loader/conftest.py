from collections.abc import Iterator

import pytest

from mloda.user import PluginLoader


@pytest.fixture(autouse=True)
def reset_plugin_loader_cache() -> Iterator[None]:
    """Reset the PluginLoader cache around every test so cached loader state never leaks between tests."""
    PluginLoader.reset_cache()
    yield
    PluginLoader.reset_cache()
