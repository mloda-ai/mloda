import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.user import PluginLoader


@pytest.fixture(autouse=True)
def reset_plugin_loader_cache() -> Iterator[None]:
    """Reset the PluginLoader cache around every test so cached loader state never leaks between tests."""
    PluginLoader.reset_cache()
    yield
    PluginLoader.reset_cache()


def _write_broken_optional_root_package(base_dir: Path, pkg_name: str, missing_subdep: str) -> None:
    """Build an installed-but-incomplete package: its __init__.py imports a nonexistent module."""
    pkg_dir = base_dir / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(f"import {missing_subdep}\n")
    importlib.invalidate_caches()
