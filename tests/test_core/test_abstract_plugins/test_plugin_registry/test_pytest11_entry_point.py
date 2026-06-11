"""Tests that the plugin-registry fixtures module ships as a pytest11 entry point."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_PYPROJECT = Path(__file__).parents[4] / "pyproject.toml"
_PLUGIN_MODULE = "mloda.core.abstract_plugins.plugin_registry.fixtures"


def test_pyproject_declares_pytest11_entry_point() -> None:
    """pyproject.toml declares a pytest11 entry point named mloda pointing at the fixtures module."""
    with open(_PYPROJECT, "rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    entry_points = data["project"].get("entry-points", {})
    assert "pytest11" in entry_points, f"No pytest11 entry-point group in pyproject.toml. Got: {list(entry_points)}"
    assert entry_points["pytest11"].get("mloda") == _PLUGIN_MODULE


def test_installed_distribution_exposes_pytest11_entry_point() -> None:
    """The installed mloda distribution exposes the pytest11 entry point in its metadata."""
    eps = metadata.entry_points(group="pytest11", name="mloda")
    assert eps, "Installed mloda distribution exposes no pytest11 entry point named mloda"
    (ep,) = eps
    assert ep.value == _PLUGIN_MODULE


def test_fixtures_module_provides_pytest_fixture() -> None:
    """The fixtures module defines isolated_plugin_registry as a pytest fixture."""
    module = __import__(_PLUGIN_MODULE, fromlist=["isolated_plugin_registry"])
    fixture = module.isolated_plugin_registry
    is_fixture = hasattr(fixture, "_fixture_function_marker") or hasattr(fixture, "_pytestfixturefunction")
    assert is_fixture, "isolated_plugin_registry is not wrapped by pytest.fixture"
