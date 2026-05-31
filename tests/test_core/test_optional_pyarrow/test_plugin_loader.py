"""Tests for PluginLoader behavior when pyarrow is absent.

test_plugin_loader_all_without_pyarrow  — subprocess (pyarrow blocked): PluginLoader.all()
    must complete without crashing; non-pyarrow plugins load; pyarrow CFW plugins are skipped.
test_load_plugin_reraises_for_missing_mloda_module — no subprocess: _load_plugin raises
    ModuleNotFoundError for a genuinely non-existent module (not an optional dep).
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY_ALL: str = """
import sys

try:
    from mloda.user import PluginLoader
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__ + ":" + str(e))
    sys.exit(1)

try:
    pl = PluginLoader.all()
except Exception as e:
    print("ALL_FAILED:" + type(e).__name__ + ":" + str(e))
    sys.exit(1)

modules = pl.list_loaded_modules()
joined = ",".join(modules)
print("LOADED:" + joined)
"""


@pytest.mark.timeout(30)
def test_plugin_loader_all_without_pyarrow() -> None:
    """PluginLoader.all() must succeed even when pyarrow is unavailable."""
    result = run_blocked(_BODY_ALL)

    assert result.returncode == 0, (
        "PluginLoader.all() crashed under blocked pyarrow.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    # Find the LOADED: sentinel line.
    loaded_line = next((ln for ln in result.stdout.splitlines() if ln.startswith("LOADED:")), None)
    assert loaded_line is not None, f"Expected LOADED: sentinel in stdout. Got:\n{result.stdout}"

    loaded_modules = loaded_line[len("LOADED:") :].split(",")

    # At least one python_dict plugin must have loaded.
    assert any("python_dict" in m for m in loaded_modules), (
        f"Expected a python_dict plugin in loaded modules.\nModules: {loaded_modules}"
    )

    # No pyarrow compute-framework implementation must have loaded.
    assert not any("base_implementations.pyarrow" in m for m in loaded_modules), (
        f"pyarrow CFW modules must be skipped under blocked pyarrow.\nModules: {loaded_modules}"
    )


def test_load_plugin_reraises_for_missing_mloda_module() -> None:
    """_load_plugin must raise ModuleNotFoundError for a genuinely non-existent module."""
    from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

    loader = PluginLoader()
    with pytest.raises(ModuleNotFoundError):
        loader._load_plugin("compute_framework.this_module_does_not_exist_xyz")


def test_load_plugin_reraises_for_unknown_external_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_plugin must re-raise ModuleNotFoundError for an unknown (non-optional) external module."""
    from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

    def fake_import_module(name: str) -> object:
        raise ModuleNotFoundError(
            "No module named 'totally_missing_dependency_xyz'",
            name="totally_missing_dependency_xyz",
        )

    monkeypatch.setattr(
        "mloda.core.abstract_plugins.plugin_loader.plugin_loader.importlib.import_module",
        fake_import_module,
    )

    loader = PluginLoader()
    with pytest.raises(ModuleNotFoundError):
        loader._load_plugin("compute_framework.some_unloaded_plugin_xyz")


def test_load_plugin_reraises_for_nameless_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_plugin must re-raise a ModuleNotFoundError that was raised without a name= keyword (e.name is None)."""
    from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

    def fake_import_module(name: str) -> object:
        raise ModuleNotFoundError("totally_missing_dependency")

    monkeypatch.setattr(
        "mloda.core.abstract_plugins.plugin_loader.plugin_loader.importlib.import_module",
        fake_import_module,
    )

    loader = PluginLoader()
    with pytest.raises(ModuleNotFoundError):
        loader._load_plugin("compute_framework.some_unloaded_plugin_nameless_xyz")
