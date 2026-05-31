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
    """PluginLoader.all() must succeed even when pyarrow is unavailable.

    Expected RED failure: subprocess crashes with ModuleNotFoundError because
    at least one mloda_plugins module imports pyarrow at the top level.
    """
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
    """_load_plugin must raise ModuleNotFoundError for a genuinely non-existent module.

    This test does NOT block pyarrow and must pass today (characterization/guard test).
    """
    from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

    loader = PluginLoader()
    with pytest.raises(ModuleNotFoundError):
        loader._load_plugin("compute_framework.this_module_does_not_exist_xyz")
