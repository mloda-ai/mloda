"""Regression tests for deliberate plugin-registry seeding (issue #583, part 2)."""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOC_TEST_FILE = _REPO_ROOT / "tests" / "test_documentation" / "test_documentation.py"


def _is_plugin_loader_all_call(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if not isinstance(func, ast.Attribute) or func.attr != "all":
        return False
    target = func.value
    if isinstance(target, ast.Name) and target.id == "PluginLoader":
        return True
    # PluginLoader().all() spelling.
    return isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id == "PluginLoader"


def _module_level_plugin_loader_all_lines(source: str) -> list[int]:
    """Line numbers of module-level PluginLoader.all() / PluginLoader().all() expression statements."""
    tree = ast.parse(source)
    return [node.lineno for node in tree.body if isinstance(node, ast.Expr) and _is_plugin_loader_all_call(node.value)]


def test_documentation_has_no_import_time_plugin_load() -> None:
    """No module-level PluginLoader.all() may seed the registry at import time (issue #583)."""
    module_level_load_lines = _module_level_plugin_loader_all_lines(_DOC_TEST_FILE.read_text())
    assert module_level_load_lines == [], (
        "Found a module-level 'PluginLoader.all()' call in "
        f"{_DOC_TEST_FILE} at lines {module_level_load_lines}; it seeds PluginRegistry.default() at import time. "
        "Seeding must happen in a deliberate session-scoped fixture instead."
    )


def test_helper_detects_module_level_call() -> None:
    source = "from mloda.user import PluginLoader\nPluginLoader.all()\nPluginLoader().all()\n"
    assert _module_level_plugin_loader_all_lines(source) == [2, 3]


def test_helper_ignores_string_constants_and_function_bodies() -> None:
    source = "DRIVER = '''\nPluginLoader.all()\n'''\n\ndef seed() -> None:\n    PluginLoader.all()\n"
    assert _module_level_plugin_loader_all_lines(source) == []
