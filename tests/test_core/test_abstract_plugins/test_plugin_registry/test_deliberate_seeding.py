"""Regression tests for deliberate plugin-registry seeding (issue #583, part 2)."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOC_TEST_FILE = _REPO_ROOT / "tests" / "test_documentation" / "test_documentation.py"


def test_documentation_has_no_import_time_plugin_load() -> None:
    """No module-level PluginLoader.all() may seed the registry at import time (issue #583)."""
    source = _DOC_TEST_FILE.read_text()
    module_level_load_lines = [
        line
        for line in source.splitlines()
        # Unindented call only; indented (in-function) calls are legitimate.
        if line[:1] not in (" ", "\t") and line.split("#", 1)[0].rstrip() == "PluginLoader.all()"
    ]
    assert module_level_load_lines == [], (
        "Found a module-level 'PluginLoader.all()' call in "
        f"{_DOC_TEST_FILE}; it seeds PluginRegistry.default() at import time. "
        "Seeding must happen in a deliberate session-scoped fixture instead."
    )
