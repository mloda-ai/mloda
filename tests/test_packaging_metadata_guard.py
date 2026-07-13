"""Files that pyproject packaging metadata points at must be guarded by an always-on workflow.

ci.yaml's docs-only fast path skips CI for markdown in paths-ignore. README.md is in that list, but
pyproject's `readme` entry points at it, so a PR that deletes or renames it skips CI, merges green,
and breaks the sdist/wheel build for everyone afterwards. An always-on workflow (no paths filters)
must assert those files still exist.
"""

import re
from pathlib import Path
from typing import Any

import yaml

from tests.test_ci_paths_ignore import PROJECT_ROOT, TRIGGERS, _matches, _trigger_config


WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# `readme = { file = "README.md", ... }` / `license = { file = "LICENSE" }` / `readme = "README.md"`.
METADATA_FILE = re.compile(r"""^\s*(?:readme|license)\s*=\s*(?:\{[^}]*?\bfile\s*=\s*)?["']([^"']+)["']""", re.MULTILINE)
README_KEY = re.compile(r"^\s*readme\s*=", re.MULTILINE)


def _packaging_metadata_files() -> set[str]:
    return set(METADATA_FILE.findall(PYPROJECT.read_text(encoding="utf-8")))


def _workflow_triggers(config: dict[Any, Any]) -> dict[str, Any]:
    # pyyaml (YAML 1.1) parses the `on:` key as the boolean True.
    triggers: dict[str, Any] = config.get("on", config.get(True, {})) or {}
    return triggers


def _always_on_workflows() -> list[Path]:
    """Workflows that run on every push and pull_request, unfiltered by path."""
    always_on: list[Path] = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        config: dict[Any, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        triggers = _workflow_triggers(config)
        if not all(event in triggers for event in TRIGGERS):
            continue
        if any((triggers.get(event) or {}).get(key) for event in TRIGGERS for key in ("paths", "paths-ignore")):
            continue
        always_on.append(path)
    return always_on


# No bare markdown filename literal appears anywhere in this module on purpose. The sibling guard
# (test_ci_paths_ignore.py) scans tests/ for bare .md string literals and treats each one as markdown
# whose *content* the suite depends on, forbidding ci.yaml from putting it in paths-ignore. This module
# only asserts such files *exist*, so hardcoding a name here would wrongly protect it from the fast path.
def test_pyproject_metadata_files_are_discovered() -> None:
    metadata_files = _packaging_metadata_files()
    assert metadata_files, "pyproject packaging metadata parse found no files: this guard has gone stale"
    for target in sorted(metadata_files):
        assert (PROJECT_ROOT / target).is_file(), f"pyproject packaging metadata points at missing file {target!r}"
    assert README_KEY.search(PYPROJECT.read_text(encoding="utf-8")), (
        "pyproject no longer declares a `readme` key: the parse target of this guard has disappeared"
    )


def test_metadata_files_in_paths_ignore_are_guarded_by_an_always_on_workflow() -> None:
    metadata_files = _packaging_metadata_files()
    guards = {path: path.read_text(encoding="utf-8") for path in _always_on_workflows()}
    for event in TRIGGERS:
        for pattern in _trigger_config(event).get("paths-ignore", []):
            for target in sorted(metadata_files):
                if not _matches(str(pattern), target):
                    continue
                assert any(target in body for body in guards.values()), (
                    f"{target!r} is referenced by pyproject packaging metadata but CI trigger '{event}' "
                    f"paths-ignore pattern {pattern!r} skips the gate when it changes. Deleting or renaming "
                    f"it would merge green and break the sdist/wheel build. Add an always-on workflow in "
                    f".github/workflows/ (push + pull_request, no paths/paths-ignore filters) that asserts "
                    f"{target!r} exists. Always-on workflows found: {sorted(p.name for p in guards)}"
                )
