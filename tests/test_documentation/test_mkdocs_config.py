"""Tests for mkdocs.yml configuration integrity."""

from pathlib import Path
from typing import Any

import yaml


# Anchored to this file, not to the cwd. Run from anywhere but the repo root, a
# relative path here died with a bare FileNotFoundError that named the symptom
# rather than the cause (issue #937).
REPO_ROOT = Path(__file__).resolve().parents[2]
MKDOCS_YML = REPO_ROOT / "docs" / "mkdocs.yml"


def _load_mkdocs_config() -> dict[str, Any]:
    result: dict[str, Any] = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    return result


def test_no_stale_todo_comments_in_mkdocs_yml() -> None:
    """Ensure mkdocs.yml contains no stale TODO/TBD comments."""
    raw_text = MKDOCS_YML.read_text(encoding="utf-8")
    lines_with_todo = [
        (i + 1, line.rstrip())
        for i, line in enumerate(raw_text.splitlines())
        if line.strip().startswith("#") and any(marker in line.upper() for marker in ("TODO", "TBD"))
    ]
    assert not lines_with_todo, f"mkdocs.yml contains stale TODO/TBD comments: {lines_with_todo}"


def test_search_plugin_configured() -> None:
    """When plugins are explicitly listed, the search plugin must be included."""
    config = _load_mkdocs_config()
    plugins = config.get("plugins", [])

    plugin_names: list[str] = []
    for entry in plugins:
        if isinstance(entry, str):
            plugin_names.append(entry)
        elif isinstance(entry, dict):
            plugin_names.extend(entry.keys())

    assert "search" in plugin_names, (
        f"'search' plugin missing from explicit plugins list: {plugin_names}. "
        "When plugins are explicitly configured in mkdocs.yml, the default search "
        "plugin is disabled and must be re-added manually."
    )


def test_getting_started_text_guides_before_notebooks() -> None:
    """In Getting Started, .md guides should appear before .ipynb notebooks."""
    config = _load_mkdocs_config()
    nav = config.get("nav", [])

    getting_started_items = None
    for section in nav:
        if isinstance(section, dict) and "Getting Started" in section:
            getting_started_items = section["Getting Started"]
            break

    assert getting_started_items is not None, "Getting Started section not found in nav"

    paths: list[str] = []
    for item in getting_started_items:
        if isinstance(item, dict):
            for path in item.values():
                if isinstance(path, str):
                    paths.append(path)

    md_indices = [i for i, p in enumerate(paths) if p.endswith(".md")]
    ipynb_indices = [i for i, p in enumerate(paths) if p.endswith(".ipynb")]

    if md_indices and ipynb_indices:
        last_md = max(md_indices)
        first_ipynb = min(ipynb_indices)
        assert last_md < first_ipynb, (
            f"Text guides (.md) should appear before notebooks (.ipynb) in Getting Started. "
            f"Last .md at index {last_md}, first .ipynb at index {first_ipynb}. "
            f"Order: {paths}"
        )


def _collect_nav_paths(node: Any, found: set[str]) -> None:
    """Collect every path string referenced anywhere in the nav tree.

    The nav mixes three shapes at any depth: a plain string (``index.md``), a
    single-key dict whose value is a path, and a dict whose value is a nested
    list of more of the same. Recursing over all three keeps the collector
    honest as the nav grows sections.
    """
    if isinstance(node, str):
        found.add(node)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_nav_paths(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_nav_paths(item, found)


def _nav_markdown_paths() -> set[str]:
    found: set[str] = set()
    _collect_nav_paths(_load_mkdocs_config().get("nav", []), found)
    # Only .md is asserted on: the nav also points at notebooks that are
    # generated from .py sources during the docs build and are not in the tree.
    return {path for path in found if path.endswith(".md")}


def test_every_docs_page_is_reachable_from_the_nav() -> None:
    """A page not listed in nav still builds, but is invisible on the site."""
    docs_dir = MKDOCS_YML.parent / "docs"
    on_disk = {path.relative_to(docs_dir).as_posix() for path in docs_dir.rglob("*.md")}

    unreachable = sorted(on_disk - _nav_markdown_paths())

    assert not unreachable, (
        "These documentation pages exist under docs/docs/ but are not reachable "
        "from the nav: block in docs/mkdocs.yml, so they build but never appear "
        f"on the published site: {unreachable}"
    )


def test_every_nav_markdown_entry_points_at_a_real_page() -> None:
    """The other direction: a typo in nav silently drops a page from the site."""
    docs_dir = MKDOCS_YML.parent / "docs"

    missing = sorted(path for path in _nav_markdown_paths() if not (docs_dir / path).is_file())

    assert not missing, (
        f"The nav: block in docs/mkdocs.yml references markdown pages that do not exist under docs/docs/: {missing}"
    )
