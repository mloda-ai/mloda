"""Tests for mkdocs.yml configuration integrity."""

from pathlib import Path
from typing import Any

import yaml


MKDOCS_YML = Path("docs/mkdocs.yml")


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
