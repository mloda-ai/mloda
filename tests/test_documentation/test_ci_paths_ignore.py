"""The CI docs-only fast path must never skip CI for files under docs/."""

import re
from pathlib import Path
from typing import Any

import yaml


CI_YAML = Path(".github/workflows/ci.yaml")

# docs/ content is executed by tests (mktestdocs) or asserted on, so it must always trigger CI.
PROTECTED_DOCS_PATHS = [
    "docs/mkdocs.yml",
    "docs/docs/in_depth/join_data.md",
    "docs/docs/index.md",
    "docs/docs/chapter1/installation.md",
]


def _load_ci_config() -> dict[Any, Any]:
    config: dict[Any, Any] = yaml.safe_load(CI_YAML.read_text(encoding="utf-8"))
    return config


def _triggers() -> dict[str, Any]:
    # pyyaml (YAML 1.1) parses the `on:` key as the boolean True.
    config = _load_ci_config()
    triggers: dict[str, Any] = config.get("on", config.get(True, {}))
    return triggers


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a GitHub Actions path filter glob into a regex."""
    out = ""
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif char == "*":
            out += "[^/]*"
            i += 1
        elif char == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(char)
            i += 1
    return re.compile(f"^{out}$")


def _matches(pattern: str, path: str) -> bool:
    return _pattern_to_regex(pattern).search(path) is not None


def test_paths_ignore_declared_for_push_and_pull_request() -> None:
    triggers = _triggers()
    for event in ("push", "pull_request"):
        config = triggers.get(event) or {}
        assert isinstance(config.get("paths-ignore"), list), (
            f"CI trigger '{event}' must declare a 'paths-ignore' list for the docs-only fast path. Got: {config}"
        )


def test_paths_ignore_never_covers_docs_directory() -> None:
    docs_paths = sorted(str(p) for p in Path("docs").rglob("*.md"))
    samples = PROTECTED_DOCS_PATHS + docs_paths
    assert Path("docs/docs/in_depth/join_data.md").exists(), "sample docs path is stale"

    for event in ("push", "pull_request"):
        for pattern in (_triggers().get(event) or {}).get("paths-ignore", []):
            covered = [path for path in samples if _matches(str(pattern), path)]
            assert not covered, (
                f"CI trigger '{event}' paths-ignore pattern {pattern!r} would skip CI for {covered[:3]}. "
                "Files under docs/ are executed by tests/test_documentation and must always trigger CI. "
                "Keep the filter root-level only, e.g. '*.md'."
            )
