"""The CI docs-only fast path must never skip CI for markdown that tests depend on."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CI_YAML = PROJECT_ROOT / ".github" / "workflows" / "ci.yaml"
TRIGGERS = ("push", "pull_request")

# Markdown string literals inside the test suite, e.g. "CONTRIBUTING.md" or "docs/docs/index.md".
MD_LITERAL = re.compile(r"""["']([^"'\n]*?\.md)["']""")


def _load_ci_config() -> dict[Any, Any]:
    config: dict[Any, Any] = yaml.safe_load(CI_YAML.read_text(encoding="utf-8"))
    return config


def _triggers() -> dict[str, Any]:
    # pyyaml (YAML 1.1) parses the `on:` key as the boolean True.
    config = _load_ci_config()
    triggers: dict[str, Any] = config.get("on", config.get(True, {}))
    return triggers


def _trigger_config(event: str) -> dict[str, Any]:
    config: dict[str, Any] = _triggers().get(event) or {}
    return config


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a GitHub Actions path filter glob into a regex."""
    out = ""
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**/", i):
            # `**/` matches zero directories too, so the slash is optional.
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
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
    return re.compile(out)


def _matches(pattern: str, path: str) -> bool:
    return _pattern_to_regex(pattern).fullmatch(path) is not None


def _markdown_referenced_by_tests() -> set[str]:
    """Markdown files that the test suite reads or asserts on, so CI must run when they change."""
    referenced: set[str] = set()
    for source in (PROJECT_ROOT / "tests").rglob("*.py"):
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            for literal in MD_LITERAL.findall(line):
                for candidate in (literal, f"docs/{literal}"):
                    # Throwaway tmp_path fixture names (a.md, doc.md, ...) do not exist in the repo.
                    if (PROJECT_ROOT / candidate).is_file():
                        referenced.add(candidate)
    return referenced


def _protected_paths() -> list[str]:
    docs = {str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "docs").rglob("*.md")}
    return sorted(docs | _markdown_referenced_by_tests())


# Composed, not a bare literal: MD_LITERAL scans tests/ for markdown names and treats every one that
# exists in the repo as content the suite depends on, which would wrongly mark the root README protected.
ROOT_README = "README" + ".md"

# Patterns paired with GitHub's own documented example matches (filter pattern cheat sheet).
GLOBSTAR_MATCHES: list[tuple[str, str]] = [
    ("**/README.md", ROOT_README),
    ("**/README.md", "js/README.md"),
    ("**/*.md", ROOT_README),
    ("**/*-post.md", "my-post.md"),
    ("**/*-post.md", "path/their-post.md"),
    ("**.js", "index.js"),
    ("**.js", "js/index.js"),
    ("**.js", "src/js/app.js"),
    ("*.js", "app.js"),
    ("docs/**", "docs/README.md"),
    ("docs/**", "docs/mona/octocat.txt"),
    ("docs/*", "docs/README.md"),
    ("docs/**/*.md", "docs/README.md"),
    ("docs/**/*.md", "docs/a/markdown/file.md"),
    ("**/docs/**", "docs/hello.md"),
    ("**/docs/**", "dir/docs/my-file.txt"),
]

# A single `*` stops at a path separator, so the helper must not over-match either.
GLOBSTAR_NON_MATCHES: list[tuple[str, str]] = [
    ("*.js", "js/app.js"),
    ("docs/*", "docs/mona/octocat.txt"),
    ("**/README.md", "README.md.bak"),
]


def _case_ids(cases: list[tuple[str, str]]) -> list[str]:
    return [f"{pattern} ~ {path}" for pattern, path in cases]


@pytest.mark.parametrize(("pattern", "path"), GLOBSTAR_MATCHES, ids=_case_ids(GLOBSTAR_MATCHES))
def test_pattern_matches_github_documented_example(pattern: str, path: str) -> None:
    assert _matches(pattern, path), (
        f"GitHub documents {path!r} as a match for {pattern!r}, but the helper produced "
        f"{_pattern_to_regex(pattern).pattern!r}. Under-matching lets the guards pass while GitHub skips the gate."
    )


@pytest.mark.parametrize(("pattern", "path"), GLOBSTAR_NON_MATCHES, ids=_case_ids(GLOBSTAR_NON_MATCHES))
def test_pattern_does_not_match_beyond_github_semantics(pattern: str, path: str) -> None:
    assert not _matches(pattern, path), (
        f"{pattern!r} must not match {path!r}, but the helper produced {_pattern_to_regex(pattern).pattern!r}. "
        "Over-matching makes the guards fail on filters that are actually safe."
    )


def test_protected_set_is_not_stale() -> None:
    protected = _protected_paths()
    assert "docs/docs/index.md" in protected, "docs/ markdown is executed by tests/test_documentation"
    assert "CONTRIBUTING.md" in protected, "CONTRIBUTING.md content is asserted by tests/test_project_structure.py"


def test_paths_ignore_declared_for_push_and_pull_request() -> None:
    for event in TRIGGERS:
        config = _trigger_config(event)
        assert isinstance(config.get("paths-ignore"), list), (
            f"CI trigger '{event}' must declare a 'paths-ignore' list for the docs-only fast path. Got: {config}"
        )


def test_paths_allowlist_is_absent() -> None:
    for event in TRIGGERS:
        assert "paths" not in _trigger_config(event), (
            f"CI trigger '{event}' must not use a 'paths' allowlist: it would silently drop coverage "
            "for any file not listed, which this guard cannot check. Use 'paths-ignore' only."
        )


def test_paths_ignore_has_no_negation_patterns() -> None:
    for event in TRIGGERS:
        for pattern in _trigger_config(event).get("paths-ignore", []):
            assert not str(pattern).startswith("!"), (
                f"CI trigger '{event}' paths-ignore pattern {pattern!r} uses negation, which this guard "
                "cannot reason about. Keep patterns positive."
            )


def test_paths_ignore_never_covers_test_relevant_markdown() -> None:
    protected = _protected_paths()
    for event in TRIGGERS:
        for pattern in _trigger_config(event).get("paths-ignore", []):
            covered = [path for path in protected if _matches(str(pattern), path)]
            assert not covered, (
                f"CI trigger '{event}' paths-ignore pattern {pattern!r} would skip CI for {covered[:3]}. "
                "These markdown files are executed or asserted on by the test suite, so a change to them "
                "must run the gate. Narrow the filter to markdown no test depends on."
            )
