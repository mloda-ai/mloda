"""Fence vocabulary guard for docs/docs markdown."""

import re
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.test_documentation.test_documentation import run_md_file_isolated

DOCS_ROOT = Path("docs/docs")

RUNNABLE_TAG = "python"
ILLUSTRATIVE_TAG = "py"
OUTPUT_TAG = "text"

PYTHON_LOOKING = re.compile(r"py(thon)?[0-9]*|pycon", re.IGNORECASE)

VOCABULARY_HINT = (
    f"Use ```{RUNNABLE_TAG} when the block runs, ```{ILLUSTRATIVE_TAG} when it is an "
    f"illustrative fragment that must not run, ```{OUTPUT_TAG} when it is output."
)

# Repo-relative doc path -> permitted number of ```py blocks. An absent file permits zero.
ILLUSTRATIVE_BLOCK_ALLOWLIST: dict[str, int] = {
    "docs/docs/chapter1/api-request.md": 1,
    "docs/docs/chapter1/compute-frameworks.md": 4,
    "docs/docs/in_depth/access-feature-data.md": 5,
    "docs/docs/in_depth/artifacts.md": 1,
    "docs/docs/in_depth/compute-framework-integration.md": 12,
    "docs/docs/in_depth/data-access-patterns.md": 7,
    "docs/docs/in_depth/data-quality.md": 2,
    "docs/docs/in_depth/data-type-enforcement.md": 1,
    "docs/docs/in_depth/discover-plugins.md": 1,
    "docs/docs/in_depth/feature-chain-parser.md": 16,
    "docs/docs/in_depth/feature-config.md": 2,
    "docs/docs/in_depth/feature-group-matching.md": 3,
    "docs/docs/in_depth/feature-group-testing.md": 5,
    "docs/docs/in_depth/filter_data.md": 1,
    "docs/docs/in_depth/framework-connection-object.md": 9,
    "docs/docs/in_depth/framework-transformers.md": 4,
    "docs/docs/in_depth/join_data.md": 4,
    "docs/docs/in_depth/mloda-api.md": 5,
    "docs/docs/in_depth/multiple_result_columns.md": 5,
    "docs/docs/in_depth/named-data-access-handles.md": 4,
    "docs/docs/in_depth/plugin-loader.md": 1,
    "docs/docs/in_depth/plugin_registry.md": 1,
    "docs/docs/in_depth/property-mapping.md": 4,
    "docs/docs/in_depth/streaming.md": 6,
    "docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md": 3,
}

MAX_REPORTED_VIOLATIONS = 60


class Fence:
    """One opening code fence in a markdown file."""

    def __init__(self, path: Path, lineno: int, text: str, info: str) -> None:
        self.path = path
        self.lineno = lineno
        self.text = text
        self.info = info

    @property
    def tag(self) -> str:
        parts = self.info.strip().split()
        return parts[0] if parts else ""

    @property
    def has_leading_space(self) -> bool:
        return bool(self.info) and self.info[0].isspace() and bool(self.info.strip())

    @property
    def location(self) -> str:
        return f"{self.path}:{self.lineno}"


def _iter_fence_openings(path: Path) -> Iterator[Fence]:
    """Yield the opening fences of a markdown file, skipping fence bodies."""
    inside = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        backticks = len(stripped) - len(stripped.lstrip("`"))
        info = stripped[backticks:]
        if inside:
            if not info.strip():
                inside = False
            continue
        inside = True
        yield Fence(path=path, lineno=lineno, text=stripped, info=info)


def _doc_files() -> list[Path]:
    return sorted(DOCS_ROOT.rglob("*.md"))


def _suggested_tag(tag: str) -> str:
    if tag.lower() == "pycon":
        return OUTPUT_TAG
    return RUNNABLE_TAG


def _fence_violations() -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in _doc_files():
        for fence in _iter_fence_openings(path):
            tag = fence.tag
            if fence.has_leading_space:
                unspaced = f"```{fence.info.strip()}"
                violations.append(
                    (
                        path,
                        fence.lineno,
                        f"{fence.location}: spaced fence {fence.text!r} -> write {unspaced!r}. {VOCABULARY_HINT}",
                    )
                )
                continue
            if not PYTHON_LOOKING.fullmatch(tag):
                continue
            if fence.info.strip() == tag and tag in {RUNNABLE_TAG, ILLUSTRATIVE_TAG}:
                continue
            if fence.info.strip() != tag:
                remedy = (
                    f"attributes after the tag also stop collection -> write a bare ```{RUNNABLE_TAG} "
                    f"with the title in a comment, or ```{ILLUSTRATIVE_TAG} if it must not run"
                )
            else:
                remedy = f"write ```{_suggested_tag(tag)}"
            violations.append(
                (
                    path,
                    fence.lineno,
                    f"{fence.location}: fence {fence.text!r} looks like Python but is never executed, "
                    f"{remedy}. {VOCABULARY_HINT}",
                )
            )
    return sorted(violations, key=lambda item: (str(item[0]), item[1]))


def _format(violations: list[tuple[Path, int, str]]) -> str:
    shown = violations[:MAX_REPORTED_VIOLATIONS]
    lines = [message for _, _, message in shown]
    omitted = len(violations) - len(shown)
    if omitted:
        lines.append(f"... and {omitted} more omitted.")
    return "\n".join(lines)


def test_doc_fences_use_the_documented_vocabulary() -> None:
    violations = _fence_violations()
    assert not violations, f"{len(violations)} doc fence violation(s):\n{_format(violations)}"


def test_illustrative_py_blocks_match_allowlist() -> None:
    errors: list[str] = []
    for path in _doc_files():
        actual = sum(1 for fence in _iter_fence_openings(path) if fence.info.strip() == ILLUSTRATIVE_TAG)
        allowed = ILLUSTRATIVE_BLOCK_ALLOWLIST.get(str(path), 0)
        if actual > allowed:
            errors.append(
                f"{path}: {actual} ```{ILLUSTRATIVE_TAG} block(s) but only {allowed} allowed. "
                f"Promote the block to ```{RUNNABLE_TAG} (or ```{OUTPUT_TAG} if it is output), "
                f"or raise its ILLUSTRATIVE_BLOCK_ALLOWLIST entry."
            )
        elif actual < allowed:
            errors.append(
                f"{path}: {actual} ```{ILLUSTRATIVE_TAG} block(s) but {allowed} allowed. "
                f"Lower the ILLUSTRATIVE_BLOCK_ALLOWLIST entry to {actual}."
            )
    assert not errors, "```py allowlist drift:\n" + "\n".join(errors)


FAILING_SNIPPET = 'raise RuntimeError("doc fence collection contract")'

RUNNABLE_MD = textwrap.dedent(
    f"""\
    # Runnable

    ```{RUNNABLE_TAG}
    {FAILING_SNIPPET}
    ```
    """
)

ILLUSTRATIVE_MD = textwrap.dedent(
    f"""\
    # Illustrative

    ```{ILLUSTRATIVE_TAG}
    {FAILING_SNIPPET}
    ```
    """
)


@pytest.mark.timeout(120)
def test_collection_contract_of_python_and_py_fences(tmp_path: Path) -> None:
    """```python is executed by the doc runner, ```py is not."""
    runnable = tmp_path / "runnable.md"
    runnable.write_text(RUNNABLE_MD, encoding="utf-8")
    with pytest.raises(AssertionError):
        run_md_file_isolated(runnable)

    illustrative = tmp_path / "illustrative.md"
    illustrative.write_text(ILLUSTRATIVE_MD, encoding="utf-8")
    run_md_file_isolated(illustrative)
