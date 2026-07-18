from collections.abc import Sequence
from pathlib import Path
import re
import subprocess  # nosec B404
import sys

import pytest


from typing import Any
import time
from mloda.steward import ExtenderHook, Extender
import logging

logger = logging.getLogger(__name__)


# We need this to test DokuExtender
class DokuExtender(Extender):
    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result


class DokuValidateInputFeatureExtender(Extender):
    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken: {time.time() - start}")
        return result


_DOC_CHECK_DRIVER = """\
import sys
from pathlib import Path
from mloda.user import PluginLoader
from mktestdocs import check_md_file
PluginLoader.all()
for path in sys.argv[1:]:
    print(f"CHECKING {path}", flush=True)
    check_md_file(fpath=Path(path), memory=True)
"""


def run_md_files_isolated(fpaths: Sequence[Path]) -> None:
    """Run the python snippets of markdown files in one fresh interpreter.

    In-process execution lets FeatureGroup subclasses leaked by other tests
    pollute doc-snippet feature resolution (issue #828). One seeded child for
    the whole doc set keeps resolution isolated from that leakage while paying
    the plugin-load cost once; docs sharing one interpreter matches the
    pre-isolation semantics. The last CHECKING line attributes a failure.
    """
    kept_paths = [fpath for fpath in fpaths if "```python" in fpath.read_text(encoding="utf-8")]
    if not kept_paths:
        return
    # Safe: fixed argv (sys.executable, constant driver, file paths), no shell, no user input.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _DOC_CHECK_DRIVER, *map(str, kept_paths)],
        capture_output=True,
        text=True,
        timeout=110,
    )
    stdout_tail = "\n".join(result.stdout.splitlines()[-50:])
    assert result.returncode == 0, (
        f"Doc snippet check failed (last CHECKING line names the file)\n"
        f"stdout (last 50 lines):\n{stdout_tail}\nstderr:\n{result.stderr}"
    )


def run_md_file_isolated(fpath: Path) -> None:
    run_md_files_isolated([fpath])


@pytest.mark.timeout(120)
def test_files_good() -> None:
    run_md_files_isolated(sorted(Path("docs").glob("**/*.md")))


CODE_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)
TEST_IMPORT_PATTERN = re.compile(r"^\s*from\s+tests\.", re.MULTILINE)


@pytest.mark.parametrize("fpath", sorted(Path("docs/docs").rglob("*.md")), ids=str)
def test_no_test_imports_in_docs(fpath: Path) -> None:
    text = fpath.read_text(encoding="utf-8")
    violations = []
    for block in CODE_BLOCK_PATTERN.finditer(text):
        for match in TEST_IMPORT_PATTERN.finditer(block.group(1)):
            violations.append(match.group().strip())
    assert not violations, f"{fpath} imports from test modules (not available to users): {violations}"


DOCS_ROOT = Path("docs/docs")
ABSOLUTE_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(https://mloda-ai\.github\.io/mloda/[^)]*\)")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _heading_to_anchor(heading: str) -> str:
    anchor = heading.strip().lower()
    anchor = re.sub(r"[^\w\s-]", "", anchor)
    anchor = re.sub(r"\s+", "-", anchor)
    return anchor


def _extract_headings(md_path: Path) -> set[str]:
    text = md_path.read_text(encoding="utf-8")
    return {_heading_to_anchor(m.group(1)) for m in HEADING_PATTERN.finditer(text)}


@pytest.mark.parametrize("fpath", sorted(DOCS_ROOT.rglob("*.md")), ids=str)
def test_no_absolute_site_links(fpath: Path) -> None:
    text = fpath.read_text(encoding="utf-8")
    matches = ABSOLUTE_LINK_PATTERN.findall(text)
    assert not matches, f"{fpath} contains absolute mloda-ai.github.io links (should be relative): {matches}"


@pytest.mark.parametrize("fpath", sorted(DOCS_ROOT.rglob("*.md")), ids=str)
def test_internal_link_targets_exist(fpath: Path) -> None:
    text = fpath.read_text(encoding="utf-8")
    errors: list[str] = []

    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        link_text, target = match.group(1), match.group(2)

        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue

        path_part, _, anchor = target.partition("#")

        if path_part:
            resolved = (fpath.parent / path_part).resolve()
            if not resolved.exists():
                errors.append(f"[{link_text}]({target}) -> file not found: {resolved}")
                continue

            if anchor:
                headings = _extract_headings(resolved)
                if anchor not in headings:
                    errors.append(f"[{link_text}]({target}) -> anchor #{anchor} not found in {resolved.name}")

    assert not errors, f"{fpath} has broken internal links:\n" + "\n".join(errors)


# Temporarily disabled - README being refactored
# def test_readme() -> None:
#     """
#     Test all Python code blocks in README.md.
#
#     This test uses mktestdocs to extract and execute all Python code blocks
#     from the README, ensuring all examples are correct and runnable.
#
#     Note: README examples use DataCreator (in-memory data generation),
#     so no external files are needed.
#     """
#     readme_path = Path("README.md")
#
#     if not readme_path.exists():
#         pytest.skip("README.md not found in repository root")
#
#     # Run mktestdocs on README with memory mode to avoid file pollution
#     check_md_file(fpath=readme_path, memory=True)
