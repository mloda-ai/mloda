from pathlib import Path
import re
import pytest

from mktestdocs import check_md_file


from typing import Set, Any
import time
from mloda.steward import ExtenderHook, Extender
from mloda.user import PluginLoader
import logging

logger = logging.getLogger(__name__)

# Load all plugins before running documentation tests
PluginLoader.all()


# We need this to test DokuExtender
class DokuExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result


class DokuValidateInputFeatureExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken: {time.time() - start}")
        return result


@pytest.mark.parametrize("fpath", Path("docs").glob("**/*.md"), ids=str)
def test_files_good(fpath: Any) -> None:
    check_md_file(fpath=fpath, memory=True)


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
