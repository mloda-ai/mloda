"""Doc-taught value guards for docs/docs markdown.

The doc runner executes ```python fences, but defining a class never calls get_domain and the
docs filter matches framework names case-insensitively, so these taught values need static checks.
"""

import difflib
import re
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user import PluginLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCS_ROOT = REPO_ROOT / "docs" / "docs"

# The two doc spellings that teach a framework name: a quoted keyword and a single-line quoted list.
SINGULAR_NAME = re.compile(r'compute_framework="(?P<name>[^"]+)"')
PLURAL_LIST = re.compile(r"compute_frameworks=\[(?P<body>[^\]]*)\]")
QUOTED_NAME = re.compile(r'"([^"]+)"')

PYTHON_BLOCK = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
GET_DOMAIN_MARKER = "def get_domain"
DOMAIN_RETURN_MARKER = "return Domain("


def _doc_files() -> list[Path]:
    files = sorted(DOCS_ROOT.rglob("*.md"))
    assert files, f"no markdown files under {DOCS_ROOT}, the taught-value checks would pass vacuously"
    return files


def _taught_framework_names(text: str) -> list[tuple[int, str]]:
    """Return (1-based line number, name) for every framework name a markdown line teaches."""
    names: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for singular in SINGULAR_NAME.finditer(line):
            names.append((lineno, singular.group("name")))
        for plural in PLURAL_LIST.finditer(line):
            names.extend((lineno, name) for name in QUOTED_NAME.findall(plural.group("body")))
    return names


def _get_domain_blocks(text: str) -> list[str]:
    """Return every python/py fenced block that defines get_domain."""
    return [block for block in PYTHON_BLOCK.findall(text) if GET_DOMAIN_MARKER in block]


def _bare_get_domain_blocks(text: str) -> list[str]:
    """Return the get_domain blocks that never return a Domain instance."""
    return [block for block in _get_domain_blocks(text) if DOMAIN_RETURN_MARKER not in block]


def _loaded_framework_names() -> set[str]:
    PluginLoader.all()
    return {subclass.__name__ for subclass in get_all_subclasses(ComputeFramework)}


def _excerpt(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    shown = lines[:3]
    suffix = " ..." if len(lines) > len(shown) else ""
    return " / ".join(shown) + suffix


def test_doc_taught_compute_framework_names_are_loaded_class_names() -> None:
    """The docs filter matches case-insensitively, but framework pins by name do not."""
    loaded = _loaded_framework_names()
    assert loaded, "no ComputeFramework subclass is loaded, the name check would pass vacuously"

    violations: list[str] = []
    for path in _doc_files():
        for lineno, name in _taught_framework_names(path.read_text(encoding="utf-8")):
            if name in loaded:
                continue
            hint = difflib.get_close_matches(name, sorted(loaded)) or sorted(loaded)
            violations.append(f"{path}:{lineno}: teaches '{name}', did you mean one of {hint}?")

    assert not violations, (
        f"{len(violations)} doc-taught compute framework name(s) match no loaded class; "
        "case-sensitive surfaces reject them:\n" + "\n".join(violations)
    )


def test_doc_get_domain_examples_return_a_domain_instance() -> None:
    """Domain.__eq__ returns NotImplemented for non-Domain, so a bare-string return matches nothing."""
    violations: list[str] = []
    for path in _doc_files():
        for block in _bare_get_domain_blocks(path.read_text(encoding="utf-8")):
            violations.append(f"{path}: get_domain block without '{DOMAIN_RETURN_MARKER}': {_excerpt(block)}")

    assert not violations, (
        f"{len(violations)} doc get_domain example(s) return a value the domain gate rejects, "
        f"write '{DOMAIN_RETURN_MARKER}...)':\n" + "\n".join(violations)
    )


# Pages the scans above must keep finding something on.
FRAMEWORK_NAME_PAGES = (
    "docs/docs/in_depth/discover-plugins.md",
    "docs/docs/in_depth/mloda-api.md",
)

GET_DOMAIN_PAGES = (
    "docs/docs/in_depth/domain.md",
    "docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md",
)


class TestScannedPagesStayScannable:
    """A moved or reworded page would otherwise silently empty the scans above."""

    @pytest.mark.parametrize("relative_path", FRAMEWORK_NAME_PAGES)
    def test_the_page_still_teaches_a_framework_name(self, relative_path: str) -> None:
        page = REPO_ROOT / relative_path
        assert page.is_file(), f"{page} was moved or renamed; update FRAMEWORK_NAME_PAGES"
        names = _taught_framework_names(page.read_text(encoding="utf-8"))
        assert names, f"{page} no longer teaches a compute framework name the scanner reads"

    @pytest.mark.parametrize("relative_path", GET_DOMAIN_PAGES)
    def test_the_page_still_shows_a_get_domain_block(self, relative_path: str) -> None:
        page = REPO_ROOT / relative_path
        assert page.is_file(), f"{page} was moved or renamed; update GET_DOMAIN_PAGES"
        blocks = _get_domain_blocks(page.read_text(encoding="utf-8"))
        assert blocks, f"{page} no longer shows a get_domain block the scanner reads"


SINGULAR_MARKDOWN = 'fgs = get_feature_group_docs(compute_framework="PandasDataFrame")\n'
PLURAL_MARKDOWN = (
    'session = mloda.prepare(feature_list, compute_frameworks=["PyArrowTable"])\n'
    'result = mloda.run_all(features, compute_frameworks=["PandasDataFrame", "PyArrowTable"])\n'
)
EMPTY_LIST_MARKDOWN = "a catalog entry with `compute_frameworks=[]` stays framework-free\n"

GOOD_DOMAIN_MARKDOWN = (
    "```python\n"
    "class Good(FeatureGroup):\n"
    "    @classmethod\n"
    "    def get_domain(cls) -> Domain:\n"
    '        return Domain("example_domain")\n'
    "```\n"
)
BARE_DOMAIN_MARKDOWN = (
    "```python\n"
    "class Bare(FeatureGroup):\n"
    "    @classmethod\n"
    "    def get_domain(cls) -> Domain:\n"
    '        return "example_domain"\n'
    "```\n"
)


class TestTaughtValueExtraction:
    """The extraction helpers driven over literal markdown."""

    def test_the_singular_form_is_captured(self) -> None:
        assert _taught_framework_names(SINGULAR_MARKDOWN) == [(1, "PandasDataFrame")]

    def test_every_name_of_a_plural_list_is_captured(self) -> None:
        expected = [(1, "PyArrowTable"), (2, "PandasDataFrame"), (2, "PyArrowTable")]
        assert _taught_framework_names(PLURAL_MARKDOWN) == expected

    def test_an_empty_list_captures_nothing(self) -> None:
        assert _taught_framework_names(EMPTY_LIST_MARKDOWN) == []

    def test_a_domain_returning_block_is_not_flagged(self) -> None:
        assert _get_domain_blocks(GOOD_DOMAIN_MARKDOWN), "the good block was not even collected"
        assert _bare_get_domain_blocks(GOOD_DOMAIN_MARKDOWN) == []

    def test_a_bare_string_return_is_flagged(self) -> None:
        flagged = _bare_get_domain_blocks(BARE_DOMAIN_MARKDOWN)
        assert len(flagged) == 1, f"expected exactly the bare block flagged, got {flagged}"
        assert GET_DOMAIN_MARKER in flagged[0]
