"""Doc-taught value guards for docs/docs markdown.

The doc runner executes ```python fences, but defining a class never calls get_domain, the docs
filter matches framework names case-insensitively, and a fenced identifier is never resolved.
"""

import difflib
import re

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure
from mloda.core.prepare.resolution_types import EvaluationResult, RenderFacts
from mloda.user import PluginLoader

from tests.docs_corpus import REPO_ROOT, doc_files

# The doc spellings that teach a framework name: quoted keyword, quoted dict key, and quoted list/set bodies.
SINGULAR_NAME = re.compile(r"""compute_framework\s*=\s*(?P<quote>["'])(?P<name>[^"']+)(?P=quote)""")
DICT_KEY_NAME = re.compile(
    r"""(?P<key_quote>["'])compute_framework(?P=key_quote)\s*:\s*(?P<quote>["'])(?P<name>[^"']+)(?P=quote)"""
)
PLURAL_LIST = re.compile(r"compute_frameworks\s*=\s*\[(?P<body>[^\]]*)\]", re.DOTALL)
PLURAL_SET = re.compile(r"compute_frameworks\s*=\s*\{(?P<body>[^}]*)\}", re.DOTALL)
QUOTED_NAME = re.compile(r"""(?P<quote>["'])(?P<name>[^"']+)(?P=quote)""")

# These fence-block regexes rely on tests/test_docs_fences.py banning the fence spellings (tilde,
# four-backtick, spaced, attributed) they would mishandle.
PYTHON_BLOCK = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
# Any fenced block, whatever its info string: quoted error output teaches framework class names too.
FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
FRAMEWORK_LIKE_IDENTIFIER = re.compile(r"\b[A-Z]\w*(?:DataFrame|Table|Framework)\b")

# Framework-like fenced identifiers that are legitimately not loaded classes: the abstract base plus the
# placeholder classes of the transformer and type-enforcement examples.
FRAMEWORK_IDENTIFIER_ALLOWLIST = frozenset({"ComputeFramework", "CustomFramework", "MyFramework", "OtherFramework"})

GET_DOMAIN_MARKER = "def get_domain"
DOMAIN_RETURN_MARKER = "return Domain("


def _line_of(text: str, offset: int) -> int:
    """1-based line number of a character offset."""
    return text[:offset].count("\n") + 1


def _taught_framework_names(text: str) -> list[tuple[int, str]]:
    """Return (1-based line number, name) for every framework name the markdown teaches, in text order."""
    found: list[tuple[int, str]] = []
    for pattern in (SINGULAR_NAME, DICT_KEY_NAME):
        for match in pattern.finditer(text):
            found.append((match.start("name"), match.group("name")))
    for pattern in (PLURAL_LIST, PLURAL_SET):
        for match in pattern.finditer(text):
            body_start = match.start("body")
            for quoted in QUOTED_NAME.finditer(match.group("body")):
                found.append((body_start + quoted.start("name"), quoted.group("name")))
    return [(_line_of(text, offset), name) for offset, name in sorted(found)]


def _framework_like_identifiers(text: str) -> list[tuple[int, str]]:
    """Return (1-based line number, identifier) for every framework-like class name inside a fenced block."""
    found: list[tuple[int, str]] = []
    for block in FENCED_BLOCK.finditer(text):
        for match in FRAMEWORK_LIKE_IDENTIFIER.finditer(block.group(1)):
            found.append((_line_of(text, block.start(1) + match.start()), match.group(0)))
    return found


def _get_domain_blocks(text: str) -> list[str]:
    """Return every python/py fenced block that defines get_domain."""
    return [block for block in PYTHON_BLOCK.findall(text) if GET_DOMAIN_MARKER in block]


def _get_domain_bodies(block: str) -> list[str]:
    """Slice each get_domain body: the lines after the def that are blank or indented deeper than it."""
    lines = block.splitlines()
    bodies: list[str] = []
    for index, line in enumerate(lines):
        if GET_DOMAIN_MARKER not in line:
            continue
        def_indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for later in lines[index + 1 :]:
            if later.strip() and len(later) - len(later.lstrip()) <= def_indent:
                break
            body.append(later)
        bodies.append("\n".join(body))
    return bodies


def _bad_get_domain_bodies(text: str) -> list[str]:
    """Return each get_domain body returning a bare string literal, or neither returning nor building a Domain."""
    bad: list[str] = []
    for block in _get_domain_blocks(text):
        for body in _get_domain_bodies(block):
            if 'return "' in body or "return '" in body:
                bad.append(body)
            elif "Domain(" not in body and "return" not in body:
                bad.append(body)
    return bad


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
    for path in doc_files():
        for lineno, name in _taught_framework_names(path.read_text(encoding="utf-8")):
            if name in loaded:
                continue
            hint = difflib.get_close_matches(name, sorted(loaded)) or sorted(loaded)
            violations.append(f"{path}:{lineno}: teaches '{name}', did you mean one of {hint}?")

    assert not violations, (
        f"{len(violations)} doc-taught compute framework name(s) match no loaded class; "
        "case-sensitive surfaces reject them:\n" + "\n".join(violations)
    )


def test_doc_fenced_framework_identifiers_are_loaded_class_names() -> None:
    """A fenced example naming a framework class that does not exist teaches an unusable spelling."""
    loaded = _loaded_framework_names()
    assert loaded, "no ComputeFramework subclass is loaded, the identifier check would pass vacuously"
    known = loaded | FRAMEWORK_IDENTIFIER_ALLOWLIST

    violations: list[str] = []
    for path in doc_files():
        for lineno, name in _framework_like_identifiers(path.read_text(encoding="utf-8")):
            if name in known:
                continue
            hint = difflib.get_close_matches(name, sorted(loaded)) or sorted(loaded)
            violations.append(f"{path}:{lineno}: names '{name}', did you mean one of {hint}?")

    assert not violations, (
        f"{len(violations)} fenced framework-like identifier(s) match no loaded class and no allowlist entry:\n"
        + "\n".join(violations)
    )


def test_the_identifier_allowlist_names_no_loaded_class() -> None:
    """An allowlist entry shadowing a real class would exempt that class from the identifier guard."""
    assert not FRAMEWORK_IDENTIFIER_ALLOWLIST & _loaded_framework_names()


def test_doc_get_domain_examples_return_a_domain_instance() -> None:
    """Domain.__eq__ returns NotImplemented for non-Domain, so a bare-string return matches nothing."""
    violations: list[str] = []
    for path in doc_files():
        for body in _bad_get_domain_bodies(path.read_text(encoding="utf-8")):
            violations.append(f"{path}: get_domain body without a Domain return: {_excerpt(body)}")

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
SINGLE_QUOTED_MARKDOWN = "fgs = get_feature_group_docs(compute_framework='PandasDataFrame')\n"
SPACED_EQUALS_MARKDOWN = 'fgs = get_feature_group_docs(compute_framework = "PyArrowTable")\n'
DICT_KEY_MARKDOWN = (
    'feature = Feature("id", options={"compute_framework": "PyArrowTable"})\n'
    "feature = Feature('id', options={'compute_framework': 'PandasDataFrame'})\n"
)
SET_FORM_MARKDOWN = 'entry = RegistryEntry(compute_frameworks={"PyArrowTable"})\n'
CLASS_OBJECT_SET_MARKDOWN = "entry = RegistryEntry(compute_frameworks={PyArrowTable, PandasDataFrame})\n"
PLURAL_MARKDOWN = (
    'session = mloda.prepare(feature_list, compute_frameworks=["PyArrowTable"])\n'
    'result = mloda.run_all(features, compute_frameworks=["PandasDataFrame", "PyArrowTable"])\n'
)
MULTILINE_LIST_MARKDOWN = (
    "result = mloda.run_all(\n"
    "    features,\n"
    "    compute_frameworks = [\n"
    '        "PandasDataFrame",\n'
    "        'PyArrowTable',\n"
    "    ],\n"
    ")\n"
)
EMPTY_LIST_MARKDOWN = "a catalog entry with `compute_frameworks=[]` stays framework-free\n"

IDENTIFIER_MARKDOWN = (
    "PandasDataFrame in prose is never captured.\n"
    "```python\n"
    "if compute_framework is SQLiteFramework:\n"
    "    return pd.DataFrame()\n"
    "```\n"
    "```\n"
    "pinned compute framework 'SparkFramework' is not among its supported\n"
    "```\n"
)

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
MIXED_DOMAIN_MARKDOWN = (
    "```python\n"
    "class Good(FeatureGroup):\n"
    "    @classmethod\n"
    "    def get_domain(cls) -> Domain:\n"
    '        return Domain("example_domain")\n'
    "\n"
    "class Bare(FeatureGroup):\n"
    "    @classmethod\n"
    "    def get_domain(cls) -> Domain:\n"
    '        return "example_domain"\n'
    "```\n"
)
INDIRECT_DOMAIN_MARKDOWN = (
    "```python\n"
    'SALES = Domain("sales")\n'
    "\n"
    "class Indirect(FeatureGroup):\n"
    "    @classmethod\n"
    "    def get_domain(cls) -> Domain:\n"
    "        return SALES\n"
    "```\n"
)
RETURNLESS_DOMAIN_MARKDOWN = (
    "```python\nclass Stub(FeatureGroup):\n    @classmethod\n    def get_domain(cls) -> Domain:\n        ...\n```\n"
)


class TestTaughtValueExtraction:
    """The extraction helpers driven over literal markdown."""

    def test_the_singular_form_is_captured(self) -> None:
        assert _taught_framework_names(SINGULAR_MARKDOWN) == [(1, "PandasDataFrame")]

    def test_a_single_quoted_singular_form_is_captured(self) -> None:
        assert _taught_framework_names(SINGLE_QUOTED_MARKDOWN) == [(1, "PandasDataFrame")]

    def test_spaces_around_the_equals_sign_are_accepted(self) -> None:
        assert _taught_framework_names(SPACED_EQUALS_MARKDOWN) == [(1, "PyArrowTable")]

    def test_the_dict_key_form_is_captured_with_either_quote(self) -> None:
        assert _taught_framework_names(DICT_KEY_MARKDOWN) == [(1, "PyArrowTable"), (2, "PandasDataFrame")]

    def test_a_quoted_set_form_is_captured(self) -> None:
        assert _taught_framework_names(SET_FORM_MARKDOWN) == [(1, "PyArrowTable")]

    def test_an_unquoted_class_object_set_is_ignored(self) -> None:
        assert _taught_framework_names(CLASS_OBJECT_SET_MARKDOWN) == []

    def test_every_name_of_a_plural_list_is_captured(self) -> None:
        expected = [(1, "PyArrowTable"), (2, "PandasDataFrame"), (2, "PyArrowTable")]
        assert _taught_framework_names(PLURAL_MARKDOWN) == expected

    def test_a_multi_line_list_is_captured_with_per_name_line_numbers(self) -> None:
        assert _taught_framework_names(MULTILINE_LIST_MARKDOWN) == [(4, "PandasDataFrame"), (5, "PyArrowTable")]

    def test_an_empty_list_captures_nothing(self) -> None:
        assert _taught_framework_names(EMPTY_LIST_MARKDOWN) == []

    def test_framework_like_identifiers_are_captured_inside_any_fence(self) -> None:
        """Prose (line 1) and attribute access (line 4) contribute nothing; both fences are read."""
        assert _framework_like_identifiers(IDENTIFIER_MARKDOWN) == [(3, "SQLiteFramework"), (7, "SparkFramework")]

    def test_a_domain_returning_block_is_not_flagged(self) -> None:
        assert _get_domain_blocks(GOOD_DOMAIN_MARKDOWN), "the good block was not even collected"
        assert _bad_get_domain_bodies(GOOD_DOMAIN_MARKDOWN) == []

    def test_a_bare_string_return_is_flagged(self) -> None:
        flagged = _bad_get_domain_bodies(BARE_DOMAIN_MARKDOWN)
        assert len(flagged) == 1, f"expected exactly the bare body flagged, got {flagged}"
        assert 'return "example_domain"' in flagged[0]

    def test_a_fence_mixing_good_and_bad_definitions_flags_only_the_bad_body(self) -> None:
        flagged = _bad_get_domain_bodies(MIXED_DOMAIN_MARKDOWN)
        assert len(flagged) == 1, f"expected exactly the bare body flagged, got {flagged}"
        assert 'return "example_domain"' in flagged[0]

    def test_a_body_returning_a_fence_bound_domain_is_not_flagged(self) -> None:
        assert _get_domain_blocks(INDIRECT_DOMAIN_MARKDOWN), "the indirect block was not even collected"
        assert _bad_get_domain_bodies(INDIRECT_DOMAIN_MARKDOWN) == []

    def test_a_returnless_body_is_flagged(self) -> None:
        flagged = _bad_get_domain_bodies(RETURNLESS_DOMAIN_MARKDOWN)
        assert len(flagged) == 1, f"expected exactly the returnless body flagged, got {flagged}"


TROUBLESHOOTING_PAGE = "docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md"
ABSTRACT_ONLY_HEADING = "### Only abstract feature group bases matched"


def _abstract_only_doc_blocks() -> tuple[str, str]:
    """Return the two messages fenced after the abstract-only heading: the bare variant, then the frameworks one."""
    page = REPO_ROOT / TROUBLESHOOTING_PAGE
    assert page.is_file(), f"{page} was moved or renamed; update TROUBLESHOOTING_PAGE"
    text = page.read_text(encoding="utf-8")
    assert ABSTRACT_ONLY_HEADING in text, f"{page} lost the heading '{ABSTRACT_ONLY_HEADING}'"
    blocks = FENCED_BLOCK.findall(text[text.index(ABSTRACT_ONLY_HEADING) :])
    assert len(blocks) >= 2, f"{page} no longer fences two messages after '{ABSTRACT_ONLY_HEADING}'"
    return blocks[0].rstrip("\n"), blocks[1].rstrip("\n")


class TestTroubleshootingAbstractOnlyBlocksAreLiveRendererOutput:
    """The two quoted abstract-only messages must stay byte-identical to what the renderer emits."""

    def test_the_bare_variant_is_the_rendered_message(self) -> None:
        result = EvaluationResult(identified={}, abstract_matched={FeatureGroup})
        assert result.failure_kind == "abstract_only"
        bare_block, _ = _abstract_only_doc_blocks()
        assert render_resolution_failure(result, Feature("my_feature")) == bare_block

    def test_the_frameworks_variant_is_the_rendered_message(self) -> None:
        result = EvaluationResult(
            identified={},
            abstract_matched={FeatureGroup},
            facts=RenderFacts(concrete_frameworks=("PandasDataFrame",)),
        )
        assert result.failure_kind == "abstract_only"
        _, frameworks_block = _abstract_only_doc_blocks()
        assert render_resolution_failure(result, Feature("my_feature")) == frameworks_block
