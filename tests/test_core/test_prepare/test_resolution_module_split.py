"""Pins the split of identify_feature_group into resolution_types and resolution_failure_renderer.

Which module owns which name, that the matcher is not a facade (no __all__, no call site importing a
name it does not own from it), and the acyclic direction resolution_types <- resolution_failure_renderer
<- identify_feature_group.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess  # nosec B404
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from mloda.core.prepare import identify_feature_group

MATCHER_MODULE = "mloda.core.prepare.identify_feature_group"
TYPES_MODULE = "mloda.core.prepare.resolution_types"
RENDERER_MODULE = "mloda.core.prepare.resolution_failure_renderer"

TYPES_NAMES = (
    "CandidateFrameworks",
    "EliminationStage",
    "Elimination",
    "RenderFacts",
    "EvaluationResult",
    "ResolutionRecord",
    "PARTIAL_RECORDS_CAP",
    "ResolutionDiagnosis",
)

RENDERER_NAMES = (
    "TROUBLESHOOTING_URL",
    "scope_callout",
    "_candidate_sort_key",
    "_supported_feature_names",
    "_prefix_name",
    "_STAGE_LABELS",
    "_render_near_miss_block",
    "_render_multiple",
    "_render_abstract_only",
    "_render_none",
    "render_resolution_failure",
)

# What stays in the matcher module, with its definition, and the only names a call site may import from it.
MATCHER_KEPT_NAMES = (
    "matches_feature_group_scope",
    "validate_single_framework_pin",
    "FeatureResolutionError",
    "ComputeFrameworkPinError",
    "IdentifyFeatureGroupClass",
    "evaluate_and_render",
    "resolve_or_raise",
)

# Directories swept for imports of the matcher, relative to the repo root. docs/ is in because the marimo
# notebooks under docs/docs/examples/ are excluded from both mypy and ruff F401, and because mktestdocs runs the
# exact ```python spelling only, leaving every other form ungated.
SWEPT_DIRS = ("mloda", "mloda_plugins", "tests", "docs")

# Last segment of MATCHER_MODULE: what the sweep gates files and markdown blocks on, see the gate comments below.
MATCHER_BASENAME = MATCHER_MODULE.rsplit(".", 1)[-1]

# Package of the matcher as directory names, for the synthetic trees the sweep tests below build under tmp_path.
MATCHER_PACKAGE_PARTS = tuple(MATCHER_MODULE.split(".")[:-1])

# Opening fence of a markdown python block in every spelling mkdocs renders as python: a space after the
# backticks, the py and python3 aliases, and any info string behind the language, such as title="x".
_PYTHON_FENCE = re.compile(r"```\s*(python3?|py)(\s|$)")

_SUBPROCESS_TIMEOUT = 8.0

_CLEAN_IMPORT_SNIPPET = (
    "import importlib\n"
    "import sys\n"
    f"importlib.import_module({RENDERER_MODULE!r})\n"
    f"print('present' if {MATCHER_MODULE!r} in sys.modules else 'absent')\n"
)


def _prepare_dir() -> Path:
    """Directory of the prepare package, read off the matcher module that already exists."""
    matcher_file = identify_feature_group.__file__
    assert matcher_file is not None
    return Path(matcher_file).parent


def _repo_root() -> Path:
    """Repo root, derived from this file's location at <root>/tests/test_core/test_prepare.

    Deliberately not off _prepare_dir(): under `tox -e installed` the matcher resolves to site-packages,
    which has no tests tree, so a module-derived root would sweep nothing and still pass. The test file
    is always in the checkout, whichever import mode is in play.
    """
    return Path(__file__).parents[3]


def _missing_names(module: ModuleType, names: Sequence[str]) -> list[str]:
    return [name for name in names if not hasattr(module, name)]


def _resolved_module(path: Path, node: ast.ImportFrom, root: Path) -> str | None:
    """Module an import names, with a relative level resolved against the package of `path`.

    None when the level walks above `root`: unresolvable, so not the matcher.
    """
    if node.level == 0:
        return node.module
    package = path.relative_to(root).parts[:-1]  # __init__.py and a plain module share the same package
    if node.level > len(package):  # the deepest level still at or below the root is the package depth
        return None
    base = package[: len(package) - node.level + 1]
    return ".".join((*base, node.module) if node.module else base)


def _foreign_names(path: Path, tree: ast.Module, root: Path, offset: int) -> list[str]:
    """`path:line -> name` per name taken from the matcher that it does not own, `offset` lines into `path`."""
    offenders: list[str] = []
    for node in ast.walk(tree):  # walk, not tree.body: function-local imports count too
        if not isinstance(node, ast.ImportFrom) or _resolved_module(path, node, root) != MATCHER_MODULE:
            continue
        offenders.extend(
            f"{path}:{node.lineno + offset} -> {alias.name}"
            for alias in node.names
            if alias.name not in MATCHER_KEPT_NAMES
        )
    return offenders


def _foreign_matcher_imports(path: Path, source: str, root: Path) -> list[str]:
    """`path:line -> name` for every name imported from the matcher that the matcher does not own."""
    return _foreign_names(path, ast.parse(source, filename=str(path)), root, 0)


def _python_blocks(source: str) -> list[tuple[int, str]]:
    """(line of the opening fence, block source) per python block, the line being the offset into the file."""
    lines = source.splitlines()
    blocks: list[tuple[int, str]] = []
    opened: int | None = None
    width = 0
    is_python = False
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        backticks = len(stripped) - len(stripped.lstrip("`"))
        if opened is None:
            opened, width, is_python = lineno, backticks, _PYTHON_FENCE.match(stripped) is not None
            continue
        if backticks < width:  # a shorter fence is content of the open block, as in a ``` shown inside a ````
            continue
        if is_python:
            blocks.append((opened, "\n".join(lines[opened : lineno - 1])))
        opened = None
    if opened is not None and is_python:  # a fence left open at the end of the file still delimits a block
        blocks.append((opened, "\n".join(lines[opened:])))
    return blocks


def _foreign_matcher_imports_in_markdown(path: Path, source: str, root: Path) -> list[str]:
    """The same check over the python blocks of a markdown file, reported at their line in the file.

    Only a block naming the matcher is parsed: the docs are full of pseudo-code and of fences indented into a
    list, neither of which parses. Once a block has named the matcher, a parse error is worth a loud failure.
    """
    offenders: list[str] = []
    for offset, block in _python_blocks(source):
        if MATCHER_BASENAME not in block:
            continue
        tree = ast.parse(textwrap.dedent(block), filename=str(path))  # dedent keeps the line count, so offset holds
        offenders.extend(_foreign_names(path, tree, root, offset))
    return offenders


def _sweep(root: Path) -> list[str]:
    """`path:line -> name` per name a file under `root` imports from the matcher that the matcher does not own."""
    paths: list[Path] = []
    for directory in SWEPT_DIRS:
        base = root / directory
        if not base.is_dir():  # a directory may be absent (slim checkout, sdist layout); the rest still gets swept
            continue
        paths.extend([*base.rglob("*.py"), *base.rglob("*.md")])
    offenders: list[str] = []
    for path in sorted(paths):
        source = path.read_text(encoding="utf-8")
        # Substring gate: a handful of the repo's files get parsed instead of all of them. It only keeps the
        # sweep cheap, which it is either way, so drop it freely if it gets in the way. Gating on the bare last
        # segment rather than the dotted path keeps `from a.b . c import X` spellings in scope.
        if MATCHER_BASENAME not in source:
            continue
        if path.suffix == ".md":
            offenders.extend(_foreign_matcher_imports_in_markdown(path, source, root))
        else:
            offenders.extend(_foreign_matcher_imports(path, source, root))
    return offenders


def _imported_modules(path: Path) -> set[str]:
    """Module names every import statement in one file references.

    Alias names are included so the `from . import sibling` form is covered too.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                modules.add(node.module)
            modules.update(alias.name for alias in node.names)
    return modules


def _references(modules: set[str], target: str) -> list[str]:
    return sorted(module for module in modules if module == target or module.endswith(f".{target}"))


def _markdown_with_block(language: str, *lines: str) -> str:
    """Markdown whose block opens on line 3: its first code line is line 4 of the file, line 1 of the block."""
    return "\n".join(["# Guide", "", f"```{language}", *lines, "```", ""])


def test_resolution_types_module_defines_its_names() -> None:
    types_module = importlib.import_module(TYPES_MODULE)
    missing = _missing_names(types_module, TYPES_NAMES)
    assert missing == [], f"{TYPES_MODULE} does not define {missing}"


def test_resolution_failure_renderer_module_defines_its_names() -> None:
    renderer = importlib.import_module(RENDERER_MODULE)
    missing = _missing_names(renderer, RENDERER_NAMES)
    assert missing == [], f"{RENDERER_MODULE} does not define {missing}"


def test_kept_names_stay_defined_in_the_matcher_module() -> None:
    for name in MATCHER_KEPT_NAMES:
        assert hasattr(identify_feature_group, name), f"{MATCHER_MODULE} no longer defines {name}"
        owner = getattr(identify_feature_group, name).__module__
        assert owner == MATCHER_MODULE, f"{name} moved out of {MATCHER_MODULE} to {owner}"


def test_matcher_declares_no_all() -> None:
    """A module always exports what it defines, so __all__ here would only re-export imports: the facade is back."""
    declared = getattr(identify_feature_group, "__all__", None)
    assert declared is None, f"{MATCHER_MODULE} declares __all__ = {declared}, re-exporting names it does not own"


def test_no_call_site_imports_a_foreign_name_from_the_matcher() -> None:
    """No call site imports a name from the matcher that the matcher does not own.

    Relative spellings of the import and the python blocks of markdown pages are in scope.

    Not the main line of defence: `mypy --strict` sweeps the same trees and its no_implicit_reexport
    rejects every static evasion (aliased imports, module-object attribute access, relative imports).
    Its one blind spot is what this test exists for: re-adding __all__ to the matcher makes those
    re-exports legal again for mypy, and the call sites route through the facade unchallenged.
    """
    offenders = _sweep(_repo_root())
    listed = "\n".join(offenders)
    assert offenders == [], (
        f"these imports take a name from {MATCHER_MODULE} that it does not own; import each from "
        f"{TYPES_MODULE} or {RENDERER_MODULE} instead:\n{listed}"
    )


def test_a_relative_import_of_a_foreign_name_is_reported(tmp_path: Path) -> None:
    """A relative spelling of the matcher import is the same offence as the dotted one."""
    path = tmp_path.joinpath(*MATCHER_PACKAGE_PARTS, "call_site.py")
    source = "from .identify_feature_group import render_resolution_failure\n"
    assert _foreign_matcher_imports(path, source, tmp_path) == [f"{path}:1 -> render_resolution_failure"]


def test_a_multi_level_relative_import_of_a_foreign_name_is_reported(tmp_path: Path) -> None:
    path = tmp_path.joinpath(*MATCHER_PACKAGE_PARTS, "nested", "call_site.py")
    source = "from ..identify_feature_group import RenderFacts\n"
    assert _foreign_matcher_imports(path, source, tmp_path) == [f"{path}:1 -> RenderFacts"]


def test_a_relative_import_of_a_kept_name_is_not_reported(tmp_path: Path) -> None:
    path = tmp_path.joinpath(*MATCHER_PACKAGE_PARTS, "call_site.py")
    source = "from .identify_feature_group import resolve_or_raise\n"
    assert _foreign_matcher_imports(path, source, tmp_path) == []


def test_a_relative_import_of_a_sibling_module_is_not_reported(tmp_path: Path) -> None:
    """Taking the name from the module that owns it is the fix, not the offence."""
    path = tmp_path.joinpath(*MATCHER_PACKAGE_PARTS, "call_site.py")
    source = "from .resolution_types import RenderFacts\n"
    assert _foreign_matcher_imports(path, source, tmp_path) == []


def test_a_relative_import_walking_above_the_root_is_not_reported(tmp_path: Path) -> None:
    """The highest level that stays at or below the root is the package depth, not one more.

    `tests.foo.bar` has a package two deep, so level 3 is `attempted relative import beyond top-level
    package` for CPython: the import never runs, so it can never take a name from the matcher.
    """
    path = tmp_path / "tests" / "foo" / "bar.py"
    source = f"from ...{MATCHER_MODULE} import RenderFacts\n"
    offenders = _foreign_matcher_imports(path, source, tmp_path)
    assert offenders == [], f"a level walking above the root resolved to the matcher: {offenders}"


def test_a_markdown_snippet_importing_a_foreign_name_is_reported(tmp_path: Path) -> None:
    """The reported line is the line in the markdown file, not the line within the block."""
    path = tmp_path / "docs" / "guide.md"
    source = _markdown_with_block("python", f"from {MATCHER_MODULE} import render_resolution_failure")
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [f"{path}:4 -> render_resolution_failure"], f"expected the markdown line, got {offenders}"


def test_a_markdown_snippet_importing_a_kept_name_is_not_reported(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "guide.md"
    source = _markdown_with_block("python", f"from {MATCHER_MODULE} import resolve_or_raise")
    assert _foreign_matcher_imports_in_markdown(path, source, tmp_path) == []


def test_a_non_python_fenced_block_is_not_scanned(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "guide.md"
    source = _markdown_with_block("text", f"from {MATCHER_MODULE} import render_resolution_failure")
    assert _foreign_matcher_imports_in_markdown(path, source, tmp_path) == []


@pytest.mark.parametrize("language", [" python", " py", " python3"])
def test_a_spaced_or_aliased_python_fence_is_scanned(tmp_path: Path, language: str) -> None:
    """mkdocs renders ``` python, ```py and ```python3 as python and mktestdocs runs none of them."""
    path = tmp_path / "docs" / "guide.md"
    source = _markdown_with_block(language, f"from {MATCHER_MODULE} import render_resolution_failure")
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [f"{path}:4 -> render_resolution_failure"], f"```{language} went unscanned: {offenders}"


@pytest.mark.parametrize("language", ['python title="x"', ' python title="x"'])
def test_a_titled_python_fence_is_scanned(tmp_path: Path, language: str) -> None:
    """The spelling the SWEPT_DIRS comment cites, in both the tight and the spaced form."""
    path = tmp_path / "docs" / "guide.md"
    source = _markdown_with_block(language, f"from {MATCHER_MODULE} import render_resolution_failure")
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [f"{path}:4 -> render_resolution_failure"], f"```{language} went unscanned: {offenders}"


@pytest.mark.parametrize("language", ["pythonic", "pytest"])
def test_a_near_miss_language_fence_is_not_scanned(tmp_path: Path, language: str) -> None:
    """Widening the fence pattern to the python aliases must not swallow a language that merely starts alike."""
    path = tmp_path / "docs" / "guide.md"
    source = _markdown_with_block(language, f"from {MATCHER_MODULE} import render_resolution_failure")
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [], f"```{language} is not python, yet it was scanned: {offenders}"


def test_an_indented_block_that_does_not_name_the_matcher_is_not_parsed(tmp_path: Path) -> None:
    """A fence indented into a numbered list is not parseable on its own, and it names no import of ours.

    Real precedent: docs/docs/in_depth/feature-group-matching.md, three-space fences inside a numbered list.
    """
    path = tmp_path / "docs" / "guide.md"
    source = "\n".join(
        [
            "# Guide",
            "",
            f"Matching lives in {MATCHER_BASENAME}.",
            "",
            "1. **Class Name Match**:",
            "   ```python",
            "   feature_name == FeatureGroup.get_class_name()",
            "   ```",
            "",
        ]
    )
    assert _foreign_matcher_imports_in_markdown(path, source, tmp_path) == []


def test_a_pseudo_code_block_that_does_not_name_the_matcher_is_not_parsed(tmp_path: Path) -> None:
    """Prose in a python fence is common in the docs; only a block naming the matcher is worth parsing."""
    path = tmp_path / "docs" / "guide.md"
    source = "\n".join(
        [
            "# Guide",
            "",
            f"Matching lives in {MATCHER_BASENAME}.",
            "",
            "```python",
            "run(features=[...], key=1, 2)",
            "```",
            "",
        ]
    )
    assert _foreign_matcher_imports_in_markdown(path, source, tmp_path) == []


def test_an_indented_block_importing_a_foreign_name_is_reported(tmp_path: Path) -> None:
    """Indentation is no exemption: a block naming the matcher is checked, at its line in the markdown file."""
    path = tmp_path / "docs" / "guide.md"
    source = "\n".join(
        [
            "# Guide",
            "",
            "1. Take the renderer from the matcher:",
            "   ```python",
            f"   from {MATCHER_MODULE} import render_resolution_failure",
            "   ```",
            "",
        ]
    )
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [f"{path}:5 -> render_resolution_failure"], f"expected the markdown line, got {offenders}"


def test_a_four_backtick_block_does_not_hide_a_later_python_block(tmp_path: Path) -> None:
    """A ```` block displaying a bare fence must not shift the pairing of every fence after it."""
    path = tmp_path / "docs" / "guide.md"
    source = "\n".join(
        [
            "# Guide",
            "",
            "````",
            "```",
            "````",
            "",
            "```python",
            f"from {MATCHER_MODULE} import render_resolution_failure",
            "```",
            "",
        ]
    )
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [f"{path}:8 -> render_resolution_failure"], f"the later block went unseen: {offenders}"


def test_prose_in_a_four_backtick_block_is_not_parsed_as_python(tmp_path: Path) -> None:
    """The other direction of the same mis-pairing: content of a ```` block is content, never a python block."""
    path = tmp_path / "docs" / "guide.md"
    source = "\n".join(
        [
            "# Guide",
            "",
            "````",
            "```",
            "```python",
            "run(features=[...], key=1, 2)",
            "```",
            "````",
            "",
        ]
    )
    assert _foreign_matcher_imports_in_markdown(path, source, tmp_path) == []


def test_an_unterminated_python_fence_at_end_of_file_is_scanned(tmp_path: Path) -> None:
    """A block left open at the end of the file is still a block: it must be flushed, not dropped."""
    path = tmp_path / "docs" / "guide.md"
    source = "\n".join(
        [
            "# Guide",
            "",
            "```python",
            f"from {MATCHER_MODULE} import render_resolution_failure",
            "",
        ]
    )
    offenders = _foreign_matcher_imports_in_markdown(path, source, tmp_path)
    assert offenders == [f"{path}:4 -> render_resolution_failure"], f"the open block was dropped: {offenders}"


def test_the_sweep_reports_a_relative_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end over a synthetic root: the relative spelling reaches the sweep's report."""
    path = tmp_path.joinpath(*MATCHER_PACKAGE_PARTS, "call_site.py")
    path.parent.mkdir(parents=True)
    path.write_text("from .identify_feature_group import render_resolution_failure\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "_repo_root", lambda: tmp_path)
    with pytest.raises(AssertionError, match="render_resolution_failure"):
        test_no_call_site_imports_a_foreign_name_from_the_matcher()


def test_the_sweep_reports_a_foreign_import_in_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep must walk *.md too: nothing else gates a python block in the docs."""
    path = tmp_path / "docs" / "guide.md"
    path.parent.mkdir(parents=True)
    source = _markdown_with_block("python", f"from {MATCHER_MODULE} import render_resolution_failure")
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "_repo_root", lambda: tmp_path)
    with pytest.raises(AssertionError, match="render_resolution_failure"):
        test_no_call_site_imports_a_foreign_name_from_the_matcher()


def test_renderer_does_not_import_the_matcher() -> None:
    """The renderer importing the matcher is the cycle that sank an earlier attempt at this split."""
    path = _prepare_dir() / "resolution_failure_renderer.py"
    assert path.is_file(), f"{path} does not exist"
    offenders = _references(_imported_modules(path), "identify_feature_group")
    assert offenders == [], f"{path.name} imports {offenders}, which closes the cycle"


def test_resolution_types_imports_neither_the_matcher_nor_the_renderer() -> None:
    path = _prepare_dir() / "resolution_types.py"
    assert path.is_file(), f"{path} does not exist"
    modules = _imported_modules(path)
    offenders = _references(modules, "identify_feature_group") + _references(modules, "resolution_failure_renderer")
    assert offenders == [], f"{path.name} imports {offenders}, but it is the base of the import order"


def test_renderer_imports_in_a_clean_interpreter_without_the_matcher() -> None:
    """Runtime counterpart of the static check: a fresh interpreter must not pull the matcher in transitively."""
    # Safe: fixed argv (sys.executable plus a literal snippet), no shell, no user input.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _CLEAN_IMPORT_SNIPPET],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, f"importing {RENDERER_MODULE} in a fresh interpreter failed:\n{result.stderr}"
    assert result.stdout.strip() == "absent", f"{MATCHER_MODULE} was imported transitively by {RENDERER_MODULE}"
