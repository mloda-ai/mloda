"""Pins the split of identify_feature_group into resolution_types and resolution_failure_renderer.

Which module owns which name, that the matcher is not a facade (no __all__, no call site importing a
name it does not own from it), and the acyclic direction resolution_types <- resolution_failure_renderer
<- identify_feature_group.
"""

from __future__ import annotations

import ast
import importlib
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

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
    "FeatureResolutionError",
    "ComputeFrameworkPinError",
    "IdentifyFeatureGroupClass",
    "evaluate_and_render",
    "resolve_or_raise",
)

# Directories swept for imports of the matcher, relative to the repo root. docs/ is in because the marimo
# notebooks under docs/docs/examples/ are excluded from both mypy and ruff F401, so nothing else gates them.
SWEPT_DIRS = ("mloda", "mloda_plugins", "tests", "docs")

# Last segment of MATCHER_MODULE: what the sweep gates on, see the gate comment below.
MATCHER_BASENAME = MATCHER_MODULE.rsplit(".", 1)[-1]

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


def _foreign_matcher_imports(path: Path, source: str) -> list[str]:
    """`path:line -> name` for every name imported from the matcher that the matcher does not own."""
    tree = ast.parse(source, filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):  # walk, not tree.body: function-local imports count too
        if not isinstance(node, ast.ImportFrom) or node.module != MATCHER_MODULE:
            continue
        offenders.extend(
            f"{path}:{node.lineno} -> {alias.name}" for alias in node.names if alias.name not in MATCHER_KEPT_NAMES
        )
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

    Not the main line of defence: `mypy --strict` sweeps the same trees and its no_implicit_reexport
    rejects every static evasion (aliased imports, module-object attribute access, relative imports).
    Its one blind spot is what this test exists for: re-adding __all__ to the matcher makes those
    re-exports legal again for mypy, and the call sites route through the facade unchallenged.
    """
    root = _repo_root()
    offenders: list[str] = []
    for directory in SWEPT_DIRS:
        base = root / directory
        if not base.is_dir():  # a directory may be absent (slim checkout, sdist layout); the rest still gets swept
            continue
        for path in sorted(base.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            # Substring gate: parses 27 of 907 files, ~0.15s instead of ~1.5s. A 10x saving well inside the
            # 10s timeout either way, so drop it freely if it ever gets in the way. Gating on the bare last
            # segment rather than the dotted path keeps `from a.b . c import X` spellings in scope.
            if MATCHER_BASENAME not in source:
                continue
            offenders.extend(_foreign_matcher_imports(path, source))
    listed = "\n".join(offenders)
    assert offenders == [], (
        f"these imports take a name from {MATCHER_MODULE} that it does not own; import each from "
        f"{TYPES_MODULE} or {RENDERER_MODULE} instead:\n{listed}"
    )


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
