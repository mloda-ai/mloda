"""Pins the split of identify_feature_group into resolution_types and resolution_failure_renderer.

Ownership, re-export identity for the public names the matcher keeps, and the acyclic direction
resolution_types <- resolution_failure_renderer <- identify_feature_group.
"""

from __future__ import annotations

import ast
import importlib
import subprocess  # nosec B404
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType

from mloda.core.prepare import identify_feature_group

MATCHER_MODULE = "mloda.core.prepare.identify_feature_group"
TYPES_MODULE = "mloda.core.prepare.resolution_types"
RENDERER_MODULE = "mloda.core.prepare.resolution_failure_renderer"

# Policy: the whole block stays a matcher re-export; docs/docs/in_depth/mloda-api.md:201 documents it as importable.
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

# render_resolution_failure is the public rendering entry point paired with evaluate_and_render, which returns its
# output, so call sites may legitimately take it from the matcher.
REEXPORTED_RENDERER_NAMES = ("render_resolution_failure",)

# Rendering-only: nothing reaches these through the matcher, so it must not import or re-export them.
RENDERER_ONLY_NAMES = (
    "TROUBLESHOOTING_URL",
    "scope_callout",
)

# What stays in the matcher module, with its definition, not as a re-export.
MATCHER_KEPT_NAMES = (
    "matches_feature_group_scope",
    "FeatureResolutionError",
    "ComputeFrameworkPinError",
    "IdentifyFeatureGroupClass",
    "evaluate_and_render",
    "resolve_or_raise",
)

# Every name a call site imports from the matcher module today; the split must keep all of them importable.
CALL_SITE_NAMES = (
    "CandidateFrameworks",
    "ComputeFrameworkPinError",
    "Elimination",
    "EvaluationResult",
    "FeatureResolutionError",
    "IdentifyFeatureGroupClass",
    "PARTIAL_RECORDS_CAP",
    "RenderFacts",
    "ResolutionDiagnosis",
    "ResolutionRecord",
    "evaluate_and_render",
    "matches_feature_group_scope",
    "render_resolution_failure",
    "resolve_or_raise",
)

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


def _missing_names(module: ModuleType, names: Sequence[str]) -> list[str]:
    return [name for name in names if not hasattr(module, name)]


def _public(names: Iterable[str]) -> list[str]:
    return [name for name in names if not name.startswith("_")]


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
    classified = set(REEXPORTED_RENDERER_NAMES) | set(RENDERER_ONLY_NAMES)
    assert set(_public(RENDERER_NAMES)) == classified, (
        "a new public renderer name must be classified into one bucket or the other, REEXPORTED_RENDERER_NAMES or "
        f"RENDERER_ONLY_NAMES, else nothing pins it: {sorted(set(_public(RENDERER_NAMES)) ^ classified)}"
    )


def test_public_moved_names_are_reexported_by_identity() -> None:
    """The matcher must re-export the same objects, not copies, so isinstance and `is` checks keep working."""
    for module_name, names in ((TYPES_MODULE, TYPES_NAMES), (RENDERER_MODULE, REEXPORTED_RENDERER_NAMES)):
        owner = importlib.import_module(module_name)
        for name in _public(names):
            assert hasattr(identify_feature_group, name), f"{MATCHER_MODULE} no longer re-exports {name}"
            assert getattr(identify_feature_group, name) is getattr(owner, name), (
                f"{MATCHER_MODULE}.{name} is not the same object as {module_name}.{name}"
            )


def test_rendering_only_names_are_not_reexported_by_the_matcher() -> None:
    """Rendering-only: the matcher must neither import nor re-export these, so no consumer can route through it."""
    renderer = importlib.import_module(RENDERER_MODULE)
    for name in RENDERER_ONLY_NAMES:
        assert hasattr(renderer, name), f"{RENDERER_MODULE} no longer owns {name}"
        assert name not in identify_feature_group.__all__, f"{MATCHER_MODULE}.__all__ still re-exports {name}"
        assert not hasattr(identify_feature_group, name), f"{MATCHER_MODULE} still imports {name}"


def test_kept_names_stay_defined_in_the_matcher_module() -> None:
    for name in MATCHER_KEPT_NAMES:
        assert hasattr(identify_feature_group, name), f"{MATCHER_MODULE} no longer defines {name}"
        owner = getattr(identify_feature_group, name).__module__
        assert owner == MATCHER_MODULE, f"{name} moved out of {MATCHER_MODULE} to {owner}"


def test_call_site_names_remain_importable_from_the_matcher() -> None:
    matcher = importlib.import_module(MATCHER_MODULE)
    missing = _missing_names(matcher, CALL_SITE_NAMES)
    assert missing == [], f"{MATCHER_MODULE} no longer exposes {missing}, so its call sites break"
    # hasattr alone is blind to __all__, but mypy --strict implies --no-implicit-reexport: the re-export
    # surface it checks against is __all__, so a name missing there breaks type checking at every call site
    # even while the runtime attribute is still there.
    unexported = [name for name in CALL_SITE_NAMES if name not in matcher.__all__]
    assert unexported == [], (
        f"{MATCHER_MODULE}.__all__ omits {unexported}; mypy --strict's no-implicit-reexport checks against "
        f"__all__, so its call sites fail to type-check even though the attributes still exist"
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
