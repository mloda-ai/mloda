"""Pins the split of the author-time guard subsystem out of feature_chain_parser.

Ownership of the moved names, the runtime match path staying put, the acyclic direction
feature_chain_parser <- feature_chain_author_guards, and the absence of re-exports back
onto FeatureChainParser.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from mloda.core.abstract_plugins.components.feature_chainer import feature_chain_parser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser

PARSER_MODULE = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"
GUARDS_MODULE = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"

# Author-time constants that move with the guards.
GUARDS_CONSTANTS = (
    "REQUIRED_WHEN_GUARD_FLAG",
    "NAME_PATH_PRESENCE_GUARD_FLAG",
    "CAPTURELESS_DIAGNOSTIC_FLAG",
    "_UNIVERSAL_MATCHER_PROBE_NAME",
    "REQUIRED_WHEN_GUARD_DEPTH",
    "NAME_PATH_PRESENCE_GUARD_DEPTH",
)

# Class-definition-time validation and guard installation, as module-level functions.
GUARDS_FUNCTIONS = (
    "validate_name_binding",
    "warn_captureless_without_binding",
    "warn_universal_optional_matcher",
    "check_required_when",
    "install_required_when_guard",
    "install_name_path_presence_guard",
    "_matcher_is_staticmethod",
    "_reject_staticmethod_matcher",
    "_resolve_match_arguments",
    "_pattern_named_and_total_groups",
    "_flatten_patterns",
    "_str_reachable_values",
)

# The public moved names, which must NOT come back as FeatureChainParser attributes.
NO_REEXPORT_NAMES = (
    "validate_name_binding",
    "warn_captureless_without_binding",
    "warn_universal_optional_matcher",
    "install_required_when_guard",
    "install_name_path_presence_guard",
    "check_required_when",
)

# The runtime match path, which stays on FeatureChainParser, including the members the guards call back into.
PARSER_KEPT_METHODS = (
    "parse_name",
    "parse_feature_name",
    "match_configuration_feature_chain_parser",
    "build_effective_options",
    "bind_name_captures",
    "prefix_patterns_of",
    "has_required_when_predicates",
    "_name_identifies_group",
    "_merge_bindings",
    "_check_name_path_required_presence",
    "_can_skip_required_check",
    "_name_path_missing_required_keys",
    "extract_property_values",
    "name_path_presence_rejection_reason",
    "extract_in_feature",
    "validate_property_mapping_defaults",
)

# Module-level names of the parser that carry a __module__ to check.
PARSER_KEPT_MODULE_LEVEL = (
    "record_match_rejection",
    "option_key_is_present",
    "PropertyValueRejection",
)

# Module-level parser constants, which have no __module__ to check.
PARSER_KEPT_CONSTANTS = (
    "MATCH_REJECTION_REASONS",
    "CHAIN_SEPARATOR",
    "COLUMN_SEPARATOR",
    "INPUT_SEPARATOR",
)

_SUBPROCESS_TIMEOUT = 8.0

# Import the parser first, record whether the guards came along, then prove the guards module exists.
_CLEAN_IMPORT_SNIPPET = (
    "import importlib\n"
    "import sys\n"
    f"importlib.import_module({PARSER_MODULE!r})\n"
    f"pulled_in = {GUARDS_MODULE!r} in sys.modules\n"
    f"importlib.import_module({GUARDS_MODULE!r})\n"
    "print('present' if pulled_in else 'absent')\n"
)


def _chainer_dir() -> Path:
    """Directory of the feature_chainer package, read off the parser module that already exists."""
    parser_file = feature_chain_parser.__file__
    assert parser_file is not None
    return Path(parser_file).parent


def _repo_root() -> Path:
    """Repo root, four levels above mloda/core/abstract_plugins/components/feature_chainer."""
    return _chainer_dir().parents[4]


def _missing_names(module: ModuleType, names: Sequence[str]) -> list[str]:
    return [name for name in names if not hasattr(module, name)]


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


def test_author_guards_module_defines_the_moved_constants() -> None:
    guards = importlib.import_module(GUARDS_MODULE)
    missing = _missing_names(guards, GUARDS_CONSTANTS)
    assert missing == [], f"{GUARDS_MODULE} does not define {missing}"


def test_author_guards_module_defines_the_moved_functions() -> None:
    guards = importlib.import_module(GUARDS_MODULE)
    missing = _missing_names(guards, GUARDS_FUNCTIONS)
    assert missing == [], f"{GUARDS_MODULE} does not define {missing}"


def test_moved_functions_are_module_level_functions() -> None:
    """The moved guards are plain functions taking the owner class, not classmethods of some new class."""
    guards = importlib.import_module(GUARDS_MODULE)
    for name in GUARDS_FUNCTIONS:
        member = getattr(guards, name)
        assert inspect.isfunction(member), f"{GUARDS_MODULE}.{name} is {type(member)!r}, not a module-level function"
        assert member.__module__ == GUARDS_MODULE, f"{name} is defined in {member.__module__}, not {GUARDS_MODULE}"


def test_moved_names_are_not_reexported_on_the_parser_class() -> None:
    """No compatibility shim: a re-export would let call sites keep the old spelling and undo the split."""
    reexported = [name for name in NO_REEXPORT_NAMES if hasattr(FeatureChainParser, name)]
    assert reexported == [], f"FeatureChainParser still exposes {reexported}, so the split is only cosmetic"


def test_runtime_match_path_stays_on_the_parser_class() -> None:
    for name in PARSER_KEPT_METHODS:
        assert hasattr(FeatureChainParser, name), f"FeatureChainParser no longer defines {name}"
        owner = getattr(FeatureChainParser, name).__module__
        assert owner == PARSER_MODULE, f"{name} moved out of {PARSER_MODULE} to {owner}"


def test_parser_module_level_names_stay_defined_in_the_parser() -> None:
    for name in PARSER_KEPT_MODULE_LEVEL:
        assert hasattr(feature_chain_parser, name), f"{PARSER_MODULE} no longer defines {name}"
        owner = getattr(feature_chain_parser, name).__module__
        assert owner == PARSER_MODULE, f"{name} moved out of {PARSER_MODULE} to {owner}"
    missing = _missing_names(feature_chain_parser, PARSER_KEPT_CONSTANTS)
    assert missing == [], f"{PARSER_MODULE} no longer defines {missing}"


def test_parser_does_not_import_the_author_guards() -> None:
    """The guards import the parser, never the reverse; that one direction is what keeps the split acyclic."""
    guards_path = _chainer_dir() / "feature_chain_author_guards.py"
    assert guards_path.is_file(), f"{guards_path} does not exist"
    parser_path = _chainer_dir() / "feature_chain_parser.py"
    offenders = _references(_imported_modules(parser_path), "feature_chain_author_guards")
    assert offenders == [], f"{parser_path.name} imports {offenders}, which closes the cycle"


def test_parser_imports_in_a_clean_interpreter_without_the_author_guards() -> None:
    """Runtime counterpart of the static check: a fresh interpreter must not pull the guards in transitively."""
    # Safe: fixed argv (sys.executable plus a literal snippet), no shell, no user input.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _CLEAN_IMPORT_SNIPPET],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        cwd=_repo_root(),
    )
    assert result.returncode == 0, f"importing {PARSER_MODULE} in a fresh interpreter failed:\n{result.stderr}"
    assert result.stdout.strip() == "absent", f"{GUARDS_MODULE} was imported transitively by {PARSER_MODULE}"
