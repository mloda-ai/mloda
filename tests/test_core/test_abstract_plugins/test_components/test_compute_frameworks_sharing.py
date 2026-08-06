"""Rot guard: ``Feature.compute_frameworks`` is rebound, never mutated in place.

The set is shared by reference and feeds ``__eq__``/``__hash__``, so an in-place write shifts the hash of
every other holder. Blind spot: a local alias (``cfs = feature.compute_frameworks``) is invisible to this
source scan, and a same-named attribute on a non-Feature class would be a false positive.
"""

from __future__ import annotations

import ast
from copy import copy
from pathlib import Path

import mloda
import mloda_plugins
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.filter.single_filter import SingleFilter
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

ATTRIBUTE = "compute_frameworks"

MUTATING_METHODS = frozenset(
    {
        "add",
        "clear",
        "difference_update",
        "discard",
        "intersection_update",
        "pop",
        "remove",
        "symmetric_difference_update",
        "update",
    }
)

AUG_OP_SYMBOLS: dict[type[ast.operator], str] = {
    ast.BitOr: "|=",
    ast.BitAnd: "&=",
    ast.Sub: "-=",
    ast.BitXor: "^=",
}

# Namespace package: the directories come from __path__, and an editable install repeats one, hence the dedup.
SCAN_ROOTS = sorted(
    {Path(entry).resolve() for entry in (*mloda.__path__, *mloda_plugins.__path__) if Path(entry).is_dir()}
)

_HINT = (
    "Rebind instead: feature.compute_frameworks = {...}. The set is shared by reference and feeds "
    "Feature.__eq__/__hash__. If the flagged holder is not a Feature, teach the guard to skip it."
)

EVERY_IN_PLACE_FORM = """
def rot(feature, framework):
    feature.compute_frameworks.add(framework)
    feature.compute_frameworks.update({framework})
    feature.compute_frameworks.discard(framework)
    feature.compute_frameworks.remove(framework)
    feature.compute_frameworks.clear()
    feature.compute_frameworks.pop()
    feature.compute_frameworks.difference_update({framework})
    feature.compute_frameworks.intersection_update({framework})
    feature.compute_frameworks.symmetric_difference_update({framework})
    feature.compute_frameworks |= {framework}
    feature.compute_frameworks &= {framework}
    feature.compute_frameworks -= {framework}
    feature.compute_frameworks ^= {framework}
"""

LEGAL_FORMS = """
def legal(feature, other, framework):
    feature.compute_frameworks = {framework}
    feature.compute_frameworks = None
    other.compute_frameworks = feature.compute_frameworks
    pinned = frozenset(feature.compute_frameworks)
    if feature.compute_frameworks:
        for entry in feature.compute_frameworks:
            pinned |= {entry}
    feature.other_set.add(framework)
    feature.other_set |= {framework}
    return pinned
"""

CFS_FEATURE = "cfs_shared_feature"
CFS_FILTER_FEATURE = "cfs_filter_feature"


def _is_target_attribute(node: ast.expr) -> bool:
    """Any ``<expr>.compute_frameworks``, whatever the base expression is."""
    return isinstance(node, ast.Attribute) and node.attr == ATTRIBUTE


def _mutating_method(node: ast.Call) -> str | None:
    """The set method the call mutates the attribute with, None for anything else."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in MUTATING_METHODS:
        return None
    return func.attr if _is_target_attribute(func.value) else None


def _violations_in_source(source: str, filename: str) -> list[str]:
    """In-place writes to a ``<expr>.compute_frameworks`` in one module's source."""
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            method = _mutating_method(node)
            if method is not None:
                found.append(f"{filename}:{node.lineno}: .{ATTRIBUTE}.{method}(...)")
        elif isinstance(node, ast.AugAssign) and _is_target_attribute(node.target):
            symbol = AUG_OP_SYMBOLS.get(type(node.op), f"{type(node.op).__name__}=")
            found.append(f"{filename}:{node.lineno}: .{ATTRIBUTE} {symbol} ...")
    return found


def _violations(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        found.extend(_violations_in_source(path.read_text(encoding="utf-8"), str(path)))
    return found


def _forms(violations: list[str]) -> list[str]:
    """The write form each violation names, without its file:line."""
    return [violation.rsplit(": ", 1)[1] for violation in violations]


def _configured_forms() -> set[str]:
    return {f".{ATTRIBUTE}.{name}(...)" for name in MUTATING_METHODS} | {
        f".{ATTRIBUTE} {symbol} ..." for symbol in AUG_OP_SYMBOLS.values()
    }


def test_no_module_mutates_compute_frameworks_in_place() -> None:
    """Scope: every module under mloda/ and mloda_plugins/."""
    # Vacuity floor, NOT a target: a sweep over half the tree would pass this guard trivially.
    assert {root.name for root in SCAN_ROOTS} >= {"mloda", "mloda_plugins"}, f"roots: {SCAN_ROOTS}"

    violations: list[str] = []
    for root in SCAN_ROOTS:
        violations.extend(_violations(root))

    assert violations == [], "compute_frameworks is written in place:\n" + "\n".join(violations) + "\n" + _HINT


def test_the_sweep_reaches_both_trees() -> None:
    """A sweep over no files would pass trivially."""
    scanned = [path for root in SCAN_ROOTS for path in root.rglob("*.py")]
    # Vacuity floor, NOT a target: well under the measured total (288 today) so deleting a module stays legitimate.
    assert len(scanned) > 100, f"only {len(scanned)} files scanned"


def test_guard_flags_every_in_place_write_form() -> None:
    """Positive control."""
    assert sorted(_forms(_violations_in_source(EVERY_IN_PLACE_FORM, "rot.py"))) == sorted(
        [
            ".compute_frameworks &= ...",
            ".compute_frameworks -= ...",
            ".compute_frameworks ^= ...",
            ".compute_frameworks |= ...",
            ".compute_frameworks.add(...)",
            ".compute_frameworks.clear(...)",
            ".compute_frameworks.difference_update(...)",
            ".compute_frameworks.discard(...)",
            ".compute_frameworks.intersection_update(...)",
            ".compute_frameworks.pop(...)",
            ".compute_frameworks.remove(...)",
            ".compute_frameworks.symmetric_difference_update(...)",
            ".compute_frameworks.update(...)",
        ]
    )


def test_the_positive_control_covers_every_configured_form() -> None:
    assert set(_forms(_violations_in_source(EVERY_IN_PLACE_FORM, "rot.py"))) == _configured_forms()


def test_guard_reports_the_file_and_line_of_a_violation() -> None:
    source = "x = 1\nfeature.compute_frameworks.add(x)\n"
    assert _violations_in_source(source, "some/module.py") == ["some/module.py:2: .compute_frameworks.add(...)"]


def test_guard_flags_a_mutation_through_a_nested_base_expression() -> None:
    """Only the attribute the expression ends in decides."""
    source = "def rot(f, filters):\n    filters[0].filter_feature.compute_frameworks.add(f)\n"
    assert _forms(_violations_in_source(source, "nested.py")) == [".compute_frameworks.add(...)"]


def test_guard_allows_rebinding_reading_and_unrelated_attributes() -> None:
    """Negative control."""
    assert _violations_in_source(LEGAL_FORMS, "legal.py") == []


def _feature_with_framework(name: str = CFS_FEATURE) -> Feature:
    return Feature(name, compute_framework=PythonDictFramework.get_class_name())


def test_copy_owns_its_compute_frameworks_set() -> None:
    """Value-equal to the original, never the same object."""
    feature = _feature_with_framework()
    assert feature.compute_frameworks == {PythonDictFramework}
    duplicate = copy(feature)
    assert duplicate.compute_frameworks == feature.compute_frameworks
    assert duplicate.compute_frameworks is not feature.compute_frameworks


def test_copy_of_a_feature_without_frameworks_keeps_none() -> None:
    """Characterization."""
    feature = Feature(CFS_FEATURE)
    assert feature.compute_frameworks is None
    assert copy(feature).compute_frameworks is None


def test_global_filter_adoption_copies_the_features_set_never_aliases_it() -> None:
    """Ownership: a later write into the adopted set can never reach the host's."""
    feat = _feature_with_framework()
    single_filter = SingleFilter(Feature(CFS_FILTER_FEATURE), FilterType.EQUAL, {"value": 1})
    assert single_filter.filter_feature.compute_frameworks is None

    assert GlobalFilter().compute_framework(single_filter, feat) is True
    assert single_filter.filter_feature.compute_frameworks == feat.compute_frameworks
    assert single_filter.filter_feature.compute_frameworks is not feat.compute_frameworks


def test_two_independently_built_features_own_separate_sets() -> None:
    """Characterization: sharing only ever comes from a copy or an alias."""
    first = _feature_with_framework()
    second = _feature_with_framework()
    assert first.compute_frameworks == second.compute_frameworks
    assert first.compute_frameworks is not second.compute_frameworks


def _aliased_pair() -> tuple[Feature, Feature]:
    """Two features sharing one set object, the sanctioned aliasing form."""
    feature = _feature_with_framework()
    alias = _feature_with_framework(CFS_FILTER_FEATURE)
    alias.compute_frameworks = feature.compute_frameworks
    return feature, alias


def test_an_in_place_write_through_an_alias_loses_a_feature_from_a_set() -> None:
    """Why the guard exists."""
    feature, alias = _aliased_pair()
    holder = {feature}
    assert feature in holder

    assert alias.compute_frameworks is not None
    alias.compute_frameworks.add(SqliteFramework)

    assert feature not in holder, "the in-place write must be what loses the feature, not the aliasing"
    assert feature.compute_frameworks == {PythonDictFramework, SqliteFramework}


def test_a_rebind_through_an_alias_leaves_the_other_holder_findable() -> None:
    """The sanctioned form, from the same angle."""
    feature, alias = _aliased_pair()
    holder = {feature}

    alias.compute_frameworks = {SqliteFramework}

    assert feature in holder
    assert feature.compute_frameworks == {PythonDictFramework}
