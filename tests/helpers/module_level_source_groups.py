"""Guard against module-level source FeatureGroups that can claim other modules' sources (#2598).

A FeatureGroup subclass defined at module level in a test module stays registered for the whole
worker process, so it can match credentials or files that other test modules use and make their
resolution ambiguous. The registry-pollution fixture in tests/conftest.py only reports subclasses
created inside a test; module-level classes predate every snapshot.

A module-level source group is safe when it narrows what it can claim: overriding at least one of
the matching hooks below, relative to its production base class, counts as narrowed (for example
SecretsRecordingSqliteFG gates is_valid_credentials on a marker key). tests/conftest.py fails
collection when a collected test module defines one that does not narrow.
"""

from __future__ import annotations

import inspect
from collections.abc import Collection

from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup

NARROWING_HOOKS = ("is_valid_credentials", "match_feature_group_criteria", "find_sources", "match_connection")


def _walk_subclasses(cls: type) -> list[type]:
    found: list[type] = []
    for subclass in cls.__subclasses__():
        found.append(subclass)
        found.extend(_walk_subclasses(subclass))
    return found


def _production_base(cls: type, module_names: Collection[str]) -> type | None:
    """The nearest ancestor not defined in a test module: the base the hooks are compared against."""
    for ancestor in cls.__mro__[1:]:
        module = ancestor.__module__
        if module not in module_names and not module.startswith(("tests.", "test_")):
            return ancestor
    return None


def _hook_owner(cls: type, hook: str) -> type | None:
    """The first class in the MRO defining ``hook`` (raw class-dict lookup, so an inherited
    classmethod compares equal to itself instead of to a freshly bound method)."""
    for klass in cls.__mro__:
        if hook in klass.__dict__:
            return klass
    return None


def module_level_source_group_offenders(module_names: Collection[str]) -> list[str]:
    """Sorted ``module::Class`` names of unnarrowed module-level source groups in ``module_names``.

    Only concrete subclasses of SourceFeatureGroup defined in one of the given (collected test)
    modules are considered; classes created while tests run are the registry-pollution fixture's
    business, not this guard's.
    """
    offenders: list[str] = []
    for cls in _walk_subclasses(SourceFeatureGroup):
        if cls.__module__ not in module_names or inspect.isabstract(cls):
            continue
        base = _production_base(cls, module_names)
        if base is None:
            continue
        if all(_hook_owner(cls, hook) is _hook_owner(base, hook) for hook in NARROWING_HOOKS):
            offenders.append(f"{cls.__module__}::{cls.__name__}")
    return sorted(offenders)
