"""Shape check for an operation group: its Implementations declare consistent supported and divergent subtypes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import Implementation, current_implementations, serving_implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup

Probe = Callable[[str], tuple[str, Options]]


def _implementations(group: type[OperationFeatureGroup]) -> list[type[Implementation[Any]]]:
    current = list(current_implementations())
    found: list[type[Implementation[Any]]] = []
    for framework in group.compute_framework_definition():
        impl = serving_implementation(group, framework, current)
        if impl is not None and impl not in found:
            found.append(impl)
    return found


def _declaration_failures(
    impl: type[Implementation[Any]], universe: frozenset[str], alias_pairs: Sequence[tuple[str, str]]
) -> list[str]:
    name = impl.__name__
    failures: list[str] = []
    if impl.supported is not None:
        extra = sorted(impl.supported - universe)
        if extra:
            failures.append(f"supported: {name} lists {extra} outside the subtype universe")
    for subtype, reason in impl.divergent.items():
        if subtype not in universe:
            failures.append(f"divergent: {name} marks '{subtype}' outside the subtype universe")
        if not isinstance(reason, str) or not reason.strip():
            failures.append(f"divergent: {name} gives '{subtype}' no reason")
    for first, second in alias_pairs:
        missing = [s for s in (first, second) if s not in universe]
        if missing:
            failures.append(f"aliases: ({first}, {second}) names {missing} outside the subtype universe")
        elif (first in impl.divergent) != (second in impl.divergent):
            failures.append(f"aliases: {name} diverges for only one of '{first}' and '{second}'")
    return failures


def _shape_a_probe(group: type[OperationFeatureGroup]) -> Probe | None:
    declaration = group.SUBTYPES
    key = None if declaration is None else declaration.key
    if key is None:
        return None
    return lambda subtype: (f"conformance_{subtype}", Options(group={key: subtype}))


def _decline_failures(
    group: type[OperationFeatureGroup],
    impl: type[Implementation[Any]],
    universe: frozenset[str],
    probe: Probe | None,
) -> list[str]:
    if not impl.compute_framework.is_available():
        return []
    probe = probe or _shape_a_probe(group)
    if probe is None:
        return [f"declines: cannot probe {impl.__name__}; pass probe="]
    supported = universe if impl.supported is None else universe & impl.supported
    failures: list[str] = []

    def verdict(subtype: str) -> tuple[bool, bool]:
        name, options = probe(subtype)
        resolved = group.resolve_subtype(name, options) == subtype
        return resolved, group.supports_compute_framework(name, options, impl.compute_framework)

    if supported:
        control = sorted(supported)[0]
        resolved, accepted = verdict(control)
        if not (resolved and accepted):
            failures.append(f"declines: probe not accepted for '{control}'")
    for subtype in sorted(universe - supported):
        resolved, accepted = verdict(subtype)
        if not resolved:
            failures.append(f"declines: probe does not resolve to '{subtype}'")
        elif accepted:
            failures.append(f"declines: {impl.__name__} does not decline unsupported '{subtype}'")
    return failures


def check_operation_group_shape(
    group: type[OperationFeatureGroup],
    alias_pairs: Sequence[tuple[str, str]] = (),
    probe: Probe | None = None,
) -> CheckResult:
    """Check each serving Implementation declares valid supported and divergent subtypes and declines the rest."""
    universe = group.subtype_universe()
    impls = _implementations(group)
    failures: list[str] = []
    if not impls:
        failures.append(f"implementations: no loaded Implementation serves {group.__name__}")
    for impl in impls:
        failures.extend(_declaration_failures(impl, universe, alias_pairs))
        failures.extend(_decline_failures(group, impl, universe, probe))
    return CheckResult(check="operation_group_shape", subject=group.__name__, failures=tuple(failures))


def assert_operation_group_shape(
    group: type[OperationFeatureGroup],
    alias_pairs: Sequence[tuple[str, str]] = (),
    probe: Probe | None = None,
) -> None:
    """Fail the running pytest test if the operation group shape check reports failures."""
    assert_passed(check_operation_group_shape(group, alias_pairs, probe))
