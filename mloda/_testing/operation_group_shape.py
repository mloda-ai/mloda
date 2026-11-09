"""Shape check for an operation group: its Implementations declare consistent supported subtypes and divergences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.divergence import DivergenceKind, divergence_problems
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.implementation import Implementation, current_implementations, serving_implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup

Probe = Callable[[str], tuple[str, Options]]
"""Maps a subtype to a (feature name, options) request that the Implementation accepts."""


def _implementations(group: type[OperationFeatureGroup]) -> dict[type[Implementation[Any]], type[ComputeFramework]]:
    current = list(current_implementations())
    found: dict[type[Implementation[Any]], type[ComputeFramework]] = {}
    for framework in group.compute_framework_definition():
        impl = serving_implementation(group, framework, current)
        if impl is not None and impl not in found:
            found[impl] = framework
    return found


def _reasons(impl: type[Implementation[Any]], subtype: str) -> list[tuple[str, str]]:
    records = (r for r in impl.divergences if r.kind is not DivergenceKind.GAP and r.applies_to(subtype, None))
    return sorted((r.kind.value, r.describe()) for r in records)


def _alias_failures(
    impls: Sequence[type[Implementation[Any]]], universe: frozenset[str], alias_pairs: Sequence[tuple[str, str]]
) -> list[str]:
    failures: list[str] = []
    for first, second in alias_pairs:
        missing = [n for n in (first, second) if n not in universe]
        if missing:
            failures.append(f"aliases: ({first}, {second}) names {missing} outside the subtype universe")
            continue
        for impl in impls:
            if _reasons(impl, first) != _reasons(impl, second):
                failures.append(f"aliases: {impl.__name__} declares different divergences for '{first}' and '{second}'")
    return failures


def _declaration_failures(
    impl: type[Implementation[Any]], universe: frozenset[str], parts: tuple[str, ...] | None, variants: frozenset[str]
) -> list[str]:
    name = impl.__name__
    failures: list[str] = []
    if impl.supported is not None:
        extra = sorted(impl.supported - universe)
        if extra:
            failures.append(f"supported: {name} lists {extra} outside the subtype universe")
    failures.extend(
        f"divergences: {p}" for p in divergence_problems(name, impl.divergences, universe, parts, variants=variants)
    )
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
    framework: type[ComputeFramework],
    universe: frozenset[str],
    probe: Probe | None,
) -> list[str]:
    if not universe or not framework.is_available():
        return []
    probe = probe or _shape_a_probe(group)
    if probe is None:
        return [] if impl.supported is None else [f"declines: cannot probe {impl.__name__}; pass probe="]
    supported = universe if impl.supported is None else universe & impl.supported
    failures: list[str] = []

    def verdict(subtype: str) -> tuple[bool, bool, str | None]:
        name, options = probe(subtype)
        resolved = group.resolve_subtype(name, options) == subtype
        accepted = group.supports_compute_framework(name, options, framework)
        return resolved, accepted, impl.canonical_subtype(group, name, options)

    if supported:
        control = sorted(supported)[0]
        resolved, accepted, _canonical = verdict(control)
        if not (resolved and accepted):
            failures.append(f"declines: probe not accepted for '{control}'")
    for subtype in sorted(universe - supported):
        resolved, accepted, canonical = verdict(subtype)
        if not resolved:
            failures.append(f"declines: probe does not resolve to '{subtype}'")
        elif accepted:
            failures.append(f"declines: {impl.__name__} does not decline unsupported '{subtype}'")
        elif canonical is None or canonical in supported:
            failures.append(f"declines: {impl.__name__} declines '{subtype}' for a reason other than supported")
    return failures


def check_operation_group_shape(
    group: type[OperationFeatureGroup],
    alias_pairs: Sequence[tuple[str, str]] = (),
    probe: Probe | None = None,
) -> CheckResult:
    """Check Implementations declare valid supported subtypes and divergences and decline the rest.

    The acceptance control uses one supported subtype."""
    universe = group.subtype_universe()
    impls = _implementations(group)
    failures: list[str] = []
    if not impls:
        failures.append(f"implementations: no loaded Implementation serves {group.__name__}")
    failures.extend(_alias_failures(list(impls), universe, alias_pairs))
    failures.extend(
        f"divergences: {p}"
        for p in divergence_problems(
            group.__name__, group.DIVERGENCES, universe, group.OUTPUT_PARTS, variants=group.VARIANTS
        )
    )
    for impl, framework in impls.items():
        failures.extend(_declaration_failures(impl, universe, group.OUTPUT_PARTS, group.VARIANTS))
        failures.extend(_decline_failures(group, impl, framework, universe, probe))
    return CheckResult(check="operation_group_shape", subject=group.__name__, failures=tuple(failures))


def assert_operation_group_shape(
    group: type[OperationFeatureGroup],
    alias_pairs: Sequence[tuple[str, str]] = (),
    probe: Probe | None = None,
) -> None:
    """Fail the running pytest test if the operation group shape check reports failures."""
    assert_passed(check_operation_group_shape(group, alias_pairs, probe))
