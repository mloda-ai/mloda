"""Scope check for a FeatureGroup subclass: it takes over only when its selecting option is set."""

from __future__ import annotations

from collections.abc import Sequence

from typing import Any

from mloda._testing.feature_group_shape import (
    _base_options,
    _first_value,
    _framework_gate_reason,
    _input_problems,
    _no_calculate_reason,
    _run as _run_group,
    _same,
)
from mloda._testing.framework_adapter import FrameworkAdapter, PythonDictAdapter
from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.divergence import Divergence, DivergenceKind
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation, current_implementations
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.prepare.identify_feature_group import evaluate_and_render


def _mapping(
    parent: type[FeatureGroup], subclass: type[FeatureGroup]
) -> tuple[FeatureGroupEnvironmentMapping, list[str]]:
    loaded = PreFilterPlugins.get_cfw_subclasses()
    mapping: FeatureGroupEnvironmentMapping = {}
    failures: list[str] = []
    for cls in (parent, subclass):
        mapping[cls] = {cfw for cfw in cls.compute_framework_definition() if cfw in loaded}
        if not mapping[cls]:
            failures.append(f"setup: no compute framework of {cls.__name__} is loaded")
    return mapping, failures


def _combined(base: Options, selecting: Options) -> Options:
    return Options(group={**base.group, **selecting.group}, context={**base.context, **selecting.context})


def _optional_probes(parent: type[FeatureGroup], base: Options, keys: set[str]) -> list[tuple[str, Options]]:
    taken = {*base.group, *base.context, *keys}
    probes = []
    for key, spec in (parent.PROPERTY_MAPPING or {}).items():
        if key in taken or not isinstance(spec, PropertySpec):
            continue
        value = _first_value(parent, key, spec)
        if value is not None:
            probes.append((key, Options(group={**base.group, key: value}, context=dict(base.context))))
    return probes


def _plain_failure(
    name: str,
    options: Options,
    label: str,
    mapping: FeatureGroupEnvironmentMapping,
    parent: type[FeatureGroup],
    subclass: type[FeatureGroup],
) -> str | None:
    result, message = evaluate_and_render(Feature(name, options), mapping)
    found = set(result.identified)
    if found == {parent}:
        return None
    if found == {subclass}:
        return f"steals: '{name}' {label} resolves to {subclass.__name__}"
    if found:
        return f"setup: '{name}' {label} is ambiguous between {', '.join(sorted(c.__name__ for c in found))}"
    return f"setup: {parent.__name__} itself does not resolve '{name}' {label}: {message}"


def _scope_failures(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    names: list[str],
    base: Options,
    selecting: Options,
) -> list[str]:
    keys = {*selecting.group, *selecting.context}
    mapping, failures = _mapping(parent, subclass)
    if failures:
        return failures
    label = f"without {sorted(keys)}"
    probes = [("", base), *_optional_probes(parent, base, keys)]
    for name in names:
        for key, options in probes:
            failure = _plain_failure(name, options, f"with option '{key}'" if key else label, mapping, parent, subclass)
            if failure is not None:
                failures.append(failure)
        selected, selected_message = evaluate_and_render(Feature(name, _combined(base, selecting)), mapping)
        if set(selected.identified) != {subclass}:
            found = selected_message or ", ".join(sorted(c.__name__ for c in selected.identified))
            failures.append(f"misses: '{name}' with {sorted(keys)} resolves to {found}")
    return failures


_EXCUSING = (DivergenceKind.VALUE, DivergenceKind.TYPE, DivergenceKind.VARIANT)


def _own_records(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    sub_impl: type[Implementation[Any]] | None,
    parent_impl: type[Implementation[Any]] | None,
) -> list[Divergence]:
    records = [r for r in subclass.DIVERGENCES if r not in parent.DIVERGENCES]
    if sub_impl is not None and parent_impl is not None and sub_impl is not parent_impl:
        records += [r for r in sub_impl.divergences if r not in parent_impl.divergences]
    return [r for r in records if r.kind in _EXCUSING]


def _compared(columns: list[str], names: list[str]) -> list[tuple[str, str, frozenset[str]]]:
    found = []
    for column in columns:
        for name in names:
            if column == name:
                found.append((column, name, frozenset[str]()))
            elif column.startswith(f"{name}~"):
                found.append((column, name, frozenset({column[len(name) + 1 :]})))
    return found


def _value_failures(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    names: list[str],
    base: Options,
    selecting: Options,
    data: TypedTestData,
    adapter: FrameworkAdapter,
    notes: list[str],
) -> list[str]:
    for cls in (parent, subclass):
        reason = _framework_gate_reason(cls, adapter)
        if reason is None and not is_nominal_subclass(cls, OperationFeatureGroup):
            reason = _no_calculate_reason(cls)
        if reason is None and is_nominal_subclass(cls, SourceFeatureGroup):
            reason = "not applicable to source groups"
        if reason is not None:
            notes.append(f"values: {reason}")
            return []
    chosen = _combined(base, selecting)
    impls: dict[type[FeatureGroup], type[Implementation[Any]] | None] = {parent: None, subclass: None}
    for cls in (parent, subclass):
        if is_nominal_subclass(cls, OperationFeatureGroup):
            impls[cls] = cls.select_implementation(adapter.compute_framework(), list(current_implementations()))
            if impls[cls] is None:
                notes.append(
                    f"values: no implementation of {cls.__name__} serves {adapter.compute_framework().__name__}"
                )
                return []
    problems = _input_problems(parent, names, base, data) + _input_problems(subclass, names, chosen, data)
    if problems:
        return [f"values: {p}" for p in dict.fromkeys(problems)]
    expected, error = _run_group(parent, adapter, data, names, base, impls[parent])
    if error is not None:
        return [f"setup: {parent.__name__} run raised: {error}"]
    actual, error = _run_group(subclass, adapter, data, names, chosen, impls[subclass])
    if error is not None:
        return [f"values: {subclass.__name__} run raised: {error}"]
    records = _own_records(subclass, parent, impls[subclass], impls[parent])
    present = adapter.column_names(actual)
    failures = []
    for column, name, parts in _compared(adapter.column_names(expected), names):
        if column not in present:
            failures.append(f"values: '{column}' is missing from {subclass.__name__}'s result")
        elif not _same(adapter.column_values(expected, column), adapter.column_values(actual, column)):
            subtype = Implementation.canonical_subtype(subclass, name, chosen)
            if not any(r.applies_to(subtype, parts) for r in records):
                failures.append(
                    f"values: '{column}' differs from {parent.__name__} and {subclass.__name__} declares no divergence for it"
                )
    return failures


def check_subclass_scope(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    selecting: Options,
    requests: Sequence[str] | None = None,
    *,
    input_data: TypedTestData | None = None,
    adapter: FrameworkAdapter | None = None,
) -> CheckResult:
    """Check subclass replaces parent only with the selecting options; only these two classes are considered.

    With input_data it also compares the requested output columns with the parent's, assuming identical row order.
    """
    failures = _run(subclass, parent, selecting, requests)
    notes: list[str] = []
    if input_data is None:
        notes.append("values: not compared (no input_data)")
    elif not any(f.startswith(("setup:", "names:")) for f in failures):
        names = list(requests) if requests is not None else sorted(parent.feature_names_supported())
        base = _base_options(parent)
        if base is not None:
            failures += _value_failures(
                subclass, parent, names, base, selecting, input_data, adapter or PythonDictAdapter(), notes
            )
    return CheckResult(check="subclass_scope", subject=subclass.__name__, failures=tuple(failures), notes=tuple(notes))


def _run(
    subclass: type[FeatureGroup], parent: type[FeatureGroup], selecting: Options, requests: Sequence[str] | None
) -> list[str]:
    if not is_nominal_subclass(subclass, parent) or subclass is parent:
        return [f"setup: {subclass.__name__} is not a proper subclass of {parent.__name__}"]
    declared = sorted(parent.feature_names_supported())
    names = list(requests) if requests is not None else declared
    if not names:
        if requests is not None:
            return ["names: no requests to check"]
        return ["names: parent declares no feature names; pass requests="]
    base = _base_options(parent)
    if base is None:
        return ["setup: cannot build a request for the parent; declare allowed_values or default"]
    present = sorted({*selecting.group, *selecting.context} & {*base.group, *base.context})
    if present:
        return [f"setup: selecting keys {present} are already in the base options and would mask a steal"]
    return _scope_failures(subclass, parent, names, base, selecting)


def assert_subclass_scope(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    selecting: Options,
    requests: Sequence[str] | None = None,
    *,
    input_data: TypedTestData | None = None,
    adapter: FrameworkAdapter | None = None,
) -> None:
    """Fail the running pytest test if the subclass scope check reports failures."""
    assert_passed(check_subclass_scope(subclass, parent, selecting, requests, input_data=input_data, adapter=adapter))
