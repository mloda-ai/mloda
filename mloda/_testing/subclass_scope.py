"""Scope check for a FeatureGroup subclass: it takes over only when its selecting option is set."""

from __future__ import annotations

from collections.abc import Sequence

from mloda._testing.feature_group_shape import _base_options, _first_value
from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.abstract_plugins.feature_group import FeatureGroup
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


def check_subclass_scope(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    selecting: Options,
    requests: Sequence[str] | None = None,
) -> CheckResult:
    """Check subclass replaces parent only with the selecting options; only these two classes are considered."""
    failures = _run(subclass, parent, selecting, requests)
    return CheckResult(check="subclass_scope", subject=subclass.__name__, failures=tuple(failures))


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
) -> None:
    """Fail the running pytest test if the subclass scope check reports failures."""
    assert_passed(check_subclass_scope(subclass, parent, selecting, requests))
