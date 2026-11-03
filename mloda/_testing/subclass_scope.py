"""Scope check for a FeatureGroup subclass: it takes over only when its selecting option is set."""

from __future__ import annotations

from collections.abc import Sequence

from mloda._testing.feature_group_shape import _base_options
from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.prepare.identify_feature_group import evaluate_and_render


def _mapping(parent: type[FeatureGroup], subclass: type[FeatureGroup]) -> FeatureGroupEnvironmentMapping:
    loaded = PreFilterPlugins.get_cfw_subclasses()
    return {
        cls: {cfw for cfw in cls.compute_framework_definition() if cfw in loaded} or set(loaded)
        for cls in (parent, subclass)
    }


def _combined(base: Options, selecting: Options) -> Options:
    return Options(group={**base.group, **selecting.group}, context={**base.context, **selecting.context})


def _scope_failures(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    names: list[str],
    base: Options,
    selecting: Options,
) -> list[str]:
    keys = sorted({*selecting.group, *selecting.context})
    mapping = _mapping(parent, subclass)
    failures: list[str] = []
    for name in names:
        plain, plain_message = evaluate_and_render(Feature(name, base), mapping)
        if set(plain.identified) == {subclass} and parent in plain.specialized_from:
            failures.append(
                f"steals: '{name}' without {keys} resolves to {subclass.__name__} (specialized from {parent.__name__})"
            )
        elif set(plain.identified) != {parent}:
            failures.append(f"setup: '{name}' without {keys} does not resolve to {parent.__name__}: {plain_message}")
        selected, selected_message = evaluate_and_render(Feature(name, _combined(base, selecting)), mapping)
        if set(selected.identified) != {subclass} or parent not in selected.specialized_from:
            found = selected_message or ", ".join(sorted(c.__name__ for c in selected.identified))
            failures.append(f"misses: '{name}' with {keys} resolves to {found}")
    return failures


def check_subclass_scope(
    subclass: type[FeatureGroup],
    parent: type[FeatureGroup],
    selecting: Options,
    requests: Sequence[str] | None = None,
) -> CheckResult:
    """Check that subclass replaces parent only with the selecting options; value differences are not checked."""
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
