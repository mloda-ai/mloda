"""Shape check for a plain FeatureGroup: names, foreign names, inputs, required and strict options."""

from __future__ import annotations

from itertools import product
from typing import Any

from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.match_rejection import CoreDeclineCode
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec, is_no_default
from mloda.core.abstract_plugins.components.utils import safe_field_with_error
from mloda.core.abstract_plugins.feature_group import FeatureGroup

_FOREIGN_NAME = "conformance_foreign_feature"
_UNKNOWN_VALUE = "conformance_unknown_value"


def _mapping(cls: type[FeatureGroup]) -> dict[str, PropertySpec]:
    return {k: v for k, v in (cls.PROPERTY_MAPPING or {}).items() if isinstance(v, PropertySpec)}


def _required_keys(cls: type[FeatureGroup]) -> list[str]:
    return FeatureChainParser._name_path_missing_required_keys(Options(), _mapping(cls))


def _first_value(cls: type[FeatureGroup], key: str, spec: PropertySpec) -> Any:
    if not is_no_default(spec.default) and spec.default is not None:
        return spec.default
    if spec.allowed_values:
        return next(iter(spec.allowed_values))
    values = sorted(cls.declared_option_values(key))
    return values[0] if values else None


def _base_options(cls: type[FeatureGroup]) -> Options | None:
    mapping = _mapping(cls)
    group: dict[str, Any] = {}
    for key in _required_keys(cls):
        value = _first_value(cls, key, mapping[key])
        if value is None:
            return None
        group[key] = value
    return cls.options_with_defaults(Options(group=group))


def _without_key(options: Options, key: str) -> Options:
    return Options(
        group={k: v for k, v in options.group.items() if k != key},
        context={k: v for k, v in options.context.items() if k != key},
    )


def _with_value(options: Options, key: str, value: Any) -> Options:
    return Options(
        group={**{k: v for k, v in options.group.items() if k != key}, key: value},
        context={k: v for k, v in options.context.items() if k != key},
    )


def _probe(cls: type[FeatureGroup], name: str, options: Options) -> tuple[bool, Any]:
    outcome = probe_match_criteria(cls, name, options)
    return outcome.matched, outcome.rejection


def _foreign_name(cls: type[FeatureGroup]) -> str:
    name = _FOREIGN_NAME
    supported = cls.feature_names_supported()
    while name == cls.get_class_name() or name.startswith(cls.prefix()) or name in supported:
        name = "x" + name
    return name


def _check_names(cls: type[FeatureGroup], names: list[str], base: Options) -> list[str]:
    failures = []
    for name in names:
        if not _probe(cls, name, base)[0]:
            failures.append(f"names: '{name}' is declared but {cls.__name__} does not match it")
    return failures


def _check_foreign(cls: type[FeatureGroup], base: Options) -> list[str]:
    name = _foreign_name(cls)
    undeclared = "conformance_undeclared_key"
    while undeclared in cls.declared_option_keys():
        undeclared = "x" + undeclared
    failures = []
    for label, options in (("base options", base), ("an undeclared option", Options(group={undeclared: "1"}))):
        matched, rejection = _probe(cls, name, options)
        if matched:
            failures.append(f"foreign: matches the undeclared name '{name}' with {label}")
        elif rejection is not None and rejection.code is None:
            failures.append(f"foreign: rejection of '{name}' with {label} carries no decline code")
    return failures


def _check_inputs(cls: type[FeatureGroup], names: list[str], base: Options) -> list[str]:
    failures: list[str] = []
    # A group keeping the default input_features is a root and declares no inputs.
    if cls.input_features is FeatureGroup.input_features:
        return failures
    for name in names:
        declared, error = safe_field_with_error(lambda: cls.declared_input_features(base, FeatureName(name)), None)
        if error is not None:
            failures.append(f"inputs: '{name}' input_features raised: {error}")
            continue
        if declared is None:
            continue
        if not isinstance(declared, set) or not all(isinstance(f, Feature) for f in declared):
            failures.append(f"inputs: '{name}' declares inputs that are not a set of Feature")
            continue
        for feature in declared:
            input_name = str(feature.name)
            if not input_name:
                failures.append(f"inputs: '{name}' declares an input with an empty name")
            elif input_name == name:
                failures.append(f"inputs: '{name}' declares itself as an input")
    return failures


def _check_required(cls: type[FeatureGroup], names: list[str], base: Options) -> list[str]:
    """Dropping each required key must record REQUIRED_OPTION_ABSENT naming it."""
    failures = []
    for name, key in product(names, _required_keys(cls)):
        matched, rejection = _probe(cls, name, _without_key(base, key))
        if matched:
            failures.append(f"required: matches without required option '{key}'")
        elif rejection is None:
            failures.append(f"required: no rejection recorded without required option '{key}'")
        elif rejection.code != CoreDeclineCode.REQUIRED_OPTION_ABSENT:
            failures.append(
                f"required: rejection without '{key}' has code {rejection.code}, not required_option_absent"
            )
        elif key not in rejection.reason:
            failures.append(f"required: rejection reason does not name the absent option '{key}'")
    return failures


def _check_strict(cls: type[FeatureGroup], names: list[str], base: Options) -> list[str]:
    failures = []
    strict = [(k, v) for k, v in _mapping(cls).items() if v.strict_validation and v.allowed_values]
    for name, (key, _) in product(names, strict):
        matched, rejection = _probe(cls, name, _with_value(base, key, _UNKNOWN_VALUE))
        if matched:
            failures.append(f"strict: matches an unknown value for strict option '{key}'")
        elif rejection is None or rejection.code != CoreDeclineCode.OPTION_VALUE_REJECTED:
            failures.append(f"strict: rejection of an unknown '{key}' value lacks code option_value_rejected")
    return failures


def check_feature_group_shape(cls: type[FeatureGroup], base_options: Options | None = None) -> CheckResult:
    """Check a plain FeatureGroup's match contract; a raising matcher counts as a non-match."""
    base = base_options if base_options is not None else _base_options(cls)
    if base is None:
        failures = ["names: cannot build a matching request; declare allowed_values or default, or pass base_options"]
        return CheckResult(check="feature_group_shape", subject=cls.__name__, failures=tuple(failures))
    declared = sorted(cls.feature_names_supported())
    names = [cls.get_class_name(), *declared]
    failures = _check_names(cls, names, base)
    failures += _check_foreign(cls, base)
    failures += _check_inputs(cls, names, base)
    failures += _check_required(cls, names, base)
    failures += _check_strict(cls, names, base)
    return CheckResult(check="feature_group_shape", subject=cls.__name__, failures=tuple(failures))


def assert_feature_group_shape(cls: type[FeatureGroup], base_options: Options | None = None) -> None:
    """Fail the running pytest test if the shape check reports failures."""
    assert_passed(check_feature_group_shape(cls, base_options))
