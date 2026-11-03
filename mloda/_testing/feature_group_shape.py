"""Shape check for a plain FeatureGroup: names, foreign names, inputs, required and strict options."""

from __future__ import annotations

import inspect
import math
from itertools import product
from typing import Any

from mloda._testing.framework_adapter import FrameworkAdapter, PyArrowAdapter, PythonDictAdapter
from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda._testing.typed_data import TypedTestData
from mloda._testing.value_fit import value_fits
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.match_rejection import CoreDeclineCode
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
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


def _cell_same(x: Any, y: Any) -> bool:
    if x is y:
        return True
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return bool((x == y) is True)


def _same(a: list[Any], b: list[Any]) -> bool:
    return len(a) == len(b) and all(_cell_same(x, y) for x, y in zip(a, b))


def _run(
    cls: type[FeatureGroup], adapter: FrameworkAdapter, data: TypedTestData, names: list[str], base: Options
) -> tuple[Any, str | None]:
    """Run through the adapter's ComputeFramework like a feature group step, then validate the output."""

    def go() -> Any:
        features = FeatureSet()
        for name in names:
            features.add(Feature(name, options=base))
        cfw = adapter.compute_framework()(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        cfw.run_calculation(cls, features, None, adapter.from_typed(data))
        cls.validate_output_features(cfw.data, features)
        return cfw.data

    return safe_field_with_error(go, None)


def _declared_inputs(cls: type[FeatureGroup], names: list[str], base: Options) -> set[str]:
    if cls.input_features is FeatureGroup.input_features:
        return set()
    found: set[str] = set()
    for name in names:
        declared, _ = safe_field_with_error(lambda: cls.declared_input_features(base, FeatureName(name)), None)
        found |= {str(f.name) for f in declared or ()}
    return found


def _expected_types(cls: type[FeatureGroup], names: list[str], base: Options) -> dict[str, DataType]:
    types = {}
    for name in names:
        rule, _ = safe_field_with_error(lambda: cls.return_data_type_rule(Feature(name, options=base)), None)
        if rule is not None:
            types[name] = rule
    return types


def _check_outputs(
    cls: type[FeatureGroup], adapter: FrameworkAdapter, data: TypedTestData, names: list[str], result: Any
) -> list[str]:
    present = adapter.column_names(result)
    failures = [f"outputs: '{n}' is missing from the result" for n in names if n not in present]
    if cls.replaces_rows():
        return failures
    for column, values in data.to_python_dict().items():
        if column in names or column not in present:
            continue
        if not _same(adapter.column_values(result, column), values):
            failures.append(f"outputs: input column '{column}' was changed")
    return failures


def _check_types(adapter: FrameworkAdapter, expected: dict[str, DataType], result: Any, notes: list[str]) -> list[str]:
    failures = []
    exact = isinstance(adapter, PyArrowAdapter)
    for name, data_type in expected.items():
        if name not in adapter.column_names(result):
            continue
        if exact:
            if adapter.lost_type(result, name):
                failures.append(f"type: '{name}' lost its type")
            elif adapter.column_type(result, name) != DataType.to_arrow_type(data_type):
                failures.append(f"type: '{name}' is {adapter.column_type(result, name)}, expected {data_type.name}")
        elif not all(value_fits(data_type, v) for v in adapter.column_values(result, name)):
            failures.append(f"type: '{name}' holds values that do not fit {data_type.name}")
    if not exact and adapter.capabilities.typed_columns:
        notes.append("type: exact column types are only checked on the PyArrow adapter")
    return failures


def _check_zero_rows(
    cls: type[FeatureGroup],
    adapter: FrameworkAdapter,
    data: TypedTestData,
    names: list[str],
    base: Options,
    result: Any,
) -> list[str]:
    empty = TypedTestData({n: (t, []) for n, t in data.types.items()})
    out, error = _run(cls, adapter, empty, names, base)
    if error is not None:
        return [f"zero_rows: run on zero rows raised: {error}"]
    failures = []
    if adapter.row_count(out) != 0:
        failures.append(f"zero_rows: {adapter.row_count(out)} rows came back for zero input rows")
    columns = adapter.column_names(result)
    if sorted(adapter.column_names(out)) != sorted(columns):
        failures.append("zero_rows: column names differ from the run with rows")
    elif adapter.capabilities.typed_columns:
        for column in columns:
            if adapter.column_type(out, column) != adapter.column_type(result, column):
                failures.append(f"zero_rows: column '{column}' changes type on zero rows")
    return failures


def _check_error_values(
    cls: type[FeatureGroup],
    adapter: FrameworkAdapter,
    data: TypedTestData,
    names: list[str],
    base: Options,
    notes: list[str],
) -> list[str]:
    """A raise on perturbed string cells must not echo a fixture cell; only raises we trigger are seen."""
    types = data.types
    columns = data.to_python_dict()
    strings = {n for n, t in types.items() if t == DataType.STRING}
    cells = {str(v) for n in strings for v in columns[n] if v is not None and len(str(v)) >= 4}
    perturbed: dict[str, tuple[DataType, list[Any]]] = {}
    inserted: set[str] = set()
    for n, t in types.items():
        values = columns[n]
        if n in strings:
            values = [None if v is None else f"perturbed_{n}_{i}" for i, v in enumerate(values)]
            inserted |= {v for v in values if v is not None}
        perturbed[n] = (t, values)
    _, error = _run(cls, adapter, TypedTestData(perturbed), names, base)
    if error is None:
        notes.append("error_values: the perturbed run did not raise, nothing to inspect")
        return []
    return [f"error_values: the error text echoes the input cell '{c}'" for c in sorted(cells | inserted) if c in error]


def _check_determinism(
    cls: type[FeatureGroup],
    adapter: FrameworkAdapter,
    data: TypedTestData,
    names: list[str],
    base: Options,
    result: Any,
) -> list[str]:
    again, error = _run(cls, adapter, data, names, base)
    if error is not None:
        return [f"determinism: second run raised: {error}"]
    first, second = adapter.column_names(result), adapter.column_names(again)
    return [
        f"determinism: '{n}' differs between two runs"
        for n in names
        if n in first and n in second and not _same(adapter.column_values(result, n), adapter.column_values(again, n))
    ]


def _run_checks(
    cls: type[FeatureGroup],
    names: list[str],
    base: Options,
    data: TypedTestData,
    adapter: FrameworkAdapter,
    notes: list[str],
) -> list[str]:
    if inspect.getattr_static(cls, "calculate_feature") is inspect.getattr_static(FeatureGroup, "calculate_feature"):
        notes.append("run: not applicable, the group has no calculate_feature")
        return []
    framework = adapter.compute_framework()
    rule = cls.compute_framework_rule()
    if rule is not None and framework not in rule:
        notes.append(f"run: {framework.__name__} excluded by compute_framework_rule")
        return []
    if not adapter.is_available():
        notes.append(f"run: {framework.__name__} is not available")
        return []
    missing = sorted(_declared_inputs(cls, names, base) - set(data.types))
    if missing:
        return [f"outputs: declared input '{m}' is missing from input_data" for m in missing]
    clobbered = [n for n in names if n in data.types]
    if clobbered:
        return [f"outputs: requested name '{n}' is already in input_data" for n in clobbered]
    result, error = _run(cls, adapter, data, names, base)
    if error is not None:
        return [f"outputs: run raised: {error}"]
    failures = _check_outputs(cls, adapter, data, names, result)
    failures += _check_types(adapter, _expected_types(cls, names, base), result, notes)
    if data.types:
        failures += _check_zero_rows(cls, adapter, data, names, base, result)
        failures += _check_error_values(cls, adapter, data, names, base, notes)
    else:
        notes.append("run: zero_rows and error_values were not run (input_data has no columns)")
    failures += _check_determinism(cls, adapter, data, names, base, result)
    return failures


def check_feature_group_shape(
    cls: type[FeatureGroup],
    base_options: Options | None = None,
    *,
    input_data: TypedTestData | None = None,
    adapter: FrameworkAdapter | None = None,
) -> CheckResult:
    """Check a plain FeatureGroup's match contract and, given input_data, its run; a raising matcher is a non-match."""
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
    notes: list[str] = []
    if input_data is None:
        notes.append("run: not run (no input_data); pass input_data to check outputs, types and determinism")
    else:
        run_names = declared or [cls.get_class_name()]
        failures += _run_checks(cls, run_names, base, input_data, adapter or PythonDictAdapter(), notes)
    return CheckResult(check="feature_group_shape", subject=cls.__name__, failures=tuple(failures), notes=tuple(notes))


def assert_feature_group_shape(
    cls: type[FeatureGroup],
    base_options: Options | None = None,
    *,
    input_data: TypedTestData | None = None,
    adapter: FrameworkAdapter | None = None,
) -> None:
    """Fail the running pytest test if the shape check reports failures."""
    assert_passed(check_feature_group_shape(cls, base_options, input_data=input_data, adapter=adapter))
