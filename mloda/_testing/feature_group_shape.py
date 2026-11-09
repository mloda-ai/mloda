"""Shape check for a FeatureGroup: names, foreign names, inputs, required and strict options."""

from __future__ import annotations

import inspect
import math
from itertools import product
from typing import Any

from mloda._testing.framework_adapter import FrameworkAdapter, PythonDictAdapter
from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda._testing.typed_data import TypedTestData
from mloda._testing.value_fit import value_fits
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.divergence import Divergence
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    ANY_OUTPUT_PART,
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.match_rejection import CoreDeclineCode
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.property_spec import PropertySpec, is_no_default
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass, safe_field_with_error
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import (
    Implementation,
    current_implementations,
    unallowed_variants,
    variant_block_reason,
)
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup

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


def _serving_implementation(cls: type[FeatureGroup], adapter: FrameworkAdapter) -> type[Implementation[Any]] | None:
    """The Implementation an operation group runs through on the adapter's framework; None for any other group."""
    if not is_nominal_subclass(cls, OperationFeatureGroup):
        return None
    return cls.select_implementation(adapter.compute_framework(), list(current_implementations()))


def _run(
    cls: type[FeatureGroup],
    adapter: FrameworkAdapter,
    data: TypedTestData,
    names: list[str],
    base: Options,
) -> tuple[Any, str | None]:
    """Run through the adapter's ComputeFramework like a feature group step, then validate the output."""

    def go() -> Any:
        features = FeatureSet()
        features.implementation = _serving_implementation(cls, adapter)
        for name in names:
            features.add(Feature(name, options=base))
        cfw = adapter.compute_framework()(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        cfw.run_calculation(cls, features, None, adapter.from_typed(data))
        cls.validate_output_features(cfw.data, features)
        return cfw.data

    return safe_field_with_error(go, None)


def _compared(columns: list[str], names: list[str]) -> list[tuple[str, str, frozenset[str]]]:
    unique = list(dict.fromkeys(names))
    found = []
    for column in columns:
        if column in unique:
            found.append((column, column, frozenset[str]()))
            continue
        prefixed = [n for n in unique if column.startswith(f"{n}~")]
        if prefixed:
            name = max(prefixed, key=len)
            found.append((column, name, frozenset({column[len(name) + 1 :]})))
    return found


def _written(
    adapter: FrameworkAdapter, data: TypedTestData, result: Any, names: list[str]
) -> list[tuple[str, str, frozenset[str]]]:
    """The result columns mapped to a requested name; columns that came in with input_data never count."""
    return _compared([c for c in adapter.column_names(result) if c not in data.types], names)


def _concrete_parts(cls: type[FeatureGroup], name: str, base: Options) -> tuple[str, ...] | None:
    """The declared output parts when they are a concrete tuple; None when undeclared or named at run time."""
    parts, _ = safe_field_with_error(lambda: cls.output_parts(name, base), None)
    return None if parts is None or ANY_OUTPUT_PART in parts else tuple(parts)


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
    cls: type[FeatureGroup],
    adapter: FrameworkAdapter,
    data: TypedTestData,
    names: list[str],
    base: Options,
    result: Any,
) -> list[str]:
    present = adapter.column_names(result)
    columns = {c for c, _, _ in _written(adapter, data, result, names)}
    written = {n for _, n, _ in _written(adapter, data, result, names)}
    failures = [f"outputs: '{n}' is missing from the result" for n in names if n not in written]
    for name in names:
        for part in _concrete_parts(cls, name, base) or ():
            if f"{name}~{part}" not in columns:
                failures.append(f"outputs: declared part '{name}~{part}' is missing from the result")
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
    for name, data_type in expected.items():
        if name not in adapter.column_names(result):
            continue
        exact = adapter.has_type(result, name, data_type)
        if exact is True:
            continue
        if exact is False:
            if adapter.lost_type(result, name):
                failures.append(f"type: '{name}' lost its type")
            else:
                failures.append(f"type: '{name}' is {adapter.column_type(result, name)}, expected {data_type.name}")
            continue
        if not all(value_fits(data_type, v) for v in adapter.column_values(result, name)):
            failures.append(f"type: '{name}' holds values that do not fit {data_type.name}")
        if adapter.capabilities.typed_columns:
            notes.append(
                f"type: '{name}' checked by value only, {type(adapter).__name__} has no exact {data_type.name} type"
            )
    return failures


def _check_zero_rows(
    cls: type[FeatureGroup],
    adapter: FrameworkAdapter,
    data: TypedTestData,
    names: list[str],
    base: Options,
    result: Any,
    expected: dict[str, DataType],
) -> list[str]:
    empty = TypedTestData({n: (t, []) for n, t in data.types.items()})
    out, error = _run(cls, adapter, empty, names, base)
    if error is not None:
        return [f"zero_rows: run on zero rows raised: {error}"]
    failures = []
    if adapter.row_count(out) != 0:
        failures.append(f"zero_rows: {adapter.row_count(out)} rows came back for zero input rows")
    run_time = {n for n in names if _concrete_parts(cls, n, base) is None}

    def named(frame: Any) -> list[str]:
        columns = adapter.column_names(frame)
        hidden = {c for c, n, parts in _compared(columns, names) if parts and n in run_time}
        return sorted(c for c in columns if c not in hidden)

    columns = adapter.column_names(result)
    out_columns = adapter.column_names(out)
    if named(out) != named(result):
        differing = sorted(set(named(out)) ^ set(named(result)))
        failures.append(f"zero_rows: column names differ from the run with rows: {differing}")
    elif adapter.capabilities.typed_columns:
        for column in (c for c in columns if c in out_columns):
            mapped = _compared([column], names)
            data_type = expected.get(mapped[0][1]) if mapped else None
            exact = None if data_type is None else adapter.has_type(result, column, data_type)
            if exact is False:
                continue
            if exact is True and data_type is not None:
                if data_type in (DataType.LIST, DataType.TIMESTAMP_MICROS_TZ):
                    changed = adapter.column_type(out, column) != adapter.column_type(result, column)
                else:
                    changed = adapter.has_type(out, column, data_type) is not True
            else:
                changed = adapter.column_type(out, column) != adapter.column_type(result, column)
            if changed:
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
    first = {c for c, _, _ in _written(adapter, data, result, names)}
    second = {c for c, _, _ in _written(adapter, data, again, names)}
    failures = []
    for column in sorted(first | second):
        if column not in first or column not in second:
            failures.append(f"determinism: '{column}' is written in only one of two runs")
        elif not _same(adapter.column_values(result, column), adapter.column_values(again, column)):
            failures.append(f"determinism: '{column}' differs between two runs")
    return failures


def _not_applicable_reason(cls: type[FeatureGroup]) -> str | None:
    if is_nominal_subclass(cls, SourceFeatureGroup):
        return "not applicable to source groups"
    if inspect.getattr_static(cls, "calculate_feature") is inspect.getattr_static(FeatureGroup, "calculate_feature"):
        return "not applicable, the group has no calculate_feature"
    return None


def _availability_reason(cls: type[FeatureGroup], adapter: FrameworkAdapter) -> str | None:
    framework = adapter.compute_framework()
    rule = cls.compute_framework_rule()
    if rule is not None and framework not in rule:
        return f"{framework.__name__} excluded by compute_framework_rule"
    if not adapter.is_available():
        return f"{framework.__name__} is not available"
    return None


def _no_implementation_reason(cls: type[FeatureGroup], adapter: FrameworkAdapter) -> str | None:
    if is_nominal_subclass(cls, OperationFeatureGroup) and _serving_implementation(cls, adapter) is None:
        return f"no implementation of {cls.__name__} serves {adapter.compute_framework().__name__}"
    return None


def _framework_gate_reason(cls: type[FeatureGroup], adapter: FrameworkAdapter) -> str | None:
    return _availability_reason(cls, adapter) or _no_implementation_reason(cls, adapter)


def _route_block(
    cls: type[FeatureGroup], adapter: FrameworkAdapter, name: str, options: Options
) -> tuple[str | None, str | None]:
    """(block, raised): why the planner would not route name to the adapter's framework; a raising hook is not a block."""
    framework = adapter.compute_framework()
    split = probe_match_criteria(cls, name, options._fork(), resolve_parts=True).part
    hook_name, part = split if split is not None else (name, None)
    routed, error = safe_field_with_error(
        lambda: cls.supports_compute_framework(hook_name, options, framework, part=part), True
    )
    if error is not None:
        return None, f"'{name}' supports_compute_framework or accepts raised: {error}"
    if not routed:
        return f"'{name}' is not routed to {framework.__name__} by {cls.__name__}", None
    impl = _serving_implementation(cls, adapter)
    found, error = safe_field_with_error(
        lambda: unallowed_variants(cls, impl, hook_name, options, part), tuple[Divergence, ...]()
    )
    if error is not None:
        return None, f"'{name}' variant check raised: {error}"
    if found:
        return f"'{name}': {variant_block_reason(cls, {framework: found}, {framework: impl}, options)}", None
    return None, None


def _input_problems(cls: type[FeatureGroup], names: list[str], base: Options, data: TypedTestData) -> list[str]:
    missing = sorted(_declared_inputs(cls, names, base) - set(data.types))
    if missing:
        return [f"declared input '{m}' is missing from input_data" for m in missing]
    return [f"requested name '{n}' is already in input_data" for n in names if n in data.types]


def _run_checks(
    cls: type[FeatureGroup],
    names: list[str],
    base: Options,
    data: TypedTestData,
    adapter: FrameworkAdapter,
    notes: list[str],
    require_run: bool,
) -> list[str]:
    reason = _not_applicable_reason(cls) or _availability_reason(cls, adapter)
    if reason is not None:
        notes.append(f"run: {reason}")
        return []
    reason = _no_implementation_reason(cls, adapter)
    if reason is not None:
        if require_run:
            return [f"run: {reason}"]
        notes.append(f"run: {reason}")
        return []
    failures: list[str] = []
    remaining = []
    for name in names:
        block, raised = _route_block(cls, adapter, name, base)
        if raised is not None:
            failures.append(f"run: {raised}")
        elif block is not None:
            if require_run:
                failures.append(f"run: {block}")
            else:
                notes.append(f"run: {block}")
        else:
            remaining.append(name)
    if not remaining:
        return failures
    names = remaining
    problems = _input_problems(cls, names, base, data)
    if problems:
        return failures + [f"outputs: {p}" for p in problems]
    result, error = _run(cls, adapter, data, names, base)
    if error is not None:
        return [*failures, f"outputs: run raised: {error}"]
    failures += _check_outputs(cls, adapter, data, names, base, result)
    expected = _expected_types(cls, names, base)
    typed = {column: expected[name] for column, name, _ in _written(adapter, data, result, names) if name in expected}
    failures += _check_types(adapter, typed, result, notes)
    if data.types:
        failures += _check_zero_rows(cls, adapter, data, names, base, result, expected)
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
    require_run: bool = False,
) -> CheckResult:
    """Check a FeatureGroup's match contract and, given input_data, its run; a raising matcher is a non-match.

    With require_run, a missing serving Implementation or a route block is a failure instead of a note.
    """
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
        failures += _run_checks(cls, run_names, base, input_data, adapter or PythonDictAdapter(), notes, require_run)
    return CheckResult(check="feature_group_shape", subject=cls.__name__, failures=tuple(failures), notes=tuple(notes))


def assert_feature_group_shape(
    cls: type[FeatureGroup],
    base_options: Options | None = None,
    *,
    input_data: TypedTestData | None = None,
    adapter: FrameworkAdapter | None = None,
    require_run: bool = True,
) -> None:
    """Fail the running pytest test if the shape check reports failures; require_run defaults to True here."""
    assert_passed(
        check_feature_group_shape(cls, base_options, input_data=input_data, adapter=adapter, require_run=require_run)
    )
