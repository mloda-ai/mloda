# PROPERTY_MAPPING Configuration

`PROPERTY_MAPPING` defines parameter validation and classification for modern
feature groups via the unified parser.

## Basic structure

``` python
from mloda.provider import DefaultOptionKeys

PROPERTY_MAPPING = {
    "parameter_name": {
        "value1": "Description of value1",
        "value2": "Description of value2",
        DefaultOptionKeys.context: True,           # classification (see below)
        DefaultOptionKeys.strict_validation: True,  # validation mode
    },
    DefaultOptionKeys.in_features: {
        "explanation": "Source feature description",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
    },
}
```

In this flattened form the allowed values share one dict with the metadata flags.
The parser recovers the value space by subtracting the reserved metadata keys
(`RESERVED_PROPERTY_KEYS`: `explanation`, `allowed_values`, `default`, `context`,
`group`, `strict_validation`, `validation_function`, `required_when`,
`type_validator`). It stays fully supported.

## Recommended: explicit `allowed_values`

Declaring the value space under `DefaultOptionKeys.allowed_values` keeps it
separate from the flags, so a doc-only key can never widen an allowed set:

``` python
"operation_type": {
    DefaultOptionKeys.allowed_values: {"add": "Addition", "sub": "Subtraction"},
    DefaultOptionKeys.context: True,
    DefaultOptionKeys.strict_validation: True,
}
```

When present, the parser uses `allowed_values` directly and ignores other non-flag
keys. It may be a value-to-docstring mapping or a re-iterable collection (list,
tuple, set), but not a one-shot iterator (a generator would exhaust, behaving like
an empty set; `property_spec` materializes iterables for you). `allowed_values` and
`explanation` are reserved names and cannot be literal accepted values.

### Builder: `property_spec`

`property_spec` builds the same dict and validates its invariants at construction
(strict needs a non-empty `allowed_values` or a `validation_function`;
`allowed_values` without strict is a no-op; a strict `default` must be accepted).
The contract stays a plain dict, so this is optional sugar:

``` python
from mloda.provider import property_spec

PROPERTY_MAPPING = {
    "operation_type": property_spec(
        "Arithmetic operation",
        strict=True,
        allowed_values={"add": "Addition", "sub": "Subtraction"},
        default="add",
    ),
}
```

Three optional callable passthroughs are emitted only when provided:

- `validation_function`: requires `strict=True`; then strict no longer needs
  `allowed_values`, and a declared `default` is checked through the function. If it
  raises on the default, the builder wraps it as a `ValueError` (original chained as
  `__cause__`), mirroring core's class-definition check.
- `required_when`: conditional-requirement predicate; no strict requirement.
- `type_validator`: raw-value shape check; no strict requirement.

## Parameter classification

``` python
# Context parameter: does not affect Feature Group splitting
"aggregation_type": {"sum": "Sum aggregation", DefaultOptionKeys.context: True}

# Group parameter: affects Feature Group splitting
"data_source": {"production": "Production data", DefaultOptionKeys.group: True}
```

## Validation modes

`strict_validation` defaults to `False` (listed values are illustrative). Set it to
`True` to accept only listed values.

**`validation_function`** (requires `strict_validation=True`) validates individual
parsed elements; the parser unpacks lists and calls it per element:

``` python
"window_size": {
    "explanation": "Size of time window",
    DefaultOptionKeys.validation_function: lambda x: isinstance(x, int) and x > 0,
    DefaultOptionKeys.strict_validation: True,
}
```

**`type_validator`** checks the shape of the raw value before any list unpacking. It
does not require `strict_validation` and must return truthy for the match to
succeed:

``` python
def _is_list_of_strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

"partition_by": {
    "explanation": "List of columns to partition by",
    DefaultOptionKeys.context: True,
    DefaultOptionKeys.type_validator: _is_list_of_strings,
}
```

When both are present, `validation_function` runs first (per parsed element), then
`type_validator` on the raw value. Both must be pure functions.

## Default values

``` python
"method": {"linear": "Linear", "cubic": "Cubic", DefaultOptionKeys.default: "linear"}
```

### Declared defaults must be honored

Under `strict_validation: True`, a declared `default` must be a value the key
accepts. mloda enforces this at class-definition time: a `PROPERTY_MAPPING` whose
strict `default` is outside the accepted set (or fails its `validation_function`)
raises `ValueError` immediately, naming the class, key, default, and accepted
values. This closes a gap where omitting the key would apply an unrevalidated
default.

``` python
# Rejected at class definition: "mul" is not accepted.
"operation_type": {
    "add": "Addition",
    "sub": "Subtraction",
    DefaultOptionKeys.strict_validation: True,
    DefaultOptionKeys.default: "mul",
}
```

Resolve it by adding the default to the accepted values, or removing it (a key with
no default and no `required_when` is unconditionally required). `required_when` does
NOT exempt a key: when its predicate returns `False` and the key is omitted, the
default still applies, so an unaccepted default would still slip through. A `default`
of `None` is always legal (the "unset" sentinel), and the check is a no-op when
`strict_validation` is `False`.

## Usage in feature groups

``` python
class MyFeatureGroup(FeatureGroup):
    PROPERTY_MAPPING = {
        "operation_type": {
            "sum": "Sum operation",
            "avg": "Average operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {"explanation": "Source feature", DefaultOptionKeys.context: True},
    }

    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name, options, property_mapping=cls.PROPERTY_MAPPING
        )
```

``` python
Options(context={"operation_type": "sum"})     # valid: "sum" is in mapping
Options(context={"operation_type": "custom"})  # strict: raises ValueError
Options(context={"in_features": "any_name"})   # flexible: any value allowed
```

## Conditional requirements with `required_when`

Attach a predicate callable to declare an option required only under certain
conditions. It receives an effective `Options` and returns `True` when the option is
required. This works for both config-based and string-based features (for the
latter, the operation parsed from the feature name is merged in first, so predicates
see values from both sources).

``` python
_ORDER_DEPENDENT = {"first", "last"}

def _needs_order_by(options: Options) -> bool:
    return options.get("aggregation_type") in _ORDER_DEPENDENT

PROPERTY_MAPPING = {
    "aggregation_type": {
        "sum": "Sum", "avg": "Average", "first": "First", "last": "Last",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
    },
    "order_by": {
        "explanation": "Column to order by within each partition",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.required_when: _needs_order_by,
    },
}
```

When the predicate returns `True` and the option is absent,
`match_feature_group_criteria` returns `False`; entries with `required_when` are
otherwise treated as optional. The predicate must be a pure, callable
`(Options) -> bool` that does not raise. Non-callable values are skipped with a
warning; non-bool truthy returns count as `True`.

## Context propagation

By default context parameters are local: they do not flow through dependency chains,
which is correct for feature-specific config. For cross-cutting metadata (session
IDs, environment flags), use `propagate_context_keys`:

``` python
Options(
    context={"session_id": "abc", "window_function": "sum"},
    propagate_context_keys=frozenset({"session_id"}),  # only session_id flows on
)
```

Only listed keys propagate; the rest stay local. Group options differ: they forward
to input features by default. A feature group keeps consumer-local keys off its
upstreams with `forward_group_exclude`, an allowlist, or `forward_group=False` on the
child `Feature`. `propagate_context_keys` is the caller-side **push**; its
counterpart is the child-side **pull** `inherit_context_keys`. For both author-side
flows see
[Forwarding Options to Input Features](feature-chain-parser.md#forwarding-options-to-input-features).

### `group` and `context` are independent namespaces

`group` (which drives resolution and splitting) and `context` (metadata) are
separate namespaces. Within one `Options` a key may not live in both. Across a
consumer and its input feature, default forwarding is group-to-group only: a consumer
`context` key is not compared against a same-named child `group` key, so the two are
independent roles, not a conflict:

``` python
consumer = Options(context={"algo": "sum"})
child = Options(group={"algo": "mean"})

child.inherit_from(consumer)          # resolves silently, no ValueError
assert child.group["algo"] == "mean"  # child keeps its own group value
```

This holds for both differing and equal values, but only on that default path. Once
a context key is explicitly forwarded (pushed via `propagate_context_keys` or pulled
via `inherit_context_keys`) it enters the child's context, and a same-named child
`group` key then raises a cross-category conflict. The reverse of the default path (a
consumer **group** key forwarded onto a child holding it in **context**) also raises.
Keep such a key off the child with `forward_group_exclude`, an allowlist, or
`forward_group=False`.
