# PROPERTY_MAPPING Configuration

`PROPERTY_MAPPING` declares the options a feature group accepts, and how each one is
validated. This page is the single source of truth for that model: what a spec may
contain, which invariant is checked at which moment, what runs against end-user
values in what order, and what each failure produces.

Two rules carry most of the model:

1. **The spec declares the arity, not the caller.** Whether the user writes a list, a
   tuple, a set, or a frozenset never changes what a validator receives.
2. **A rejected value is not always an error.** Some failures raise; others just mean
   "this feature group does not match", which lets a different candidate win.

## The lifecycle: which invariant fires when

| Moment | Mechanism | Checks | Receives | On failure |
| --- | --- | --- | --- | --- |
| Import time (`property_spec(...)` call) | Authoring invariants | strict needs `allowed_values` or an `element_validator`; `allowed_values` without strict; empty `allowed_values`; validators are callable | The spec being built | `ValueError` at import |
| Class definition (`FeatureGroup.__init_subclass__`) | Removed-key guard | No spec uses a removed key (`validation_function`, `type_validator`) | Every spec in the mapping | `ValueError` naming the replacement |
| Class definition | `check_declared_default` | A strict, non-`None` `default` is accepted by its own key | The declared default | `ValueError` at class definition |
| Match time (parser) | `allowed_values` membership | Each element is in the accepted set | One element | `ValueError`, surfaced to the end user |
| Match time (parser) | `element_validator` | Each element satisfies a predicate | One element | `ValueError`, surfaced to the end user |
| Match time (mixin) | `required_when` | A conditionally required option is present | `Options` | Non-match (`False`) |
| Match time (mixin) | `match_guard` | The whole value has an acceptable shape | The raw value | Non-match (`False`) |
| Match time (mixin) | `MIN/MAX_IN_FEATURES` | In-feature count is within bounds | The in-features | Non-match (`False`) |

Match-time mechanisms run in that order. `element_validator` **replaces** membership
rather than adding to it: when a key declares one, `allowed_values` is not consulted.
Because the parser runs before the mixin, an element rejected by membership or by
`element_validator` short-circuits, and `match_guard` is never reached.

## Choosing a mechanism

| You want to say | Use |
| --- | --- |
| "This option accepts exactly these values" | `allowed_values` + `strict_validation: True` |
| "Each value must satisfy a rule I cannot enumerate" (positive int, float in range) | `element_validator` (requires strict) |
| "The value as a whole must have this shape" (a dict, a list of exactly 3, an ordering) | `match_guard` |
| "This option is required only when another option says so" | `required_when` |

The two callables differ on both axes, which is what their names now say:

- **`element_validator`** sees **one element**, only under `strict_validation`, and a
  falsy return **raises** `ValueError`. The user gave a value the group claims to own
  but cannot accept, so they are told.
- **`match_guard`** sees the **raw whole value**, with or without strict, and a falsy
  return is a plain **non-match**. The group is saying "not mine", so resolution moves
  on and another feature group may still take the feature.

Both must be pure functions. They may be called several times during resolution (once
per candidate feature group).

## What a validator receives

Container syntax is not part of the contract. For membership and `element_validator`,
every sequence unpacks element-wise and identically:

``` python
# All four hand the element_validator exactly "a" and then "b".
Options(context={"ops": ["a", "b"]})
Options(context={"ops": ("a", "b")})
Options(context={"ops": {"a", "b"}})
Options(context={"ops": frozenset({"a", "b"})})
```

- A `str` is a **scalar**, not a sequence of characters: `"abc"` is the single element
  `"abc"`.
- A `dict` is **one composite value**, not a sequence of its keys. Validating its shape
  is `match_guard`'s job.
- Elements keep their real type: an `int` element arrives as `1`, never `"1"`.
- An **empty** container is *present* and vacuously valid: it has no elements, so no
  validator runs, and it still satisfies the required-presence check. This is distinct
  from the option being absent.

`match_guard` is unaffected by any of this: it always receives the raw value with its
original container type.

## Authoring a spec

Three forms build the same plain dict. The contract is the dict, so all three are
equivalent to the parser.

**Recommended, explicit `allowed_values`:** keeps the value space separate from the
flags, so a doc-only key can never widen an accepted set.

``` python
"operation_type": {
    DefaultOptionKeys.allowed_values: {"add": "Addition", "sub": "Subtraction"},
    DefaultOptionKeys.context: True,
    DefaultOptionKeys.strict_validation: True,
}
```

`allowed_values` may be a value-to-docstring mapping or a re-iterable collection, but
not a one-shot iterator (a generator would exhaust and behave like an empty set).

**Builder, `property_spec`:** the same dict, with the authoring invariants checked at
construction instead of at class definition.

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

It also takes `element_validator`, `match_guard`, and `required_when`, emitted only
when provided. Its declared-default check is not a separate implementation: it calls
the same `FeatureChainParser.check_declared_default` the class-definition hook uses,
so the two cannot drift.

**Legacy flattened form:** allowed values share one dict with the metadata flags, and
the parser recovers the value space by subtracting `RESERVED_PROPERTY_KEYS`. Still
supported, but the explicit form is preferred precisely because subtraction is easy to
get wrong.

``` python
"parameter_name": {
    "value1": "Description of value1",
    "value2": "Description of value2",
    DefaultOptionKeys.context: True,
    DefaultOptionKeys.strict_validation: True,
}
```

## Parameter classification

``` python
# Context parameter: does not affect Feature Group splitting
"aggregation_type": {"sum": "Sum aggregation", DefaultOptionKeys.context: True}

# Group parameter: affects Feature Group splitting
"data_source": {"production": "Production data", DefaultOptionKeys.group: True}
```

## Declared defaults must be honored

Under `strict_validation: True`, a declared `default` must be a value the key accepts.
This is enforced at class-definition time, naming the class, key, default, and accepted
values, which closes the gap where omitting the key would apply an unrevalidated
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

Fix it by adding the default to the accepted values, or by removing it (a key with no
default and no `required_when` is unconditionally required). `required_when` does NOT
exempt a key: when its predicate returns `False` and the key is omitted, the default
still applies, so an unaccepted default would still slip through. A `default` of `None`
is always legal (the "unset" sentinel), and the check is a no-op when strict is off.

A validator that *raises* when called with the declared default is reported distinctly
from one that merely *rejects* it, with the original exception chained as `__cause__`.

## What the end user sees on a rejection

A direct `FeatureChainParser` call raises `ValueError` immediately. Going through
`FeatureChainParserMixin.match_feature_group_criteria` (the default for most feature
groups) is different: it catches that `ValueError` and returns `False`, because during
resolution one candidate's "no match" must not abort the search for another candidate
that might still accept the feature.

If every candidate rejects the feature, the final "No feature groups found" error
collects the discarded reasons and appends them:

```
Feature group(s) rejected an option value while matching 'window_size_windowed':
  - WindowedFeatureGroup: Property value '14' failed validation for 'window_size'
```

This is diagnostic only. It does not change the `True`/`False` contract, so a value
rejected by one feature group can still match another, and multi-group fallback
resolution keeps working. Authors get this for free.

## Conditional requirements with `required_when`

Attach a predicate to declare an option required only under certain conditions. It
receives an effective `Options` and returns `True` when the option is required. This
works for config-based and string-based features alike (for the latter, the operation
parsed from the feature name is merged in first, so predicates see values from both
sources).

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

When the predicate returns `True` and the option is absent, the match fails; entries
with `required_when` are otherwise optional. The predicate must be a pure, callable
`(Options) -> bool` that does not raise. Non-callable values are skipped with a
warning; non-bool truthy returns count as `True`.

## Migrating from the removed key names

`validation_function` and `type_validator` are **removed**, with no aliases. They did
not communicate their difference, and `type_validator` additionally collided by
substring with the unrelated `DataTypeValidator`.

| Removed | Use |
| --- | --- |
| `validation_function` | `element_validator` |
| `type_validator` | `match_guard` |

Both routes fail loudly, so no spec silently changes meaning: the old enum members are
gone (`AttributeError` at import), and a spec still carrying the old string key raises
at class definition, naming the replacement.

One semantic change comes with the rename. `element_validator` now genuinely receives
one element per call for every container type; previously a sequence was stringified
and arrived as, for example, `"('a', 'b')"`. A validator written to inspect a whole
container (`isinstance(x, list) and all(...)`) must therefore either become a
per-element rule, or move to `match_guard` if it really is a whole-value check.

## Where each invariant is pinned

| Invariant | Test |
| --- | --- |
| Renamed keys, removed-key guard, shared default seam, precedence | `tests/.../feature_chainer/test_property_mapping_unified_model.py` |
| Container invariance, no stringification, str-as-scalar, dict-as-composite, empty containers | `tests/.../feature_chainer/test_property_mapping_sequence_unpacking.py` |
| Plugin specs behave identically across containers | `tests/test_plugins/feature_group/experimental/test_property_mapping_container_invariance.py` |
| Declared-default invariant at class definition | `tests/test_core/test_abstract_plugins/test_property_mapping_default_invariant.py` |
| Rejection reasons surfaced to the end user | `tests/test_core/test_prepare/test_identify_feature_group_error_message.py` |
| `property_spec` authoring invariants | `tests/.../feature_chainer/test_property_spec_builder.py` |

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

A forwarded key also splits FeatureSets by value. A context key delivered via the
push (`propagate_context_keys`) or the pull (`inherit_context_keys`) enters the
child's `inherited_context_keys`, and features whose value for that key differs (e.g.
`env=prod` vs `env=staging`) then compute in separate FeatureSets, one per value;
equal values still share one computation. The split is per-FeatureGroup and by value,
not by provenance: once any feature in that scope forwards the key, every feature in
scope splits on it, including a sibling that set the same key directly in plain
context. A key that no feature in scope ever forwards stays metadata and does not
split. This isolates per-value results that previously collapsed into one computation
where a single value silently won.

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
