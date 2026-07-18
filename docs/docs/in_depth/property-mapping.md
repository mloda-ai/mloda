# PROPERTY_MAPPING Configuration

`PROPERTY_MAPPING` maps every option key a feature group accepts to a `PropertySpec`: the
typed, frozen spec that declares how that option is validated. This page is the single
source of truth for that model: what a spec may contain, which invariant is checked at
which moment, what runs against end-user values in what order, and what each failure
produces.

Two rules carry most of the model:

1. **The spec declares the arity, not the caller.** Whether the user writes a list, a
   tuple, a set, or a frozenset never changes what a validator receives.
2. **A rejected value is not always an error.** Some failures raise; others just mean
   "this feature group does not match", which lets a different candidate win.

## The spec type

``` python
from mloda.provider import PropertySpec

PROPERTY_MAPPING = {
    "operation_type": PropertySpec(
        "Arithmetic operation",
        allowed_values={"add": "Addition", "sub": "Subtraction"},
        strict_validation=True,
        default="add",
    ),
}
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `explanation` | `str` | required, positional | What the option means. Also names the spec in construction errors. |
| `allowed_values` | `Mapping[Any, str]`, any other iterable, or `None` | `None` | The declared value space. A Mapping is `{value: description}` and is kept as given; any other iterable is materialized to a tuple. |
| `default` | `Any` | `NO_DEFAULT` | Materialized into runtime options when the key is absent (see [Applying declared defaults](#applying-declared-defaults)). Leaving it at `NO_DEFAULT` declares *no default*, which makes the key required. See [Optional keys](#optional-keys). A caller's explicit `None` is treated as absent, so this default replaces it, unless the key sets `allow_explicit_none=True`. |
| `context` | `bool` | `True` | `True`: context parameter. `False`: group parameter, which splits feature groups. |
| `strict_validation` | `bool` | `False` | Enforce the value space at match time. |
| `element_validator` | `Callable \| None` | `None` | Per-element predicate. Requires `strict_validation=True`. |
| `match_guard` | `Callable \| None` | `None` | Whole-value predicate. A falsy return is a non-match. |
| `required_when` | `Callable \| None` | `None` | `(Options) -> bool`: the key is required only when it returns truthy. |
| `allow_explicit_none` | `bool` | `False` | Opt-in so an explicit `None` is honored (not treated as absent) and flows through validation. |
| `deferred_binding` | `bool` | `False` | Exempts a required key from the string-named path presence check only; its value is bound outside match-time name capture. Not optionality: the key stays required on the config path. See [Required presence on the string-named path](#required-presence-on-the-string-named-path). |

**`property_spec(...)` is the authoring path to reach for.** It is a thin builder over the
same fields, its keyword is `strict=` (which sets `strict_validation`), and it keeps the
declaration readable. `PropertySpec` is the lower-level form: it is the type the builder
returns and the type the schema is validated against, so reach for it directly when you need
the class itself (subclassing, `isinstance` checks, or constructing a spec programmatically).
Both are exported from `mloda.provider`.

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

The type IS the schema. A raw dict spec is rejected at class definition
(`... is a dict, not a PropertySpec`), and a field name that does not exist is a plain
constructor `TypeError` that mypy already flags where the spec is written. Nothing a spec
does not understand can be absorbed silently.

## The lifecycle: which invariant fires when

| Moment | Mechanism | Checks | Receives | On failure |
| --- | --- | --- | --- | --- |
| Author time | `mypy --strict` | The field exists and its declared type fits: `strict_validaton=True` (typo), `strict_validation=1`, `allowed_values=5` | The constructor call | mypy error at the spec literal. Without mypy: an unknown field is a `TypeError`; a wrong type falls through to the row below |
| Construction (`PropertySpec(...)`) | `__post_init__` | `allowed_values` is not a str/bytes and is a Mapping or an iterable; `strict_validation` is a real bool; the validators are callable; `element_validator` implies strict; strict has a non-empty value space or an `element_validator`; a strict, non-`None` `default` is accepted by the key's own rules | The spec being built | `ValueError` at import, prefixed `PropertySpec('<explanation>')` |
| Class definition (`FeatureGroup.__init_subclass__`) | Spec type | Every spec IS a `PropertySpec` | Every value in the mapping | `ValueError` naming the class and the key |
| Match time (parser) | `allowed_values` membership | Each element of a **present** option is in the accepted set | One element | `ValueError`, surfaced to the end user |
| Match time (parser) | `element_validator` | Each element of a **present** option satisfies a predicate | One element | `ValueError`, surfaced to the end user |
| Match time (parser) | Required presence (config path) | A key that declares no `default` and no `required_when` was provided | The options | Non-match (`False`) |
| Match time (parser) | Required presence (string-named path) | Same, after declared defaults and name bindings resolve; `deferred_binding=True` and the source (`in_features`) key are exempt | The effective options | Warn or non-match per `MLODA_NAME_PATH_REQUIRED_PRESENCE` (`warn` default / `enforce` / `off`) |
| Match time (mixin) | `match_guard` | The whole value has an acceptable shape | The raw value | Non-match (`False`) |
| Match time (mixin) | `MIN/MAX_IN_FEATURES` | In-feature count is within bounds | The in-features | Non-match (`False`) |
| Match time (guard installed at class definition) | `required_when` | A conditionally required option is present | `Options` | Non-match (`False`) |
| Class definition (mixin) | Universal-matcher diagnostic | An all-optional `PROPERTY_MAPPING` inherits the configuration matcher, so it matches any name with empty options | The class | `logger.warning`, unless `ALLOW_UNIVERSAL_MATCHER = True` |

A spec is constructed inside the class body, so its own rules fire before the class exists.
Class definition is left with exactly one rule: the type itself. Within `__post_init__` the
declared-default check runs last, after the shape rules, because it calls the validators the
shape rules just certified: a non-callable `element_validator` would otherwise be blamed on
the default.

Match-time mechanisms run in table order. `element_validator` **replaces** membership rather
than adding to it: when a key declares one, `allowed_values` is not consulted. Because the
parser runs before the mixin, an element rejected by membership or by `element_validator`
short-circuits, and `match_guard` is never reached.

Value validation does not depend on how the feature was created: membership and
`element_validator` run on **both** match paths, the configuration-based one and the
string-named one. Required **presence** now runs on both too, differing only in strength: the
configuration path rejects a missing required key outright, while the string-named path is
warn-by-default and becomes a non-match only under `MLODA_NAME_PATH_REQUIRED_PRESENCE=enforce`
(see [Required presence on the string-named path](#required-presence-on-the-string-named-path)).
A key the name encodes, a `deferred_binding=True` key, a declared-default key, and the source
(`in_features`) key are never falsely reported missing. So `"income__pca_2d"` with no options at
all still matches, while the same feature group with `pca_svd_solver="bogus"` in its options is
rejected either way.

`required_when` is the one mechanism that does not live inside a matcher. A class that
declares it gets its resolved `match_feature_group_criteria` wrapped at class definition,
and the wrapper runs the predicates after that matcher returns `True`. Overriding the
matcher therefore keeps the contract, whether the override delegates or not.

## Choosing a mechanism

| You want to say | Use |
| --- | --- |
| "This option accepts exactly these values" | `allowed_values` + `strict_validation=True` |
| "Each value must satisfy a rule I cannot enumerate" (positive int, float in range) | `element_validator` (requires strict) |
| "The value as a whole must have this shape" (a dict, a list of exactly 3, an ordering) | `match_guard` |
| "This option is required only when another option says so" | `required_when` |

### Positive-integer options: use the shared predicate

Do not hand-roll a positive-integer check. Python bools satisfy `isinstance(value, int)`, so
the obvious predicate accepts `horizon=True`; and `str.isdigit()` accepts `"²"`, which then
raises in `int()`. `mloda.provider` exports one predicate that gets both right, and accepts
numpy integers and decimal strings:

```python
from mloda.provider import PropertySpec, is_positive_int

WINDOW_SIZE: PropertySpec(
    "Size of the time window (must be positive integer)",
    context=True,
    strict_validation=True,
    element_validator=is_positive_int,
)
```

It accepts positive Python ints, numpy integers and decimal strings, and rejects `bool`,
zero, negatives, and non-integers. Shipped plugins (`window_size`, `horizon`, `k_value`,
`dimension`) all use it, so the value space stays consistent across feature groups. If a key
allows an extra sentinel, compose rather than fork it, as clustering does for `"auto"`:

```python
element_validator=lambda value: value == "auto" or is_positive_int(value),
```

The two callables differ on both axes, which is what their names say:

- **`element_validator`** sees **one element**, only under `strict_validation`, and a falsy
  return **raises** `ValueError`. The user gave a value the group claims to own but cannot
  accept, so they are told.
- **`match_guard`** sees the **raw whole value**, with or without strict, and a falsy return
  is a plain **non-match**. The group is saying "not mine", so resolution moves on and
  another feature group may still take the feature. On a spec that also sets
  `strict_validation=True` the guard means "this value is wrong", so its rejection is reported to
  the user instead of failing silently.

Both run on both match paths, so declaring both on one spec is about **what** is judged, not
about where: `element_validator` judges each element and produces the message, while
`match_guard` judges the raw container that element-wise unpacking would otherwise hide.

Both must be pure functions. They may be called several times during resolution (once per
candidate feature group).

## What a validator receives

Container syntax is not part of the contract. For membership and `element_validator`, every
sequence unpacks element-wise and identically:

``` python
# All four hand the element_validator exactly "a" and then "b".
Options(context={"ops": ["a", "b"]})
Options(context={"ops": ("a", "b")})
Options(context={"ops": {"a", "b"}})
Options(context={"ops": frozenset({"a", "b"})})
```

- A `str` is a **scalar**, not a sequence of characters: `"abc"` is the single element
  `"abc"`.
- A `dict` is **one composite value**, not a sequence of its keys. Validating its shape is
  `match_guard`'s job.
- Elements keep their real type: an `int` element arrives as `1`, never `"1"`.
- An **empty** container is *present* and vacuously valid: it has no elements, so no
  validator runs, and it still satisfies the required-presence check. This is distinct from
  the option being absent.

`match_guard` is unaffected by any of this: it always receives the raw value with its
original container type.

## The shape of `allowed_values`

A Mapping is the value-space-with-descriptions form and is kept as given. Any other iterable
(tuple, list, set, frozenset, generator) is materialized to a **tuple** at construction, so a
generator is consumed once, up front, and can never make matching stateful. Two shapes raise:

- a **`str` or `bytes`**, which a forgotten comma produces (`("add")` is the str `"add"`, not
  a one-tuple). Membership would silently degrade into a substring test, accepting `"a"`,
  `"ad"` and `""`.
- a **scalar**, for which `value in 5` raises a `TypeError` that the match path swallows into
  a silent reject-everything.

The shape is checked whether or not the spec is strict, because a non-strict value space is
still *consumed*: it maps a value parsed out of a feature name back onto its
`PROPERTY_MAPPING` key. It is a mapping aid there, never an enforcement.

A spec that declares no `allowed_values` declares an **empty** value space; the space is never
inferred from the spec's other fields.

## Strict validation needs a value space

`strict_validation=True` needs something to validate against: a non-empty `allowed_values`, or
an `element_validator`. With neither, the accepted set is empty and the key rejects every
value, so the spec could never match. The constructor refuses to build it.

An `element_validator` **replaces** membership, so when a key declares one, `allowed_values` is
never consulted at match time and this rule does not apply: even an empty `allowed_values` is
inert there, because the validator, not membership, decides.

## Declared defaults must be honored

Under `strict_validation=True`, a declared non-`None` `default` must be a value the key
accepts, either by membership or through the `element_validator`. This closes the gap where
omitting the key would apply an unrevalidated default.

``` python
# ValueError at construction: "mul" is not accepted.
PropertySpec(
    "Arithmetic operation",
    allowed_values={"add": "Addition", "sub": "Subtraction"},
    strict_validation=True,
    default="mul",
)
```

Fix it by adding the default to the accepted values, or by removing it. `required_when` does
NOT exempt a key: when its predicate returns `False` and the key is omitted, the default still
applies, so an unaccepted default would still slip through. A declared `default=None` applies
no value and is always legal, and the check is a no-op when strict is off.

A validator that *raises* when called with the declared default is reported distinctly from
one that merely *rejects* it, with the original exception chained as `__cause__`.

## Optional keys

Optionality is the `default` field, and the three states are distinct:

| Written | Meaning |
| --- | --- |
| `default` omitted (stays `NO_DEFAULT`) | The key declares no default and is **required**. |
| `default=None` | The key is **optional**; no value is applied when it is absent. |
| `default=<value>` | The key is **optional**; the value is materialized when it is absent (see [Applying declared defaults](#applying-declared-defaults)), and is checked under strict. |

``` python
from mloda.provider import PropertySpec

PROPERTY_MAPPING = {
    "pipeline_steps": PropertySpec(
        "List of pipeline steps as (name, transformer) tuples",
        default=None,  # optional: pipeline_name is the alternative
    ),
}
```

`NO_DEFAULT` is a sentinel exported from `mloda.provider`; it exists because on a dataclass
every field is always present, so a plain `None` cannot say both "no default declared" and
"optional with no value". The retired dict form drew the same line with a *present*
`default: None` entry. `SklearnPipelineFeatureGroup` is the in-repo example.

To ask whether a spec declares a default, use `is_no_default`, also exported from
`mloda.provider`:

``` python
from mloda.provider import is_no_default

if is_no_default(spec.default):
    ...  # the key declares no default, so it is required
```

Do **not** write `spec.default is NO_DEFAULT`. `is_no_default` is a type test rather than an
identity test on purpose: a second imported copy of the module (an editable install alongside
site-packages, or an `importlib.reload`) carries its own sentinel object, and an identity
check would read that copy's sentinel as a *declared* default.

`required_when` is for a *conditional* requirement, not for optionality: never write a
predicate that always returns `False` to make a key optional.

`deferred_binding` is not optionality either: the key stays required on the configuration path,
and the flag only defers its string-named path presence check (see
[Required presence on the string-named path](#required-presence-on-the-string-named-path)).

## Applying declared defaults

A declared default is metadata until the resolved feature group materializes it.
`FeatureGroup.options_with_defaults(options)` returns an `Options` view where every absent key
with a concrete declared default is filled from its spec (into context or group per the spec).
A present value is never overridden, even a falsy `0`/`False`/`""`. `NO_DEFAULT` and
`default=None` fill nothing. A strict default is already validated at construction, so the
materialized value is not re-checked.

The framework applies this centrally at the compute boundary: `run_calculate_feature`
materializes the declared defaults into the `FeatureSet` before `calculate_feature` and any
calculate-feature extender run, so a plugin reads `feature.options` directly.
`options_with_defaults` remains for pre-materialization contexts (e.g. `resolve_subtype`
internals). Materialization happens *after* resolution, so it never changes matching or
FeatureSet splitting.

``` python
graph_type = feature.options.get("graph_type")  # the declared default when the caller omitted it
```

`Options.get(key, default)` reads dict-style: a present key returns its stored value (even a falsy
`0`/`False`/`""`), and the call-site `default` is returned only when the key is absent from both
group and context. Because the framework has already materialized any declared default before
`calculate_feature` runs, reading it back gives a three-level precedence, an explicit caller value
first, then the declared spec default, then the call-site `default`. (Subtlety: materialization fills
any key whose value `is None`, so an explicit `None` is replaced by a concrete spec default, while the
other falsy values `0`/`False`/`""` are kept. That "None means absent" rule is the default; a spec
opts a single key out with `allow_explicit_none=True`, after which an explicit `None` overrides even a
concrete default and is honored across the match-time value mechanisms: it counts as present for the
required-presence and `required_when` checks and is seen by membership, `element_validator`, and
`match_guard`, while every flagless spec is unchanged. The flag defaults to `False`, so the existing
`is not None` presence tests keep working.)

``` python
graph_type = feature.options.get("graph_type", "spring")  # explicit value; else the spec default; else "spring"
```

## Parameter classification

There is no `group` field: a group parameter is `context=False`.

``` python
PROPERTY_MAPPING = {
    # Context parameter (the default): does not affect feature group splitting
    "aggregation_type": PropertySpec(
        "Aggregation to apply",
        allowed_values={"sum": "Sum aggregation"},
        strict_validation=True,
    ),
    # Group parameter: affects feature group splitting
    "data_source": PropertySpec(
        "Data source to read from",
        allowed_values={"production": "Production data"},
        context=False,
        strict_validation=True,
    ),
}
```

A caller who places the key explicitly in `Options(group=...)` or `Options(context=...)`
overrides the spec's classification.

## How a name-parsed value binds to a key

A value captured from the feature name binds to a PROPERTY_MAPPING key by name: a named capture
group `(?P<key>...)` binds to the key of the same name, so a secondary capture and an
`element_validator`-only spec (one with no `allowed_values`) both receive their value. When a
pattern declares any named group, binding is exclusively by name. A pattern with only positional
groups falls back to the legacy rule of binding the first capture to the single key whose
`allowed_values` already contain it. That fallback is transitional (retired by #772); a positional
pattern whose keys share a reachable value is rejected at class-definition time, so migrate such a
pattern to named capture groups.

## Required presence on the string-named path

Required presence is checked when a feature matches by its name, not on the configuration path
alone. A key is flagged when it declares no `default`, no `required_when`, and
`deferred_binding=False`, and is still absent after declared defaults and name captures are
resolved. Exempt from the check: a declared default, a `required_when` key, a
`deferred_binding=True` key, and the source key (`in_features`), whose presence the name prefix
supplies and whose count `MIN/MAX_IN_FEATURES` enforces.

`MLODA_NAME_PATH_REQUIRED_PRESENCE` selects the mode (case-insensitive; unset or invalid means
`warn`):

| Value | Effect |
| --- | --- |
| `warn` (default) | Logs a warning naming the group and the missing key(s), still **matches**. The migration default. |
| `enforce` | The missing key makes it a **non-match**. |
| `off` (or `0`, `false`, `no`) | The check is skipped. |

Under `enforce`, the resolution-failure report also names the missing key(s), not only the log.

Two migrations remove the warning for a flagged key. Give the pattern a named capture
`(?P<key>...)` so the framework binds the key from the name; or, for a key bound outside
match-time name capture (parsed by the plugin from the name, or supplied downstream), set
`deferred_binding=True`, which exempts it from this check only and leaves it required on the
config path. `TimeWindowFeatureGroup` marks its name-parsed `window_size` and `time_unit` keys
this way:

``` python
from mloda.provider import PropertySpec

PROPERTY_MAPPING = {
    "window_size": PropertySpec(
        "Window size parsed from the feature name by the plugin",
        deferred_binding=True,  # exempt from the string-named presence check only
    ),
}
```

## What the end user sees on a rejection

A direct `FeatureChainParser` call raises `ValueError` immediately. Going through
`FeatureChainParserMixin.match_feature_group_criteria` (the default for most feature groups)
is different: it catches that `ValueError` and returns `False`, because during resolution one
candidate's "no match" must not abort the search for another candidate that might still accept
the feature. This holds on both paths: a bad option value on a string-named feature is the same
non-match as on a configuration-based one.

If every candidate rejects the feature, the final "No feature groups found" error collects the
discarded reasons and appends them:

```
Feature group(s) rejected an option value while matching 'window_size_windowed':
  - WindowedFeatureGroup: Property value '14' failed validation for 'window_size'
```

This is diagnostic only. It does not change the `True`/`False` contract, so a value rejected by
one feature group can still match another, and multi-group fallback resolution keeps working.
Authors get this for free.

## Conditional requirements with `required_when`

Attach a predicate to declare an option required only under certain conditions. It receives an
effective `Options` and returns `True` when the option is required. This works for
config-based and string-based features alike (for the latter, the operation parsed from the
feature name is merged in first, so predicates see values from both sources).

``` python
_ORDER_DEPENDENT = {"first", "last"}

def _needs_order_by(options: Options) -> bool:
    return options.get("aggregation_type") in _ORDER_DEPENDENT

PROPERTY_MAPPING = {
    "aggregation_type": PropertySpec(
        "Aggregation to apply",
        allowed_values={
            "sum": "Sum", "avg": "Average", "first": "First", "last": "Last",
        },
        strict_validation=True,
    ),
    "order_by": PropertySpec(
        "Column to order by within each partition",
        required_when=_needs_order_by,
    ),
}
```

When the predicate returns `True` and the option is absent, the match fails; entries with
`required_when` are otherwise optional. The predicate must be pure and must not raise. It is
callable by construction, and a non-bool truthy return counts as `True`.

The enforcement is installed on the class, not on one matcher, so overriding
`match_feature_group_criteria` does not lose it: the predicates still run after the override
returns `True`. Nested guards (an override that delegates into an already guarded parent) do
not stack: only the outermost one evaluates, so the predicates run exactly once per match call.
The matcher must be a `classmethod`; a `staticmethod` matcher on a class that declares
`required_when` is rejected at class definition.

The guard is installed at class definition, so mutating `PROPERTY_MAPPING` or replacing
`match_feature_group_criteria` after the class body escapes it.

## Guarding against a universal configuration matcher

A key is unconditionally required only when it declares no `default` and no `required_when`. A
`PROPERTY_MAPPING` with only declared-default keys (and, as the degenerate case, an empty mapping)
has no such key, so on the configuration path it matches any feature name with empty options. A
feature group that inherits the mixin's `match_feature_group_criteria` and declares such a mapping
is a universal matcher: it claims features it was never meant to.

At class definition the mixin warns about this, naming the class and the escape hatch. A key that is
unconditionally required, or conditionally required via `required_when`, gates the match, so the
mapping is not warned. For the remaining all-default mappings, universality is confirmed by calling
the resolved matcher with an unrelated, separator-free name and empty options: a genuinely
discriminating `match_feature_group_criteria` is not warned, while a pass-through override that only
delegates to the base still is.

Set `ALLOW_UNIVERSAL_MATCHER = True` on the class to declare the universal match intentional and
silence the warning. Otherwise give one key no `default` (making it unconditionally required), or a
`required_when` predicate that fires when the option is absent.

## Migrating from the dict form

Spec dicts are gone. Every value in a `PROPERTY_MAPPING` must be a `PropertySpec`; an
unmigrated spec raises at class definition, naming the class, the key, and the remedy.

| Retired | Now |
| --- | --- |
| Any spec dict, including the flattened form (accepted values sharing one dict with the flags) | `PropertySpec(...)`, with accepted values under `allowed_values` |
| `validation_function` | `element_validator` |
| `type_validator` | `match_guard` |
| A present `default: None` entry marking a key optional | the `default=None` field (an omitted `default` means the key is required) |
| `strict_validation` as a spec-dict key | the `strict_validation` field (`strict=` on `property_spec`) |

Misspelled and retired field names need no dedicated machinery any more: they are keyword
arguments that do not exist, so the constructor raises `TypeError` at the line where the spec
is written, and mypy reports them before the code runs.

One semantic change came with the callable renames. `element_validator` receives one element
per call for every container type; the old `validation_function` saw a stringified sequence
(`"('a', 'b')"`). A validator written to inspect a whole container
(`isinstance(x, list) and all(...)`) must become a per-element rule, or move to `match_guard`
if it really is a whole-value check.

## Where each invariant is pinned

| Invariant | Test |
| --- | --- |
| `PropertySpec` construction: defaults, immutability, `allowed_values` normalization, flag and callable rules, strict needs a value space | `tests/.../feature_chainer/test_property_spec_type.py` |
| Unknown or mistyped field caught by mypy at author time | `tests/.../feature_chainer/test_property_spec_author_time.py` |
| `PROPERTY_MAPPING` accepts nothing but `PropertySpec`, and the pipeline behaves as before | `tests/.../feature_chainer/test_property_spec_hard_break.py` |
| The constructor IS the schema (no key set left to curate) | `tests/.../feature_chainer/test_property_mapping_spec_schema.py` |
| Shape rules, relocated from class definition to construction | `tests/.../feature_chainer/test_property_mapping_spec_shape.py` |
| Declared-default invariant | `tests/test_core/test_abstract_plugins/test_property_mapping_default_invariant.py` |
| Declared defaults materialized into runtime options | `tests/test_core/test_abstract_plugins/test_options_with_defaults.py` |
| Declared defaults materialized at the compute boundary | `tests/test_core/test_abstract_plugins/test_materialize_defaults_boundary.py` |
| Container invariance, no stringification, str-as-scalar, dict-as-composite, empty containers | `tests/.../feature_chainer/test_property_mapping_sequence_unpacking.py` |
| Present option values validated on the string-named path too | `tests/.../feature_chainer/test_name_path_validates_option_values.py` |
| Required presence on the string-named path: warn/enforce/off modes, and the `deferred_binding` / `in_features` exemptions | `tests/.../feature_chainer/test_name_path_required_presence.py` |
| `required_when` survives an overridden matcher, runs exactly once, and demands a classmethod | `tests/.../feature_chainer/test_required_when_enforced_on_override.py` |
| Plugin specs behave identically across containers | `tests/test_plugins/feature_group/experimental/test_property_mapping_container_invariance.py` |
| `property_spec` builder surface | `tests/.../feature_chainer/test_property_spec_builder.py` |
| Rejection reasons surfaced to the end user | `tests/test_core/test_prepare/test_identify_feature_group_error_message.py` |
| The all-optional universal-matcher diagnostic and its `ALLOW_UNIVERSAL_MATCHER` escape hatch | `tests/.../feature_chainer/test_universal_optional_matcher.py` |

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
