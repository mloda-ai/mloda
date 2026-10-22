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

```python
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
| `expected` | `str \| None` | `None` | What the `match_guard` accepts, as a phrase completing "must be ...". Makes a guard rejection a reported reason on a non-strict spec. Requires `match_guard`. |
| `required_when` | `Callable \| None` | `None` | `(Options) -> bool`: the key is required only when it returns truthy. |
| `allow_explicit_none` | `bool` | `False` | Opt-in so an explicit `None` is honored (not treated as absent) and flows through validation. |
| `deferred_binding` | `bool` | `False` | Exempts a required key from the string-named path presence check only; its value is bound outside match-time name capture. Not optionality: the key stays required on the config path. See [Required presence on the string-named path](#required-presence-on-the-string-named-path). |

**`property_spec(...)` is the authoring path to reach for.** It is a thin builder over the
same fields, its keyword is `strict=` (which sets `strict_validation`), and it keeps the
declaration readable. `PropertySpec` is the lower-level form: it is the type the builder
returns and the type the schema is validated against, so reach for it directly when you need
the class itself (subclassing, `isinstance` checks, or constructing a spec programmatically).
Both are exported from `mloda.provider`.

```python
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
| Construction (`PropertySpec(...)`) | `__post_init__` | `allowed_values` is not a str/bytes and is a Mapping or an iterable; `strict_validation` is a real bool; the validators are callable; `element_validator` implies strict; strict has a non-empty value space or an `element_validator`; a strict, non-`None` `default` is accepted by the key's own rules; `expected` is a non-empty str and needs a `match_guard` | The spec being built | `ValueError` at import, prefixed `PropertySpec('<explanation>')` |
| Class definition (`FeatureGroup.__init_subclass__`) | Mapping and spec type | The mapping is `None` or a Mapping (a format group's must be a Mapping, mixins included); every spec IS a `PropertySpec` | The mapping, then every value in it | `TypeError` naming the class for a non-mapping; `ValueError` naming the class and the key for a bad spec |
| Match time (parser) | `allowed_values` membership | Each element of a **present** option is in the accepted set | One element | `ValueError`, surfaced to the end user |
| Match time (parser) | `element_validator` | Each element of a **present** option satisfies a predicate | One element | `ValueError`, surfaced to the end user |
| Match time (parser) | Required presence (config path) | A key that declares no `default` and no `required_when` was provided | The options | Non-match (`False`) |
| Match time (parser) | Required presence (string-named path) | Same, after declared defaults and name bindings resolve; `deferred_binding=True` and the source (`in_features`) key are exempt | The name-bound options | Non-match (`False`), with a warning naming the missing key(s); for a `context=True` key the reason also names the remedy (`Options(context=...)`; for a direct input feature the consumer's `propagate_context_keys`, for an input further down a chained name the requested feature's `cascade_context_keys`, or a group option) |
| Match time (plain group) | Present values, `match_guard`, required presence | Values and `match_guard` in the default matcher of every plain group; required presence on the raw options of a group without a pattern (a patterned plain group keeps the name-path rule) | The options | Non-match (`False`), with the reason recorded (for `match_guard`, only when the spec is strict or declares `expected`) |
| Match time (mixin) | `match_guard` | The whole value has an acceptable shape | The raw value | Non-match (`False`), reported when the spec is strict or declares `expected` |
| Match time (mixin) | `MIN/MAX_IN_FEATURES` | In-feature count is within bounds | The in-features | Non-match (`False`) |
| Match time (guard installed at class definition) | `required_when` | A conditionally required option is present | `Options` | Non-match (`False`) |
| Class definition (mixin) | Universal-matcher diagnostic | An all-optional `PROPERTY_MAPPING` inherits the configuration matcher, so it matches any name once `in_features` supplies a source | The class | `logger.warning`, unless `ALLOW_UNIVERSAL_MATCHER = True` |
| Class definition (mixin) | Missing-`in_features` diagnostic | A dict `PROPERTY_MAPPING` declares no `in_features` key while `MIN_IN_FEATURES >= 1`, and the group inherits `input_features` and the matcher | The class | `logger.warning`, silenced by `MIN_IN_FEATURES = 0` or an `in_features` key |

A spec is constructed inside the class body, so its own rules fire before the class exists.
Class definition is left with the type rules alone. Within `__post_init__` the
declared-default check runs last, after the shape rules, because it calls the validators the
shape rules just certified: a non-callable `element_validator` would otherwise be blamed on
the default.

Match-time mechanisms run in table order. `element_validator` **replaces** membership rather
than adding to it: when a key declares one, `allowed_values` is not consulted. Because the
parser runs before the mixin, an element rejected by membership or by `element_validator`
short-circuits, and `match_guard` is never reached.

Value validation does not depend on how the feature was created: membership and
`element_validator` run on **both** match paths, the configuration-based one and the
string-named one. Required **presence** runs on both too: the string-named path rejects a
missing required key exactly like the configuration path does
(see [Required presence on the string-named path](#required-presence-on-the-string-named-path)).
A key the name encodes, a `deferred_binding=True` key, a declared-default key, and the source
(`in_features`) key are never falsely reported missing. So `"income__pca_2d"` with no options at
all still matches, while the same feature group with `pca_svd_solver="bogus"` in its options is
rejected either way.

`required_when` is the one mechanism that does not live inside a matcher. A class that
declares it gets its resolved `match_feature_group_criteria` wrapped at class definition,
and the wrapper runs the predicates after that matcher returns `True`. Overriding the
matcher therefore keeps the contract, whether the override delegates or not.

### A plain feature group

A `FeatureGroup` without `FeatureChainParserMixin` (the `DefaultMatcherFeatureGroup` probe in
`tests/.../feature_chainer/test_required_when_enforced_on_override.py` is the in-repo example)
matches by name and input data, so its `PROPERTY_MAPPING` is enforced on the raw options:

- On a group that declares no `PREFIX_PATTERN` or `SUFFIX_PATTERN`, a key that declares no `default`, no
  `required_when` and no `deferred_binding=True` must be present, or the group is a non-match. Declare
  `default=None` on a key that is optional. The guard is installed from `FeatureGroup.__init_subclass__`
  around the class's resolved `match_feature_group_criteria`, so an override keeps it. A plain group with a
  pattern keeps the name-path rule instead (see
  [Required presence on the string-named path](#required-presence-on-the-string-named-path)).
- `required_when` is enforced for absence by its own guard, so an **absent** required key is a non-match at
  match time, not a late `ValueError` inside `input_features`. Requiredness reads presence
  (`options.get(key) is not None`), so a present-but-falsy value (a required key passed as `[]` or
  `""`) satisfies the requirement and matches; a group that needs a non-empty value checks it itself in
  `input_features`.
- Present values are validated (`strict_validation`, `element_validator`) and `match_guard` runs, in the
  default `match_feature_group_criteria` of every plain group. A matcher override keeps them by delegating
  via `super().match_feature_group_criteria(...)`; a widening override that skips `super()` keeps them with `cls.passes_option_declarations(options)`.
- Declared defaults stay metadata until the group materializes them itself by calling
  `options_with_defaults` at its own read site. That call is what makes a declared default real at an
  `input_features` read site (see [Applying declared defaults](#applying-declared-defaults)).

## Validating a plain dict

`validate_property_values(values, mapping, *, closed_world)` (from `mloda.provider`) checks a plain dict, such as a
credentials dict, against a `PropertySpec` mapping. It raises `PropertyValidationError` (a `ValueError` with `key`
and `reason`) on the first failure.

- It checks required presence, `required_when`, and strict values.
- `required_when` runs only when the key is absent.
- Ignored: `match_guard`, `deferred_binding`, and `context`. A caller that must reject a collection checks the value's shape itself.
- `closed_world=True` rejects undeclared keys; `False` ignores them.
- Defaults are never applied: `required_when` receives an `Options` of the raw values, unlike on a feature group.
- Errors and logs never contain a value.
- A raise marked with `escalate_match_abort` (from an `element_validator` or a value's `__eq__`) propagates unchanged instead of becoming a `PropertyValidationError`.

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
  the user instead of failing silently. Declaring `expected` reports the rejection too, even on a
  non-strict spec, with a message naming what the guard accepts instead of the generic strict text.

A key that must reject a collection outright uses a `match_guard` on the raw value (as `HANDLE_SPEC` does).

Both run on both match paths, so declaring both on one spec is about **what** is judged, not
about where: `element_validator` judges each element and produces the message, while
`match_guard` judges the raw container that element-wise unpacking would otherwise hide.

Both must be pure functions. They may be called several times during resolution (once per
candidate feature group).

## What a validator receives

Container syntax is not part of the contract. For membership and `element_validator`, every
sequence unpacks element-wise and identically:

```python
from mloda.user import Options

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

```py
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

```python
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

```py
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
materialized value is not re-checked. A key the feature name binds never receives a declared default;
`FeatureGroup.name_bindings(name)` returns those bound values (the captured text, a `str`).

The framework applies this at two sites. At feature **intake**
(`Engine.add_feature_to_collection`) a feature's options are rebound through
`options_with_defaults` as it enters the collection, which fills declared defaults together with the
name-bound value for keys still absent (bound values go into context, or into group only when the key
already sits there as `None`). At the **compute boundary** `run_calculate_feature` repeats the step on
the `FeatureSet` before `calculate_feature` and any calculate-feature extender run, so a plugin reads
`feature.options` directly. Bound fills go to context because the `Options` hash is group-only: a name-only
request and the same name with an agreeing explicit context option merge into one feature at intake,
while an explicit group option stays its own feature. `options_with_defaults` remains for
pre-materialization contexts (e.g. `resolve_subtype` internals) and for read sites the framework hands
pre-default options, `input_features` above all.

For an engine-driven request, intake has already filled everything, so the compute-boundary call is a
no-op. Direct `FeatureSet` use that bypasses the engine gets the fill there, and collapsing twins raise
instead of merging.

A materialized default or bound value is never a declared option, so it never conflicts with a
name-bound value: the name-agreement check reads the caller's own keys only.

Both sites run *after* resolution, so materialization never changes **resolution matching** (filter
matching runs after intake and does observe materialized values, see below). It does change
how features **group**: intake materialization deliberately canonicalizes default-equivalent twins,
so a feature that passes a declared default explicitly and one that omits it become equal and merge
into a single feature (with a warning naming the duplicated request; twins differing only in a name-bound value merge the same way) instead of computing twice.

One consequence for authors: `input_features` is called with the DECLARED, pre-default options (the
engine stashes them before intake rebinds, and a child inherits the same pre-default options), so a
declared default does NOT reach an `input_features` read site. A group that wants it there calls
`options_with_defaults` itself (see
[A plain feature group](#a-plain-feature-group)).

Which stage sees which view of the options:

| Stage | Options view |
| --- | --- |
| Parse, bind, resolution match, resolve (subtype resolution applies defaults internally) | declared (pre-default) |
| `input_features()` and child option inheritance | declared (pre-default) |
| Splitting, planning, filter matching, `validate_input_features`, compute (`calculate_feature` and its extenders) | effective (post-default, name-bound values filled) |

The splitting row assumes intake has run, which is every engine-driven request. On the direct `FeatureSet`
path the safety net materializes inside `run_calculate_feature`, which is after
`validate_input_features`, not before it.

Filter matching is the one place the same classmethod sees both views: `GlobalFilter` calls
`match_feature_group_criteria` again after intake, passing the filter feature's own options
enriched from the resolved feature's effective ones (`unify_options`), so a key the filter feature
omits arrives materialized where feature resolution saw it declared. A key the host's name binds is not
copied into the filter feature: that value belongs to the host's name, not to a shared configuration.

```py
graph_type = feature.options.get("graph_type")  # the declared default when the caller omitted it
```

`Options.get(key, default)` reads dict-style: a present key returns its stored value (even a falsy
`0`/`False`/`""`), and the call-site `default` is returned only when the key is absent from both
group and context. Because the framework has already materialized any declared default before
`calculate_feature` runs, reading it back gives a four-level precedence: the caller's explicit value
(which must agree with the name), then the name-bound value, then the declared spec default, then the
call-site `default`. (Subtlety: materialization fills any key whose value `is None`, so an explicit
`None` is replaced by a concrete spec default, while the other falsy values `0`/`False`/`""` are kept. That "None means absent" rule is the default; a spec
opts a single key out with `allow_explicit_none=True`, after which an explicit `None` overrides even a
concrete default and is honored across the match-time value mechanisms: it counts as present for the
required-presence and `required_when` checks and is seen by membership, `element_validator`, and
`match_guard`, while every flagless spec is unchanged. The flag defaults to `False`, so the existing
`is not None` presence tests keep working.)

```py
graph_type = feature.options.get("graph_type", "spring")  # explicit value; else the spec default; else "spring"
```

Plugin code can read a declared key with `cls.read_option(key, options, feature_name)`, which raises for an undeclared key and fills the declared default or name-bound value; engine keys such as `in_features` still go through `options.get`.

## Parameter classification

There is no `group` field: a group parameter is `context=False`.

```python
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

A name owns the feature for a group when its `PREFIX_PATTERN` identifies the group, and an owned
name is authoritative for every value it encodes. Every named capture group `(?P<key>...)` of an
owned name that is a PROPERTY_MAPPING key binds to that key and is validated like an option, even
when the option is also present; an unsupported value is a recorded match-time rejection. This
covers a secondary capture and an `element_validator`-only spec (one with no `allowed_values`).

A pattern with only positional groups falls back to the legacy rule: the first capture binds only
to the single key whose `allowed_values` already contain it. A positional pattern whose keys share
a reachable value is rejected at class-definition time, so migrate it to named capture groups.

A declared option for a key the name binds, and a non-empty `in_features`, must agree with the
name; a contradiction aborts the match (`MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1` downgrades it to a
warning). Only the caller's own keys are compared, so a materialized default never conflicts. See
[Name Ownership and Agreement](feature-chain-parser.md#name-ownership-and-agreement).

## Required presence on the string-named path

Required presence is checked when a feature matches by its name, not on the configuration path
alone. A key is flagged when it declares no `default`, no `required_when`, and
`deferred_binding=False`, and is still absent after declared defaults and name captures are
resolved. Exempt from the check: a declared default, a `required_when` key, a
`deferred_binding=True` key, and the source key (`in_features`), whose presence the name prefix
supplies and whose count `MIN/MAX_IN_FEATURES` enforces (an absent `in_features` counts as zero on the configuration path).

A plain `FeatureGroup` (no mixin) without a pattern is checked the same way on its raw options, once a matching
rule has accepted the name. A missing key records the reason and logs at debug level, so no warning is emitted per
candidate probe. A plain group with a pattern keeps the name-path rule above, and a name no pattern owns keeps the
default matcher's verdict. Migration: a bare `property_spec("...")` on a plain group is now required; declare
`default=None` for an optional key.

A flagged missing key makes the match a **non-match**: a warning names the group, the feature,
and the missing key(s), and the resolution-failure report names the missing key(s) too. For missing
keys declared `context=True` the reason also names the remedy (`Options(context=...)`; for a direct input
feature the consumer's `propagate_context_keys`, for an input further down a chained name the requested
feature's `cascade_context_keys`, or a group option),
naming those keys when several are missing. A group option is part of feature identity; see
[Context propagation](#context-propagation).

Two migrations remove the warning for a flagged key. Give the pattern a named capture
`(?P<key>...)` so the framework binds the key from the name (as `ClusteringFeatureGroup` does for
`algorithm` and `k_value`); or, for a key bound outside match-time name capture (parsed by the
plugin from the name, or supplied downstream), set `deferred_binding=True`, which exempts it from
this check only and leaves it required on the config path:

```python
from mloda.provider import PropertySpec

PROPERTY_MAPPING = {
    "tenant": PropertySpec(
        "Tenant supplied downstream, not captured from the name",
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

If every candidate rejects the feature, the final "No feature groups found" error names each
discarded candidate and why it dropped:

```
Feature group(s) eliminated while matching 'window_size_windowed':
  - WindowedFeatureGroup (option value): Property value int 14 failed validation for 'window_size'
```

This is diagnostic only. It does not change the `True`/`False` contract, so a value rejected by
one feature group can still match another, and multi-group fallback resolution keeps working.
Authors get this for free.

### Naming what a guard expects

A `match_guard` rejection is silent by default unless the spec is strict. Declare `expected`
to report it anyway, with a message naming what the guard accepts:

```python
from mloda.provider import property_spec


def _is_concurrency(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


PROPERTY_MAPPING = {
    "concurrency": property_spec(
        "Parallel workers", match_guard=_is_concurrency, expected="a whole number of 1 or more"
    ),
}
```

```text
  - WorkerPoolFeatureGroup (option value): option 'concurrency' must be a whole number of 1 or more, got str '4'
```

The `expected` reason above, the strict match_guard reason (`Property value <text> rejected by
match_guard for '<key>'`), `strict_validation`'s `failed validation` and `not found in mapping`
messages all follow
the same echo rule: they echo the rejected value only when its type is exactly `str`, `int`,
`float` or `bool`, with its text cut to about 30 characters, or when it is `None`. Any other value,
a `str` subclass, a numpy scalar, or an int too large to safely print, is named by its type only.
All of these reasons can reach the "No feature groups found" error and the filter near-miss
warnings, so do not declare `expected` or `strict_validation` on a key that can carry a secret such
as a token or a connection string.

## Conditional requirements with `required_when`

Attach a predicate to declare an option required only under certain conditions. It receives an
effective `Options` and returns `True` when the option is required. This works for
config-based and string-based features alike (for the latter, the operation parsed from the
feature name is merged in first, so predicates see values from both sources).

```python
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

When the predicate returns `True` and the option is absent, the match fails and the non-match records
a rejection reason naming the owning class and the missing key, so the resolution-failure report
carries a near-miss line for it; entries with `required_when` are otherwise optional. For a key
declared with `context=True` (the default), the reason also names the remedy
(`Options(context=...)`; for a direct input feature the consumer's `propagate_context_keys`, for an
input further down a chained name the requested feature's `cascade_context_keys`, or a group option). The predicate must be pure and must not raise. It
is callable by construction, and a non-bool truthy return counts as `True`.

The enforcement is installed on the class, not on one matcher, so overriding
`match_feature_group_criteria` does not lose it: the predicates still run after the override
returns `True`. Nested guards (an override that delegates into an already guarded parent) do
not stack: only the outermost one evaluates, so the predicates run exactly once per match call.
The matcher must be a `classmethod`; a `staticmethod` matcher on a class that declares
`required_when` is rejected at class definition. A plain function, callable instance, or
`functools.partial` (no `classmethod` or `staticmethod` descriptor) is rejected on any class that
carries this guard or the name-path presence guard.

The guard is installed at class definition, so mutating `PROPERTY_MAPPING` or replacing
`match_feature_group_criteria` after the class body escapes it.

This guard, the name-path presence guard, and the class-definition diagnostics below live in
`feature_chain_author_guards.py`, which imports `feature_chain_parser` and never the reverse.

## Guarding against a universal configuration matcher

A key is unconditionally required only when it declares no `default` and no `required_when`. A
`PROPERTY_MAPPING` with only declared-default keys (and, as the degenerate case, an empty mapping)
has no such key, so on the configuration path it matches any feature name once `in_features` supplies a
source. A feature group that inherits the mixin's `match_feature_group_criteria` and declares such a
mapping is a universal matcher: it claims features it was never meant to.

At class definition the mixin warns about this, naming the class and the escape hatch. A key that is
unconditionally required, or conditionally required via `required_when`, gates the match, so the
mapping is not warned. For the remaining all-default mappings, universality is confirmed by calling
the resolved matcher with an unrelated, separator-free name and a synthetic `in_features` source: a genuinely
discriminating `match_feature_group_criteria` is not warned, while a pass-through override that only
delegates to the base still is.

Set `ALLOW_UNIVERSAL_MATCHER = True` on the class to declare the universal match intentional and
silence the warning. Otherwise give one key no `default` (making it unconditionally required), or a
`required_when` predicate that fires when the option is absent.

## Declaring the source contract

A mixin group whose `PROPERTY_MAPPING` has no `in_features` key and whose `MIN_IN_FEATURES` is 1 or more
counts an absent `in_features` as zero sources, so it matches by options only when the caller passes
`in_features`. At class definition the mixin warns, naming the class, unless the group overrides
`input_features` or `match_feature_group_criteria` (a pass-through matcher override that delegates to the
mixin is exempt too), or has a `PREFIX_PATTERN`/`SUFFIX_PATTERN` (the source can come from the name). It warns
once per hierarchy: a subclass of a class that already warned stays silent. The option path records no rejection reason for this case,
so the resolution failure report does not name it; the warning is the diagnostic.

Fix it with `MIN_IN_FEATURES = 0` if the group is source-less, or declare an `in_features` key in
`PROPERTY_MAPPING` (with a `default` if the group should match without one).

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
| Declared defaults materialized at intake, canonicalizing default-equivalent twins | `tests/test_core/test_abstract_plugins/test_intake_default_canonicalization.py` |
| Filter criteria matching observes effective options | `tests/test_core/test_filter/test_filter_criteria_effective_options.py` |
| The shared declaration mechanics: the per-key validator | `tests/.../test_components/test_declaration_surface.py` |
| A pattern-less group: enforced `required_when`, and defaults it applies itself | `tests/.../feature_chainer/test_required_when_enforced_on_override.py`, `tests/test_core/test_abstract_plugins/test_intake_default_canonicalization.py` |
| A plain group enforces required keys, strict values and `match_guard`, also through an override | `tests/.../feature_chainer/test_name_path_required_presence.py`, `test_strict_validation_returns_false.py`, `test_required_when_enforced_on_override.py` |
| A plain group's rejection reasons reach the failure report | `tests/test_core/test_prepare/test_first_pass_rejection_recording.py` |
| Container invariance, no stringification, str-as-scalar, dict-as-composite, empty containers | `tests/.../feature_chainer/test_property_mapping_sequence_unpacking.py` |
| Present option values validated on the string-named path too | `tests/.../feature_chainer/test_name_path_validates_option_values.py` |
| Required presence on the string-named path: the mandatory non-match, the retired env var stays ignored, and the `deferred_binding` / `in_features` exemptions | `tests/.../feature_chainer/test_name_path_required_presence.py` |
| A declared option or `in_features` that contradicts an owned name aborts the match, and the env var downgrades it | `tests/test_core/test_abstract_plugins/test_components/test_forwarded_name_mismatch.py` |
| Empty operands and the source count on the name path | `tests/.../feature_chainer/test_in_feature_count_gate_name_sources.py` |
| `required_when` survives an overridden matcher, runs exactly once, and demands a classmethod | `tests/.../feature_chainer/test_required_when_enforced_on_override.py` |
| A `required_when` non-match records its reason, so the failure report names the key and its owner | `tests/test_core/test_prepare/test_required_when_rejection_recording.py` |
| `required_when` end to end, including the context-key remedy in the failure report | `tests/.../feature_chainer/test_property_mapping_required_when.py` |
| Plugin specs behave identically across containers | `tests/test_plugins/feature_group/experimental/test_property_mapping_container_invariance.py` |
| `property_spec` builder surface | `tests/.../feature_chainer/test_property_spec_builder.py` |
| Rejection reasons surfaced to the end user | `tests/test_core/test_prepare/test_identify_feature_group_error_message.py` |
| The all-optional universal-matcher diagnostic and its `ALLOW_UNIVERSAL_MATCHER` escape hatch | `tests/.../feature_chainer/test_universal_optional_matcher.py` |
| The missing-`in_features` definition-time diagnostic and its zero-source rejection | `tests/.../feature_chainer/test_missing_in_features_declaration.py` (the diagnostic), `tests/.../feature_chainer/test_in_feature_count_gate_name_sources.py` (the option-path zero-source non-match records no rejection) |
| `expected` reports a `match_guard` rejection on a non-strict spec, with the capped value echo | `tests/test_core/test_prepare/test_first_pass_rejection_recording.py` |

## Context propagation

By default context parameters are local: they do not flow through dependency chains,
which is correct for feature-specific config. For cross-cutting metadata (session
IDs, environment flags), use `propagate_context_keys`:

```python
Options(
    context={"session_id": "abc", "window_function": "sum"},
    propagate_context_keys=frozenset({"session_id"}),  # only session_id flows on
)
```

Only listed keys propagate; the rest stay local. The push is one hop: listed keys land
in the context of the feature's direct input features, and an input feature does not
push them on to its own inputs. For a name-chained feature such as `a__b__c`, a key
pushed from the requested feature reaches `a__b`, not the root `a`. To continue the
push down the chain, opt in with `cascade_context_keys`:

```python
Options(
    context={"session_id": "abc"},
    cascade_context_keys=frozenset({"session_id"}),  # session_id reaches every level
)
```

- Every input that receives a cascade key pushes it on to its own inputs, down to the
  root, also through name-chained inputs such as `a__b__c`. The key must exist in
  `context`; it need not be in `propagate_context_keys`.
- The key splits FeatureSets by value at every level: each distinct value computes
  every input again, including one root read per value. Equal values share one.
- An input holding a different value, the key as a group option, or a name-bound
  value that differs raises a `ValueError` naming the input and `cascade_context_keys`.
- `forward_group=False` on an input stops the cascade there and cuts group forwarding
  to it; there is no per-key stop. A pull (`inherit_context_keys`) on it still
  delivers the value at that level only.
- An input with inputs of its own raises a `ValueError` when it is reached with different
  `cascade_context_keys` than it had when its inputs were resolved, for example when one
  consumer cascades a key and another delivers it with a one-hop push or a pull, or when
  a shared `Feature` object gains the cascade after an earlier request resolved it. Its
  inputs are resolved once. Declare the same `cascade_context_keys` on every request
  reaching it, and list the cascading request first when `Feature` objects are shared.
  A shared root feature is exempt.
- Two requests of one input that differ in compute framework resolve their inputs separately.
  When only one of them cascades a key, the plan fails with an error that the input is read
  from two steps whose options differ. Declare the same `cascade_context_keys` on both.
- The value sits in the options of every input down the chain, so every feature group
  there sees it, and it is pickled to MULTIPROCESSING workers.
- The pull `inherit_context_keys` stays one hop.

Group options remain the choice when the value should be part of feature identity and
resolution at every level, and a feature group author can pull a key level by level
with `inherit_context_keys`.

Group options forward to input features by default. A feature group keeps consumer-local keys off its
upstreams with `forward_group_exclude`, an allowlist, or `forward_group=False` on the
child `Feature`. `propagate_context_keys` (or `cascade_context_keys`) is the caller-side
**push**; its counterpart is the child-side **pull** `inherit_context_keys`. For both author-side
flows see
[Forwarding Options to Input Features](feature-chain-parser.md#forwarding-options-to-input-features).

A forwarded key also splits FeatureSets by value. A context key delivered via the
push (`propagate_context_keys` or `cascade_context_keys`) or the pull
(`inherit_context_keys`) enters the
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

```python
consumer = Options(context={"algo": "sum"})
child = Options(group={"algo": "mean"})

child.inherit_from(consumer)          # resolves silently, no ValueError
assert child.group["algo"] == "mean"  # child keeps its own group value
```

This holds for both differing and equal values, but only on that default path. Once
a context key is explicitly forwarded (pushed via `propagate_context_keys` or
`cascade_context_keys`, or pulled
via `inherit_context_keys`) it enters the child's context, and a same-named child
`group` key then raises a cross-category conflict. The reverse of the default path (a
consumer **group** key forwarded onto a child holding it in **context**) also raises.
Keep such a key off the child with `forward_group_exclude`, an allowlist, or
`forward_group=False`.
