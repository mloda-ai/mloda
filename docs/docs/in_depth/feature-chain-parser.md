# Feature Chain Parser

## Overview

The Feature Chain Parser system enables feature groups to work with both traditional string-based feature names and modern configuration-based feature creation. This unified approach provides flexibility while maintaining backward compatibility.

> **For AI Agents:** Feature chaining enables LLMs to declare complex data pipelines through simple naming conventions. Instead of writing pipeline code, agents can request `user_query__validated__retrieved__pii_redacted` - mloda resolves the full chain automatically.

## Key Concepts

### Separator System

mloda uses three separator characters in feature names, each with a specific purpose:

| Separator | Constant | Purpose | Example |
|-----------|----------|---------|---------|
| `__` | `CHAIN_SEPARATOR` | Separates chained transformations (source→suffix) | `price__mean_imputed` |
| `~` | `COLUMN_SEPARATOR` | Separates multi-column output index | `feature__pca~0` |
| `&` | `INPUT_SEPARATOR` | Separates multiple input features | `point1&point2__distance` |

These constants are available from the `mloda.provider` facade:

```python
from mloda.provider import (
    CHAIN_SEPARATOR,    # "__"
    COLUMN_SEPARATOR,   # "~"
    INPUT_SEPARATOR,    # "&"
)
```

### Feature Chaining

Feature chaining allows feature groups to be composed, where the output of one feature group becomes the input to another. This is reflected in the feature name using the chain separator (`__`):

```
{in_feature}__{operation}
```

For example:
- `sales__sum_aggr` - Simple feature
- `price__mean_imputed__sum_7_day_window__max_aggr` - Chained feature

### Multi-Feature Input

Some feature groups require multiple input features. These are separated using the input separator (`&`):

```
{feature1}&{feature2}__{operation}
```

For example:
- `point1&point2__haversine_distance` - GeoDistance with two points
- `age&income&score__cluster_kmeans_3` - Clustering with multiple features

### Name Ownership and Agreement

One resolution, `FeatureChainParserMixin.resolve_feature_name`, returns a `NameResolution` (`owned`, `bindings`, `sources`, `value_for(key)`). Matching, the rejection diagnostic, `input_features`, `_extract_source_features` and `_resolve_operation` all read it.

- **Ownership:** a name owns the feature when the group's effective name pattern (`PREFIX_PATTERN` or `SUFFIX_PATTERN`) identifies it, and an owned name is authoritative for every value it encodes.
- **Binding:** every named capture of an owned name that is a `PROPERTY_MAPPING` key is bound and validated like an option, even when the option is also set; an unsupported value is a recorded match-time rejection. A key's `name_converter` converts the captured text before validation (a raising one is a rejected value). A legacy positional capture binds only a value already in the key's `allowed_values`, unconverted.
- **Agreement:** a declared option for a key the name binds, and a non-empty `in_features`, must agree with the name (`in_features` must equal the name's direct sources). A contradiction aborts the match with an error; `MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1` downgrades it to a warning. Only own keys are compared: forwarded keys and materialized defaults never conflict here. An agreeing `in_features` supplies the input features, so their options and `feature_group` scope are kept; a downgraded mismatch uses the name's bare sources instead. An override that calls only `match_parser_criteria` skips this check. A plugin that parses a value from the name by hand (one its captures do not bind) runs the option checks (not the `in_features` one) with `cls.validate_hand_parsed_bindings(feature_name, bindings, options)`.
- **Operands:** every name or config source must be non-empty and the count within `MIN_IN_FEATURES` / `MAX_IN_FEATURES` (name path: recorded rejection; config path: silent non-match).

### Options and Configuration-Based Creation

`Options` separates group parameters (they affect feature group resolution and splitting) from context parameters (metadata that does not split); see [Parameter classification](property-mapping.md#parameter-classification). Configuration-based creation (`Feature("placeholder", Options(context={...}))` instead of a chained name) is shown in [Feature Group Matching](feature-group-matching.md#matching-examples).

## FeatureChainParserMixin

The `FeatureChainParserMixin` provides default implementations for common feature chain parsing operations. Feature groups that use feature chaining should inherit from this mixin to reduce boilerplate code.

### Basic Usage

```py
from mloda.provider import FeatureGroup, FeatureChainParserMixin
from mloda.provider import DefaultOptionKeys, PropertySpec

class MyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__my_operation$"

    # In-feature constraints
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1  # Or None for unlimited

    PROPERTY_MAPPING = {
        "operation_type": PropertySpec(
            "Operation to apply",
            allowed_values={"sum": "Sum operation", "avg": "Average operation"},
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec("Source feature"),
    }

    # input_features() inherited from FeatureChainParserMixin
    # match_feature_group_criteria() inherited from FeatureChainParserMixin
```

### Mixin Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `PREFIX_PATTERN` | `str` | Required | Regex pattern for matching feature names |
| `PROPERTY_MAPPING` | `Mapping[str, PropertySpec]` | Required | Parameter validation configuration |
| `MIN_IN_FEATURES` | `int` | `1` | Minimum required in_features (an absent `in_features` counts as zero on the configuration path) |
| `MAX_IN_FEATURES` | `int \| None` | `None` | Maximum allowed in_features (None = unlimited) |
| `IN_FEATURE_SEPARATOR` | `str` | `"&"` | Separator for multiple in_features |
| `RECOGNITION_ONLY_PATTERN` | `bool` | `False` | Declares a captureless pattern as recognition-only (binds no key from the name) |
| `NAME_ONLY_MATCH` | `bool` | `False` | Claim only a name the patterns own; a configuration-path feature (non-matching name resolved via options such as `in_features`) is declined |

Use `NAME_ONLY_MATCH = True` for a group with no discriminator option. An override of `match_feature_group_criteria` must delegate to `super()` or `match_parser_criteria` to keep it. The rejection-reason diagnostic always honors the flag, and code calling `FeatureChainParser.match_configuration_feature_chain_parser` directly passes `config_path=not cls.NAME_ONLY_MATCH`.

`Options.get_in_features()` returns sources in declared order (sets are sorted by name; duplicates are kept). Order-sensitive groups can reject sets with an `in_features` `match_guard=lambda v: not isinstance(v, (set, frozenset))` plus an `expected` text (for example "an ordered list, tuple or single source") so the rejection is reported.

### Groups That Touch Columns

To serve one column operation on several compute frameworks, write an `OperationFeatureGroup` with one `Implementation` per framework: the group owns the loop and each implementation passes its framework callables into it (see [Operation Groups](compute-framework-integration.md#operation-groups-one-group-one-implementation-per-framework)).

### Captureless Patterns and Name Binding

A captureless `PREFIX_PATTERN` (one with no capture group, e.g. `r".*__cleaned_text$"`) binds no
`PROPERTY_MAPPING` key from the feature name. The name identifies the feature group, but every value
comes from options. The old behavior of fabricating an operation token from the suffix text has been
retired.

- To bind a key **from the name**, use a named capture: `r".*__(?P<operation>pca|tsne)_reduce$"`.
- For a **recognition-only** pattern (the name identifies the group but all values come from
  options), set `RECOGNITION_ONLY_PATTERN = True`:

```py
RECOGNITION_ONLY_PATTERN = True
```

Deprecation note: a captureless pattern that also carries a non-empty `PROPERTY_MAPPING` keeps
working, but logs a definition-time warning until it either adds a named capture or sets
`RECOGNITION_ONLY_PATTERN = True`.

### Customization Hooks

#### 1. Custom Validation with `_validate_string_match()`

Override this hook when you need custom validation for string-based feature names:

```py
class MyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, in_feature: str) -> bool:
        """Called only for a name that owns the feature; reject values the group cannot serve."""
        return operation_config.isidentifier()
```

#### 2. Custom `input_features()` Method

Override when you need to add additional input features (e.g., time filter):

```py
class TimeWindowFeatureGroup(TimeReferenceMixin, FeatureChainParserMixin, OperationFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        # The mixin resolves sources from the name or from in_features; add the extras.
        sources = super().input_features(options, feature_name) or set()
        return sources | {Feature(self.get_reference_time_column(options))}
```

The mixin's `input_features` handles both forms: an owned name supplies its sources (an agreeing `in_features` keeps its options and scope), otherwise `in_features` does. Custom code checks source counts with `validate_in_feature_count` (raises) or `in_feature_count_reason` (returns the message or None).

#### 3. Custom `match_feature_group_criteria()` Method

Override for the rules a `PROPERTY_MAPPING` spec cannot express, then delegate. Conditional
requirements belong in `required_when` instead: it is enforced by a guard installed at class
definition, so it still runs on an override.

```py
class SklearnPipelineFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None) -> bool:
        """Mutual exclusivity: pipeline_name and pipeline_steps cannot both be set."""
        if options.get(cls.PIPELINE_NAME) is not None and options.get(cls.PIPELINE_STEPS) is not None:
            return False

        return super().match_feature_group_criteria(feature_name, options, data_access_collection)
```

`match_parser_criteria` calls the parser with the class's `PROPERTY_MAPPING` and patterns and turns a rejected option value into a non-match. Calling `FeatureChainParser` directly from a match hook lets that rejection escape as an exception; the engine contains it as a `match hook` near-miss for that candidate, but the rejection reason is the more useful one. Containment covers plugin raises only: a framework-owned raise, such as a forwarded option value contradicting the feature name, still aborts the whole resolution.

`match_parser_criteria` does not run `match_guard`, the in_features count or name agreement; a widening override keeps the value and guard checks with `cls.passes_option_declarations(options)`.

#### 4. Operation Resolution with `_resolve_operation()`

Use this helper to extract the value of a key from either the feature name or the options, without calling `FeatureChainParser` directly. It is key-aware: an owned name with named captures returns that key's capture, a positional pattern returns its first capture for any key, and otherwise the option value is returned:

```py
class AggregatedFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    AGGREGATION_TYPE = "aggregation_type"

    @classmethod
    def calculate_feature(cls, data, features):
        for feature in features.features:
            # Two-arg form: pass a Feature and the config key
            agg_type = cls._resolve_operation(feature, cls.AGGREGATION_TYPE)
            # ... use agg_type
```

The three-arg form `cls._resolve_operation(feature_name, options, config_key)` is also supported for contexts where the name and options are already separate (e.g., `match_feature_group_criteria`).

In `calculate_feature`, read every value through the mixin helpers (`_resolve_operation`, `_extract_single_source_feature`) so the name stays authoritative when it owns the feature. Multi-source groups use `_extract_validated_source_features`, which enforces `MIN_IN_FEATURES` / `MAX_IN_FEATURES` and rejects empty operands on a direct call.

#### 5. Whole-Value Validation with `match_guard`

Give a spec a `match_guard` callable (with an `expected` phrase to report the rejection) to validate the shape of the raw option value; `element_validator` with `strict_validation=True` judges each element instead. The guard runs only when the option is present (not None, unless the spec sets `allow_explicit_none=True`), and a falsy return is a non-match, not an error. A guard that raises counts as a rejection too, unless the raise is marked with `escalate_match_abort`. Validators must be pure, since they may be called several times during resolution.

The full model, which invariant fires at which moment, the precedence between the two callables, and what a validator receives for each container type, lives in one place: [PROPERTY_MAPPING Configuration](property-mapping.md).

## Forwarding Options to Input Features

The [Context Propagation](property-mapping.md#context-propagation) docs describe the
**caller side**: context options stay local by default, and `propagate_context_keys`
opts specific keys in for the direct input features only (`cascade_context_keys`
continues the push down the chain). This section describes the **author side**: what
happens to a
feature group's own options when it declares an input feature. To preview what an input
will receive, see
[Previewing the forwarded options](property-mapping.md#previewing-the-forwarded-options).

### Forwarding is the default

When your `input_features()` returns a child `Feature`, the engine copies **all** of
the consumer's group options (except `in_features`) onto that child. Configuration set
on a requested feature travels down self-resolving chains (for example
`price__mean_imputed__sum_aggr`) to the end of the chain without any ceremony:

```python
from mloda.provider import FeatureGroup
from mloda.user import Feature, FeatureName, Options


class GraphAnswer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        # The child inherits ALL of the consumer's group options by default.
        return {Feature("knowledge_graph")}
```

Rules:

- `in_features` is never forwarded.
- A key whose spec sets `forward_to_inputs=False` (every credential handle) is not forwarded unless an explicit allowlist names it.
- If the child already carries a forwarded key with an equal value, the merge is a
  no-op for that key.
- If the child already carries a forwarded key with a **different** value, a
  `ValueError` is raised.
- For string-parsed chained features, a consumer-forwarded value that differs from the
  value parsed from the feature name raises a `ValueError`, and so does a declared option
  or non-empty `in_features` that contradicts the name (see
  [Name Ownership and Agreement](#name-ownership-and-agreement)). Carve a forwarded key out
  with `forward_group_exclude` on the child, or set `MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1`
  to downgrade any of these errors to a warning.

### Opting out on the child

If the child must not see some (or any) of the consumer's group options, set a merge
directive on the child `Feature` you return from `input_features()`:

| Directive | Effect |
|-----------|--------|
| `forward_group=False` | Nothing flows: no group options and no consumer-side context push (the cascade stops here too). Only an explicit child-side `inherit_context_keys` pull can still deliver values. |
| `forward_group={"kg_backend"}` | Allowlist: only the listed keys flow. |
| `forward_group_exclude={"query_text"}` | Everything flows except the listed keys. |

```python
from mloda.user import Feature

# Only kg_backend flows; query_text and top_k stay on the consumer.
allowlisted = Feature("knowledge_graph", forward_group={"kg_backend"})

# Everything except query_text and top_k flows.
carved_out = Feature("knowledge_graph", forward_group_exclude={"query_text", "top_k"})

# Nothing flows.
isolated = Feature("knowledge_graph", forward_group=False)
```

- Allowlist keys absent from the consumer's group options are skipped silently.
- `forward_group=False` combined with a non-empty `forward_group_exclude` is
  contradictory and raises a `ValueError`.
- Merge directives (`forward_group`, `forward_group_exclude`, `inherit_context_keys`,
  `split_context_keys`) are not part of `Feature` identity, so declare at most one child
  per upstream feature name in a single `input_features` return; duplicates collapse like
  link/index. Duplicate requests union their `split_context_keys`; within one return, put
  the marker on the single child you return, as a set drops one of two equal Features.

### Pulling context with `inherit_context_keys`

Context options never flow implicitly, and `forward_group` does not touch them. If the
child needs a consumer **context** value, list it in `inherit_context_keys`; the engine
copies the listed keys from the consumer's context into the child's context:

```python
from mloda.user import Feature

child = Feature("knowledge_graph", inherit_context_keys={"tenant"})
```

This is the child-side pull, symmetric to the caller-side push
`propagate_context_keys` (one hop) or `cascade_context_keys` (all the way down), which
continues to work unchanged unless the child declared
`forward_group=False`: that skips the push entirely. Keys absent from the consumer's
context are skipped silently. As with group forwarding, an inherited or pushed key
whose child value differs raises a `ValueError`; equal values are a no-op. A child
that needs its own value for a pushed key opts out of per-level context values with
`forward_group=False` (only the literal `False` blocks the push; an empty allowlist
does not).

### Splitting on hand-set context with `split_context_keys`

A consumer can hand a renamed value to its input by setting it in the input's context;
that forwards nothing. Copies of a root input that differ in such a key still compute
separately when a consumer reads one of them, but copies that are all direct requests with
differing values are rejected. Declare the key with `split_context_keys` to split them
instead: the keys must already be in the feature's `options.context`
(otherwise `ValueError`), and the declaration only marks them. Within the feature group's
scope every feature then splits on the key by value, so a direct request of the same input
with another value computes separately; equal values still share one call. Duplicate
requests of a feature union their markers; `Feature` equality is unchanged.

```python
from mloda.user import Feature, Options

candidates = Feature(
    "candidates",
    Options(context={"top_k": 5}),
    split_context_keys={"top_k"},
)
```

The split applies to root inputs and `replaces_rows` groups. A marked input with inputs of
its own, reached twice with differing values over the same parents, is still rejected;
use `cascade_context_keys` there. For a `replaces_rows` input, a marker on any copy splits
every copy in the group's scope, as for roots.

### Resolution errors from over-forwarding

Because forwarding is the default, the consumer's query-specific group keys (for
example `query_text`, `top_k`) land on the child unless you opt out. If the child's
matcher (its `match_feature_group_criteria` / `PROPERTY_MAPPING`) does not accept those
extra keys, resolution fails with `No feature groups found ...`. A key whose *value* was
rejected is named in the error's value-rejection lines; a key rejected merely for being
present is not called out, because identifying it would need a second, speculative match
pass. If a feature resolves without its forwarded group options but not with them, keep
them off the child with `forward_group_exclude`, an allowlist, or `forward_group=False`.

A forwarded `query_text` makes source groups without a query route decline, so consumers that use `query_text` for themselves should exclude it from forwarding.

### Upstream feature deduplication

The engine deduplicates upstream features that share the same group identity. Declaring
the same source feature as the input of several parents therefore computes it once,
**provided the same options flow to it from each parent**. Forwarded options become
part of the child's group identity, so two parents whose merges produce the same group
options share a single computed feature, while a plain request of the same name (or one
carrying different options) is a distinct identity and is computed separately. Keep the
merge directives identical across parents when you want them to share the computation.

## Multiple Result Columns with ~ Pattern

A group that produces several columns from one input names them `feature~N` with `apply_naming_convention()` and consumers discover them with `resolve_multi_column_feature()`; see [Multiple Result Columns](multiple_result_columns.md).
