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

### Unified Parser Architecture

The modernized `FeatureChainParser` provides a unified approach through the `match_configuration_feature_chain_parser` method that handles:

- **String-based features**: Traditional pattern matching with regex
- **Configuration-based features**: Modern approach using Options and PROPERTY_MAPPING
- **Dual validation**: Features can be validated using either or both approaches

### Name Ownership and Agreement

One resolution, `FeatureChainParserMixin.resolve_feature_name`, returns a `NameResolution` (`owned`, `bindings`, `sources`, `value_for(key)`). Matching, the rejection diagnostic, `input_features`, `_extract_source_features` and `_resolve_operation` all read it.

- **Ownership:** a name owns the feature when its `PREFIX_PATTERN` identifies the group, and an owned name is authoritative for every value it encodes.
- **Binding:** every named capture of an owned name that is a `PROPERTY_MAPPING` key is bound and validated like an option, even when the option is also set; an unsupported value is a recorded match-time rejection. A legacy positional capture binds only a value already in the key's `allowed_values`.
- **Agreement:** a declared option for a key the name binds, and a non-empty `in_features`, must agree with the name (`in_features` must equal the name's direct sources). A contradiction aborts the match with an error; `MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1` downgrades it to a warning. Only own keys are compared: forwarded keys and materialized defaults never conflict here. An agreeing `in_features` supplies the input features, so their options and `feature_group` scope are kept; a downgraded mismatch uses the name's bare sources instead. An override that calls only `match_parser_criteria` skips this check.
- **Operands:** every name or config source must be non-empty and the count within `MIN_IN_FEATURES` / `MAX_IN_FEATURES` (name path: recorded rejection; config path: silent non-match).

### Options Architecture: Group vs Context Parameters

The new `Options` class separates parameters into two categories:

- **Group Parameters**: Affect Feature Group resolution and splitting (stored in `options.group`)
- **Context Parameters**: Metadata that doesn't affect splitting (stored in `options.context`)

```python
from mloda.user import Options

# New Options architecture
options = Options(
    group={
        "data_source": "production",  # Affects Feature Group splitting
    },
    context={
        "aggregation_type": "sum",    # Doesn't affect splitting
        "in_features": "sales"
    }
)
```

### Configuration-Based Feature Creation

Modern feature creation uses the Options architecture:

```python
from mloda.user import Feature, Options

# Traditional string-based approach:
feature = Feature("sales__sum_aggr")

# Modern configuration-based approach:
feature = Feature(
    "placeholder",  # Will be replaced during processing
    Options(
        context={
            "aggregation_type": "sum",
            "in_features": "sales"
        }
    )
)
```

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
| `PROPERTY_MAPPING` | `dict[str, PropertySpec]` | Required | Parameter validation configuration |
| `MIN_IN_FEATURES` | `int` | `1` | Minimum required in_features (an absent `in_features` counts as zero on the configuration path) |
| `MAX_IN_FEATURES` | `int \| None` | `None` | Maximum allowed in_features (None = unlimited) |
| `IN_FEATURE_SEPARATOR` | `str` | `"&"` | Separator for multiple in_features |
| `RECOGNITION_ONLY_PATTERN` | `bool` | `False` | Declares a captureless pattern as recognition-only (binds no key from the name) |
| `REQUIRED_COLUMNWISE_HOOKS` | `frozenset[str]` | `frozenset()` | Column-wise data hooks the family requires |

`Options.get_in_features()` returns sources in declared order (sets are sorted by name; duplicates are kept). Order-sensitive groups can reject sets with an `in_features` `match_guard=lambda v: not isinstance(v, (set, frozenset))` plus an `expected` text (for example "an ordered list, tuple or single source") so the rejection is reported.

### Column-Wise Data Hooks

Beyond parsing, the mixin declares three column-wise data hooks: `_get_available_columns`,
`_check_source_features_exist` and `_add_result_to_data`. The concrete compute-framework subclass
(pandas, PyArrow, Polars, python dict, ...) implements them; the inherited defaults raise
`NotImplementedError` naming the class and the hook, so a missing implementation fails loudly
instead of silently. A group that resolves column names against the data implements the discovery
hook too; the others only need the check/add pair. `_get_available_columns` can return `<Framework>.extract_column_names(data)`, the framework's own instance-free column listing.

Whether `_check_source_features_exist` tolerates partial presence (some source names missing) or
rejects it is a per-feature-group policy, not a framework rule.

`REQUIRED_COLUMNWISE_HOOKS` declares which of the three a family needs. A family base sets it to the
constant matching the hooks its own `calculate_feature` calls, and its compute-framework subclasses
implement them. Both constants come from `mloda.provider`.

| Constant | Declares |
|----------|----------|
| `COLUMNWISE_HOOKS` | `_check_source_features_exist` and `_add_result_to_data` |
| `COLUMN_DISCOVERY_HOOKS` | those two plus `_get_available_columns` |

```python
from mloda.provider import COLUMN_DISCOVERY_HOOKS, FeatureChainParserMixin, FeatureGroup
from mloda.user.pandas import PandasDataFrame

class RollingBase(FeatureChainParserMixin, FeatureGroup):
    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS

class PandasRolling(RollingBase):
    @classmethod
    def compute_framework_rule(cls):
        return {PandasDataFrame}

    @classmethod
    def _get_available_columns(cls, data):
        return PandasDataFrame.extract_column_names(data)

    @classmethod
    def _check_source_features_exist(cls, data, feature_names):
        if set(feature_names) - set(data.columns):
            raise ValueError(f"Missing source features, available: {list(data.columns)}")

    @classmethod
    def _add_result_to_data(cls, data, feature_name, result):
        data[feature_name] = result
        return data
```

`missing_columnwise_hooks(cls)`, also from `mloda.provider`, returns the declared hooks a class does
not implement. Assert it empty in your own test suite to catch a skipped hook there rather than
mid-run:

```python
from mloda.provider import missing_columnwise_hooks

def test_pandas_rolling_implements_its_hooks():
    assert missing_columnwise_hooks(PandasRolling) == []
```

`missing_columnwise_hooks(cls) == []` also passes vacuously for a class that never sets
`REQUIRED_COLUMNWISE_HOOKS`. Use `declared_columnwise_hooks(cls)`, also from `mloda.provider`, to
assert the declaration itself is what's expected:

```python
from mloda.provider import declared_columnwise_hooks, COLUMN_DISCOVERY_HOOKS

def test_pandas_rolling_declares_all_hooks():
    assert declared_columnwise_hooks(PandasRolling) == COLUMN_DISCOVERY_HOOKS
```

A hook that is not a `@classmethod` or `@staticmethod` counts as missing: the `cls._hook(...)` call
cannot reach it. A family base reports all of its declared hooks, since it declares the contract its
subclasses implement, so assert this on the framework-bound class rather than on the base.

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
class TimeWindowFeatureGroup(TimeReferenceMixin, FeatureChainParserMixin, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        # The mixin resolves sources from the name or from in_features; add the extras.
        sources = super().input_features(options, feature_name) or set()
        return sources | {Feature(self.get_reference_time_column(options))}
```

#### 3. Custom `match_feature_group_criteria()` Method

Override for the rules a `PROPERTY_MAPPING` spec cannot express, then delegate. Conditional
requirements belong in `required_when` instead: it is enforced by a guard installed at class
definition, so it still runs on an override.

```py
class SklearnPipelineFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None) -> bool:
        """Mutual exclusivity: pipeline_name and pipeline_steps cannot both be set."""
        if options.get(cls.PIPELINE_NAME) is not None and options.get(cls.PIPELINE_STEPS) is not None:
            return False

        return super().match_feature_group_criteria(feature_name, options, data_access_collection)
```

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

#### 5. Whole-Value Validation with `match_guard`

Give a spec a `match_guard` callable to validate the shape of the raw option value. The mixin calls it after basic matching succeeds, and a falsy return makes `match_feature_group_criteria` return False (a non-match, not an error).

Reach for `match_guard` when the constraint is about the value as a whole (a list of strings, a dict, an ordering). Reach for `element_validator` with `strict_validation=True` when the constraint is about each element on its own.

```py
def _is_list_of_strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

class GroupAggregation(FeatureChainParserMixin, FeatureGroup):
    PROPERTY_MAPPING = {
        "partition_by": PropertySpec(
            "Columns to partition by",
            match_guard=_is_list_of_strings,
            expected="a list of strings",
        ),
    }
```

`expected` names what the guard accepts, so a rejection is reported in the resolution-failure output instead of staying a silent non-match.

The guard is only called when the option is present (not None). Validators must be pure functions, since they may be called several times during resolution. If the guard raises, the value counts as rejected (match returns False).

The full model, which invariant fires at which moment, the precedence between the two callables, and what a validator receives for each container type, lives in one place: [PROPERTY_MAPPING Configuration](property-mapping.md).

## Modern Implementation in Feature Groups

### 1. Define PROPERTY_MAPPING Configuration

The modern approach uses `PROPERTY_MAPPING` to define parameter validation and classification:

```py
from mloda.provider import FeatureChainParserMixin, FeatureGroup
from mloda.user import FeatureName
from mloda.provider import DefaultOptionKeys, PropertySpec

class MyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r"__(?P<operation_type>[a-zA-Z_]+)_operation$"
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        # Feature-specific parameter
        "operation_type": PropertySpec(
            "Operation to apply",
            allowed_values={
                "sum": "Sum aggregation",
                "avg": "Average aggregation",
                "max": "Maximum aggregation",
            },
            context=True,  # Context parameter (the default)
            strict_validation=True,  # Strict validation
        ),
        # Source feature parameter
        DefaultOptionKeys.in_features: PropertySpec(
            "Source feature for the operation",
            strict_validation=False,  # Flexible validation
        ),
    }
```

### 2. Update match_feature_group_criteria

`FeatureChainParserMixin` already implements this; override it only to add your own checks, and reach the parser through `cls.match_parser_criteria`:

```py
@classmethod
def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
    return cls.match_parser_criteria(feature_name, options)
```

`match_parser_criteria` calls the parser with the class's `PROPERTY_MAPPING` and patterns and turns a rejected option value into a non-match. Calling `FeatureChainParser` directly from a match hook lets that rejection escape as an exception; the engine contains it as a `match hook` near-miss for that candidate, but the rejection reason is the more useful one. Containment covers plugin raises only: a framework-owned raise, such as a forwarded option value contradicting the feature name, still aborts the whole resolution.

### 3. Modernize input_features Method

The mixin's `input_features` already handles both forms: an owned name supplies its sources (an agreeing `in_features` keeps its options and scope), otherwise `in_features` does. Override it only to add extras:

```py
def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
    sources = super().input_features(options, feature_name) or set()
    return sources | {Feature("reference_time")}
```

Custom code checks source counts with `validate_in_feature_count` (raises) or `in_feature_count_reason` (returns the message or None).

### 4. Update calculate_feature Method

Read each value through the mixin helpers, so the name stays authoritative when it owns the feature:

```py
@classmethod
def calculate_feature(cls, data, features):
    for feature in features.features:
        operation_type = cls._resolve_operation(feature, "operation_type")
        in_feature_name = cls._extract_source_features(feature)[0]

        # Process using extracted values
        # ... implementation logic
```

### 5. Advanced PROPERTY_MAPPING Features

#### Element Validators

For per-element rules that a fixed value list cannot express:

```py
PROPERTY_MAPPING = {
    "dimension": PropertySpec(
        "Number of dimensions for reduction",
        strict_validation=True,
        element_validator=lambda x: isinstance(x, int) and x > 0,
    ),
}
```

#### Default Values

Specify default values for optional parameters:

```py
PROPERTY_MAPPING = {
    "window_size": PropertySpec(
        "Rolling window length in days",
        allowed_values={"7": "7-day window", "30": "30-day window"},
        default="7",  # Default value
        strict_validation=True,
    ),
}
```

Under strict validation a declared default must itself be an accepted value; that is
checked when the `PropertySpec` is constructed. Omitting `default` makes the key required;
`default=None` makes it optional with no value to apply. See
[Optional keys](property-mapping.md#optional-keys).

#### Group vs Context Classification

There is no `group` field: a group parameter is `context=False`.

```py
PROPERTY_MAPPING = {
    # Group parameter - affects Feature Group resolution
    "data_source": PropertySpec(
        "Data source to read from",
        allowed_values={"production": "Production data", "staging": "Staging data"},
        context=False,  # Group parameter
        strict_validation=True,
    ),
    # Context parameter - doesn't affect resolution
    "algorithm_type": PropertySpec(
        "Clustering algorithm",
        context=True,  # Context parameter (the default)
        strict_validation=False,  # Flexible validation
    ),
}
```

## Forwarding Options to Input Features

The [Context Propagation](property-mapping.md#context-propagation) docs describe the
**caller side**: context options stay local by default, and `propagate_context_keys`
opts specific keys in. This section describes the **author side**: what happens to a
feature group's own options when it declares an input feature.

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
| `forward_group=False` | Nothing flows: no group options and no consumer-side context push. Only an explicit child-side `inherit_context_keys` pull can still deliver values. |
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
- Merge directives (`forward_group`, `forward_group_exclude`, `inherit_context_keys`)
  are not part of `Feature` identity, so declare at most one child per upstream feature
  name in a single `input_features` return; duplicates collapse like link/index.

### Pulling context with `inherit_context_keys`

Context options never flow implicitly, and `forward_group` does not touch them. If the
child needs a consumer **context** value, list it in `inherit_context_keys`; the engine
copies the listed keys from the consumer's context into the child's context:

```python
from mloda.user import Feature

child = Feature("knowledge_graph", inherit_context_keys={"tenant"})
```

This is the child-side pull, symmetric to the caller-side push
`propagate_context_keys`, which continues to work unchanged unless the child declared
`forward_group=False`: that skips the push entirely. Keys absent from the consumer's
context are skipped silently. As with group forwarding, an inherited or pushed key
whose child value differs raises a `ValueError`; equal values are a no-op. A child
that needs its own value for a pushed key opts out of per-level context values with
`forward_group=False` (only the literal `False` blocks the push; an empty allowlist
does not).

### Resolution errors from over-forwarding

Because forwarding is the default, the consumer's query-specific group keys (for
example `query_text`, `top_k`) land on the child unless you opt out. If the child's
matcher (its `match_feature_group_criteria` / `PROPERTY_MAPPING`) does not accept those
extra keys, resolution fails with `No feature groups found ...`. A key whose *value* was
rejected is named in the error's value-rejection lines; a key rejected merely for being
present is not called out, because identifying it would need a second, speculative match
pass. If a feature resolves without its forwarded group options but not with them, keep
them off the child with `forward_group_exclude`, an allowlist, or `forward_group=False`.

### Upstream feature deduplication

The engine deduplicates upstream features that share the same group identity. Declaring
the same source feature as the input of several parents therefore computes it once,
**provided the same options flow to it from each parent**. Forwarded options become
part of the child's group identity, so two parents whose merges produce the same group
options share a single computed feature, while a plain request of the same name (or one
carrying different options) is a distinct identity and is computed separately. Keep the
merge directives identical across parents when you want them to share the computation.

## Multiple Result Columns with ~ Pattern

Some feature groups produce multiple result columns from a single input feature. mloda provides utilities to work with these patterns seamlessly.

### Producer Side: Creating Multi-Column Outputs

Use `apply_naming_convention()` to create properly named columns:

```py
from mloda.provider import FeatureGroup, FeatureSet

class MultiColumnProducer(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Compute results (e.g., from sklearn OneHotEncoder)
        result = encoder.transform(data)  # Returns 2D numpy array (n_samples, n_features)

        # Automatically apply naming convention
        feature_name = str(features.get_name_of_one_feature())
        named_columns = cls.apply_naming_convention(result, feature_name)
        # Returns: {"category__onehot_encoded~0": data, "~1": data, "~2": data}

        return named_columns
```

### Consumer Side: Discovering Multi-Column Features

Use `resolve_multi_column_feature()` to automatically discover columns:

```py
class MultiColumnConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        # Request base feature without ~N suffix
        return {Feature("category__onehot_encoded")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Automatically discover all matching columns
        columns = cls.resolve_multi_column_feature(
            "category__onehot_encoded",
            set(data.columns)
        )
        # Returns: ["category__onehot_encoded~0", "~1", "~2"]

        # Process all discovered columns
        result = sum(data[col] for col in columns)

        feature_name = str(features.get_name_of_one_feature())
        return {feature_name: result}
```

### Manual Column Access (Legacy)

For backwards compatibility, you can still access specific columns:

```python
# Manual specification of specific columns
base_feature = "category__onehot_encoded"  # Creates all columns
specific_column = "category__onehot_encoded~0"  # Access first column
another_column = "category__onehot_encoded~1"  # Access second column
```

**Recommended**: Use automatic discovery (`resolve_multi_column_feature`) instead of manual enumeration for cleaner, more maintainable code.

## Benefits

- **Consistent Naming**: Enforces naming conventions across feature groups
- **Composability**: Enables building complex features through chaining
- **Configuration-Based Creation**: Simplifies feature creation in client code
- **Validation**: Ensures feature names follow expected patterns
- **Multi-Column Support**: Handle transformations that produce multiple result columns
