# Feature Group Matching Criteria

## Overview

The mloda framework uses a sophisticated matching system to determine which feature group should handle a given feature. The modern approach supports both traditional string-based matching and configuration-based matching through the unified `FeatureChainParser`.

## Matching Process

When a feature is requested, the system checks all available feature groups to find the one that should handle the feature. This is done through the `match_feature_group_criteria` method in each feature group, which now typically uses the unified parser approach. Feature and column names match case-sensitively; see [Column name case sensitivity](compute-framework-integration.md#column-name-case-sensitivity).

## Modern Unified Matching

The recommended approach uses `FeatureChainParserMixin`, whose default `match_feature_group_criteria` already does this. Override it only to add your own checks, and reach the parser through `cls.match_parser_criteria`, which reads `PROPERTY_MAPPING` (configuration-based matching) and `PREFIX_PATTERN` / `SUFFIX_PATTERN` (string-based matching) off the class:

### 1. Dual Approach Support

```python
from mloda.provider import DefaultOptionKeys
from mloda.user import Options

@classmethod
def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
    return cls.match_parser_criteria(feature_name, options)
```

Do not call `FeatureChainParser.match_configuration_feature_chain_parser` directly from a match hook: it raises on an option value the `PROPERTY_MAPPING` rejects. An exception out of a match hook is contained as a `match hook` near-miss for that candidate instead of taking the whole resolution down, but a contained crash is a worse reason than a rejection. `match_parser_criteria` turns that rejection into a non-match, and the reason still reaches the user in the "No feature groups found" error.

`match_parser_criteria` does not run `match_guard`, the in_features count or name agreement; a widening override keeps the value and guard checks with `cls.passes_option_declarations(options)`.

To report why a `match_guard` declined a value without overriding `match_feature_group_criteria`, declare `expected` on the spec instead; see [PROPERTY_MAPPING Configuration](property-mapping.md).

Containment covers plugin raises only: a framework-owned raise (a two-readers conflict, a forwarded value contradicting the feature name, a rejected effective-options build) still aborts the whole resolution, because it reports a misconfiguration you have to fix.

Filter matching contains the same way: a raise is a non-match for that probe, like a `False` return, and is recorded in `GlobalFilter.dropped_filters` as a `match hook` near-miss, as is a typed decline the matcher records; a framework-owned raise still aborts. Every entry names the gate that dropped the filter and that gate's reason. [How filters reach your FeatureGroup](filter_data.md#how-filters-reach-your-featuregroup) tables the gates the two paths share and where filter policy differs.

A filter whose column is served by a different data source than its feature is dropped and recorded in `dropped_filters` as an `input data` near-miss naming both sources. A string `feature_group` scope that matches no accessible group adds a `Did you mean one of: ...?` line to the failure message.

The probe runs per feature, but a matched filter attaches to the whole `FeatureSet`, so a non-match for one feature does not suppress a filter a sibling matched. See [Filter scope](filter_data.md#filter-scope-is-the-featureset).

Every caller reads the return by truthiness: any falsy value is a non-match, any truthy value a match. Filter matching additionally reports a falsy value that is not `False`, and each distinct report is a WARNING once per setup.

A match hook that refuses the feature because of its name, for example an unknown output part, records the reason with `record_match_rejection(cls.__name__, reason, stage=NAME_STAGE)` (both from `mloda.provider`) and returns `False`; the near-miss then reads `feature name` instead of `option value`. Record the reason only when the name is otherwise addressed to the group, for example its base or prefix matches but the output part is unknown; a plain name mismatch returns `False` without recording anything.

The options view depends on the caller: feature resolution passes declared (pre-default) options, while filter matching runs after intake and passes the resolved feature's effective (post-default) options merged onto the filter feature's own. Matching logic that reads option values can see different values on the two paths. See [Applying declared defaults](property-mapping.md#applying-declared-defaults).

#### Option writes from a match hook

A match hook may write into the options it is handed. Each candidate matches against its own shallow copy of the request's options, and only the winner's copy (values, non-forwarded marks, own-key provenance) is applied to the feature. A candidate that returns `False`, raises, or is dropped by a later gate leaves nothing behind, and a failed resolution leaves the options as requested. A readerless subclass inherits its nearest replaced ancestor's reader pair, and two unrelated survivors give "Multiple feature groups found", naming each candidate's source; the reader pair is never adopted into the options. Different MatchData writes under one class name raise. The hook keeps its `bool` return, so existing hooks need no change, but must not keep the options reference after returning. Rebind a value rather than mutating a nested container in place, since the copy is shallow.

### 2. PROPERTY_MAPPING Configuration

The `PROPERTY_MAPPING` defines how configuration-based features are validated:

```python
from mloda.provider import PropertySpec

PROPERTY_MAPPING = {
    "aggregation_type": PropertySpec(
        "Aggregation to apply",
        allowed_values={
            "sum": "Sum aggregation",
            "avg": "Average aggregation",
            "max": "Maximum aggregation",
        },
        strict_validation=True,
    ),
    DefaultOptionKeys.in_features: PropertySpec(
        "Source feature for aggregation",
        strict_validation=False,
    ),
}
```

Every value in the mapping is a `PropertySpec`; accepted values go under `allowed_values`.
A raw dict spec raises at class definition, and an unknown field is a constructor
`TypeError`. See [PROPERTY_MAPPING Configuration](property-mapping.md) for the full model.
On the configuration path, a class with the default `MIN_IN_FEATURES = 1` does not match options that omit `in_features`, because an absent or explicit `None` `in_features` counts as zero; a source-less chained group sets `MIN_IN_FEATURES = 0` to opt out.

### 3. Validation Modes

#### Strict Validation
With `strict_validation=True`, parameter values must be in the value space:

```python
# This will match
options = Options(context={"aggregation_type": "sum"})  # "sum" is in mapping

# This will fail validation
options = Options(context={"aggregation_type": "custom"})  # "custom" not in mapping
```

#### Flexible Validation
With `strict_validation=False` (the default), any value is accepted:

```python
# Both will match
options = Options(context={"in_features": "sales"})      # Any value OK
options = Options(context={"in_features": "custom_feature"})  # Any value OK
```

#### Custom Validation Functions
For complex validation beyond simple value lists:

```python
from mloda.provider import is_positive_int

PROPERTY_MAPPING = {
    "window_size": PropertySpec(
        "Size of the time window",
        strict_validation=True,
        element_validator=is_positive_int,
    ),
}
```

## Legacy Default Matching Criteria

For feature groups not yet modernized, the default matching criteria still apply:

1. **Root Feature with Matching Input Data**: The feature group is a root feature (has no dependencies) and its input data matches the feature.

2. **Class Name Match**: The feature name exactly matches the feature group's class name.
   ```py
   feature_name == FeatureGroup.get_class_name()
   ```

3. **Prefix Match**: The feature name starts with the feature group's class name as a prefix.
   ```py
   feature_name.startswith(FeatureGroup.prefix())  # Default prefix is "ClassName_"
   ```

4. **Explicitly Supported**: The feature name is in the set of explicitly supported feature names.
   ```py
   feature_name in FeatureGroup.feature_names_supported()
   ```

5. **PROPERTY_MAPPING**: A group matched by rules 1 to 4 must still pass its `PROPERTY_MAPPING`: required options present, present values valid, and every `match_guard` satisfied. See [Property Mapping](property-mapping.md#a-plain-feature-group).

An owned reader veto recorded during rule 1 (the user addressed the reader family by name and its declaration rejected the request, or its probe recorded a content decline and matched nothing) gates the name-based rules 2 to 4; see [Data Access Patterns](data-access-patterns.md) for the recording contract.

## Matching Examples

### Modern Feature Group (Aggregation)

```python
from mloda.user import Feature, Options

# String-based matching
feature = Feature("sales__sum_aggr")  # Matches via pattern

# Configuration-based matching
feature = Feature(
    "placeholder",
    Options(context={
        "aggregation_type": "sum",
        "in_features": "sales"
    })
)  # Matches via PROPERTY_MAPPING validation
```

### Parameter Classification Impact

The group/context parameter separation affects matching behavior:

```python
# These create different Feature Group instances (different group parameters)
feature1 = Feature("placeholder", Options(
    group={"data_source": "production"},
    context={"aggregation_type": "sum", "in_features": "sales"}
))

feature2 = Feature("placeholder", Options(
    group={"data_source": "staging"},  # Different group parameter
    context={"aggregation_type": "sum", "in_features": "sales"}
))

# These create the same Feature Group instance (same group, different context)
feature3 = Feature("placeholder", Options(
    group={"data_source": "production"},
    context={"aggregation_type": "sum", "in_features": "sales"}
))

feature4 = Feature("placeholder", Options(
    group={"data_source": "production"},  # Same group parameter
    context={"aggregation_type": "avg", "in_features": "revenue"}  # Different context
))
```

## Declared Attributes and Input Requirements

A feature group or reader can override the classmethod `declared_attributes(features)` to return what its output means, as a mapping of `str` keys to scalar (`str`, `int`, `float`, `bool`) values, for example `{"scale": 5000, "sensor": "Kinect"}`. The default is `{}`, and non-scalar values are dropped.

A feature states what it must find declared through the `required_declarations` keyword, for example `Feature("depth_raw", required_declarations={"scale": None})`, returned from `input_features()` or passed in a request. The mapping is key to required value: `None` accepts any declared value, anything else must be equal and of the same type (`True` does not satisfy `1`). Resolution checks each input feature against the declarations of the candidate feature group merged with those of its selected reader (the reader wins on a shared key). It calls `declared_attributes(None)`, so a declaration that depends on the runtime `FeatureSet` cannot be checked. A reader in a family that misses the requirement is skipped so a sibling reader can match; if every matching reader in the family misses it, the family is refused at the input data stage. Any other candidate that misses it is eliminated at the `declarations` stage. A `declared_attributes` that raises counts as a miss. When no candidate is left, the run fails before any data is loaded, naming the consumer (or the request) and the key. The requirement belongs to the Feature object, so a shared instance carries it for every consumer that returns it. Features that differ only in `required_declarations` count as the same feature: they merge at intake, and a set keeps one of them. A chained group whose `input_features` builds bare Features from the parsed name must override `input_features` to attach one.

At execution time core calls `declared_attributes(features)` with the runtime `FeatureSet` and exposes the result to extenders (see [Extender](../chapter1/extender.md) for `HookContext.declared_attributes`, the feature group's on its calculate and validate hooks, the reader's on `INPUT_DATA_LOAD`). Plan-time checks (`required_declarations`) only see `declared_attributes(None)`, and `FEATURE_GROUP_MATCHED` carries no declarations, so a declaration a governance check relies on (license, classification) must not depend on the `FeatureSet`. Declarations describe the data and leave through telemetry, so never put option values or credentials in them.

## Migration Path

When modernizing a feature group:

1. **Add PROPERTY_MAPPING** with parameter definitions
2. **Update match_feature_group_criteria** to use unified parser
3. **Classify parameters** as group vs context appropriately
4. **Test both approaches** work correctly
5. **Update documentation** and examples
