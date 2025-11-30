# Design Document: Feature Chaining Separator Improvement

**Issue:** [https://github.com/mloda-ai/mloda/issues/114](https://github.com/mloda-ai/mloda/issues/114)
**Author:** Claude (AI Assistant) with Tom Kaltofen
**Date:** November 30, 2025
**Status:** Draft - Awaiting Approval

---

## 1. Executive Summary

This document proposes changing mloda's feature chaining separator from double underscore (`__`) to dot (`.`) to improve readability. The change affects how transformation pipelines are expressed in feature names.

| Current | Proposed |
|---------|----------|
| `max_aggr__sum_7_day_window__mean_imputed__price` | `max_aggr.sum_7_day_window.mean_imputed.price` |

---

## 2. Problem Statement

### 2.1 Current Situation

mloda uses double underscore (`__`) as the separator between transformations in chained features:

```python
# Current format
Feature("sum_aggr__sales")                                    # Simple
Feature("mean_imputed__age")                                  # Simple
Feature("max_aggr__sum_7_day_window__mean_imputed__price")   # Complex chain
```

### 2.2 The Readability Issue

The double underscore creates visual confusion because:

1. **Blends with snake_case:** Single underscores in names (`mean_imputed`, `sum_7_day_window`) make double underscores hard to spot
2. **No visual hierarchy:** All `__` separators look identical regardless of nesting depth
3. **Scales poorly:** Problem worsens with longer transformation chains

**Example - Hard to parse:**
```
standard_scaled__mean_imputed__income
       ↑              ↑           ↑
     Where are the boundaries?
```

### 2.3 User Feedback

From Issue #114:
> "The current separator `__` is difficult to read"

Proposed alternatives in the issue:
- Dot notation: `standard_scaled.mean_imputed.income`
- Forward slash: `income/mean_imputed/standard_scaled` (reversed order)

---

## 3. Industry Research

### 3.1 How Other Frameworks Handle This

| Framework    | Separator | Format                         | Use Case                     |
|--------------|-----------|--------------------------------|------------------------------|
| scikit-learn | `__`      | `MinMaxScaler__column2`        | Single transformer prefix    |
| dbt          | `__`      | `stripe__payments`, `model__daily` | Source/transformation suffix |
| Feast        | `:`       | `driver_hourly_stats:conv_rate` | Feature view:feature        |
| Spark ML     | `_` suffix | `column_IDX`                   | User-defined columns         |

**Sources:**
- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [dbt Stakeholder-Friendly Model Names](https://docs.getdbt.com/blog/stakeholder-friendly-model-names)
- [Feast Feature View](https://docs.feast.dev/getting-started/concepts/feature-view)
- [Spark ML Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html)

### 3.2 Key Insight

The `__` separator is industry standard, BUT:
- sklearn/dbt use it for **single-level prefixing** only
- mloda's **multi-level chaining** is a unique use case where `__` becomes problematic

### 3.3 Multi-Column Feature Naming

| Framework             | Format       | Example                    |
|-----------------------|--------------|----------------------------|
| sklearn OneHotEncoder | `Column_Value` | `Sex_female`, `AgeGroup_0` |
| sklearn generic       | `feature_N`  | `feature_0`, `feature_1`   |
| mloda (current)       | `feature~N`  | `onehot_encoded__category~0` |

**Conclusion:** mloda's tilde (`~`) separator for multi-column features is unique but effective.

---

## 4. Current Separator Analysis

This section provides a comprehensive analysis of **all** separator symbols currently used in mloda for feature naming.

### 4.1 Overview: Dual Separator System

mloda uses a **two-separator system** for feature naming:

| Separator | Symbol | Purpose | Direction |
|-----------|--------|---------|-----------|
| **Feature Chaining** | `__` | Separates transformation prefix from source feature | Vertical (composition) |
| **Multi-Column** | `~` | Separates base feature from column index | Horizontal (decomposition) |

```
Feature Name Structure:
┌─────────────────────────────────────────────────────────────┐
│  max_aggr__sum_7_day_window__mean_imputed__price~0          │
│  ├─────┬─┘├──────┬────────┘├──────┬─────┘├──┬──┘├┬┘        │
│  │     │  │      │         │      │      │  │   ││         │
│  │     └──┼──────┴─────────┼──────┴──────┼──┘   │└ column  │
│  │        │                │             │      │   index  │
│  │        └── chaining ────┴─ separators─┘      │          │
│  │           (__)                               │          │
│  └──────────────────────────────────────────────┴ multi-   │
│                                                   column   │
│                                                   (~)      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Separator 1: Double Underscore (`__`) - Feature Chaining

**Purpose:** Primary separator for feature transformation chaining - connects transformation prefix to source feature.

**Core Implementation:**

| Location | Code | Purpose |
|----------|------|---------|
| `feature_chain_parser.py:38` | `parts = _feature_name.split("__", 1)` | Splits feature into prefix + source |
| `feature_chain_parser.py:308` | `find("__")` | Locates separator position |

**Usage Pattern:**
```python
# Pattern: {transformation_prefix}__{source_feature}
Feature("sum_aggr__sales")           # sum_aggr + sales
Feature("mean_imputed__age")         # mean_imputed + age
Feature("standard_scaled__income")   # standard_scaled + income

# Chained: Each __ is parsed recursively
Feature("max_aggr__sum_7_day_window__mean_imputed__price")
#        ├───────┘  ├──────────────────────────────────┘
#        prefix     source (which itself contains __)
```

**Feature Groups Using `__` (12 production + 2 test):**

| Feature Group | PREFIX_PATTERN | Example |
|---------------|----------------|---------|
| AggregatedFeatureGroup | `r"^([\w]+)_aggr__"` | `sum_aggr__sales` |
| MissingValueFeatureGroup | `r"^([\w]+)_imputed__"` | `mean_imputed__price` |
| TimeWindowFeatureGroup | `r"^([\w]+)_(\d+)_([\w]+)_window__"` | `avg_7_day_window__temp` |
| DimensionalityReductionFeatureGroup | `r"^([\w]+)_(\d+)d__"` | `pca_2d__features` |
| ClusteringFeatureGroup | `r"^cluster_([\w]+)_([\w]+)__"` | `cluster_kmeans_3__features` |
| ForecastingFeatureGroup | `r"^([\w]+)_forecast_(\d+)([\w]+)__"` | `arima_forecast_7days__sales` |
| GeoDistanceFeatureGroup | `r"^([\w]+)_distance__"` | `haversine_distance__loc1__loc2` |
| NodeCentralityFeatureGroup | `r"^([\w]+)_centrality__"` | `betweenness_centrality__graph` |
| TextCleaningFeatureGroup | `r"^cleaned_text__"` | `cleaned_text__review` |
| SKLearn ScalingFeatureGroup | `r"^(standard\|minmax\|robust\|normalizer)_scaled__"` | `standard_scaled__age` |
| SKLearn EncodingFeatureGroup | `r"^(onehot\|label\|ordinal)_encoded__"` | `onehot_encoded__state` |
| SKLearn PipelineFeatureGroup | `r"^sklearn_pipeline_([\w]+)__"` | `sklearn_pipeline_v1__data` |

**Issues with `__`:**
1. Blends visually with snake_case (`mean_imputed` vs `__`)
2. Hard to spot boundaries in long chains
3. This is the separator we propose changing to `.`

### 4.3 Separator 2: Tilde (`~`) - Multi-Column Features

**Purpose:** Separates base feature name from column index when a transformation produces multiple output columns.

**Core Implementation:**

| Location | Method | Code |
|----------|--------|------|
| `abstract_feature_group.py:159` | `apply_naming_convention()` | `col_name = f"{feature_name}~{suffix}"` |
| `abstract_feature_group.py:171` | `get_column_base_feature()` | `return column_name.split("~")[0]` |
| `abstract_feature_group.py:180` | `expand_feature_columns()` | `f"{feature_name}~{i}"` |
| `abstract_feature_group.py:194` | `resolve_multi_column_feature()` | `col.startswith(f"{feature_name}~")` |

**Usage Pattern:**
```python
# Pattern: {feature_name}~{column_index_or_suffix}

# One-hot encoding produces multiple columns
Feature("onehot_encoded__category")  # Input
# Output columns: onehot_encoded__category~0, onehot_encoded__category~1, ...

# Dimensionality reduction with custom suffixes
Feature("pca_2d__features")  # Input
# Output columns: pca_2d__features~dim1, pca_2d__features~dim2

# Combined with chaining
Feature("standard_scaled__onehot_encoded__state~0")
```

**Feature Groups Using `~`:**

| Feature Group | Use Case | Example Output |
|---------------|----------|----------------|
| SKLearn EncodingFeatureGroup | One-hot encoding categories | `onehot_encoded__state~0`, `~1`, `~2` |
| DimensionalityReductionFeatureGroup | PCA/UMAP dimensions | `pca_2d__features~dim1`, `~dim2` |
| SKLearn PipelineFeatureGroup | Pipeline multi-output | `sklearn_pipeline_v1__data~0` |
| SKLearn ScalingFeatureGroup | Multi-column scaling | `standard_scaled__prices~0` |

**Analysis of `~`:**

| Criteria | Assessment |
|----------|------------|
| **Readability** | ★★★★★ Excellent - clearly distinct from `_` and `__` |
| **Uniqueness** | ★★★★★ Rare character, unlikely conflicts |
| **Shell safety** | ★★★★☆ Good - may need quoting in some shells |
| **Familiarity** | ★★★☆☆ Less common, but intuitive |
| **Python clarity** | ★★★★★ No Python syntax conflicts |

**Recommendation:** **Keep `~` unchanged** - it works well and has no readability issues.

### 4.4 Special Case: Nested `__` in GeoDistanceFeatureGroup

**Purpose:** GeoDistanceFeatureGroup uses an additional `__` to separate two point features.

**Implementation** (`geo_distance/base.py:123`):
```python
# Feature: "haversine_distance__customer_location__store_location"
source_part = FeatureChainParser.extract_source_feature(feature_name, self.PREFIX_PATTERN)
# source_part = "customer_location__store_location"
parts = source_part.split("__", 1)
# parts = ["customer_location", "store_location"]
```

**Impact of Separator Change:**
With dot separator, this becomes:
```python
# Before: "haversine_distance__customer_location__store_location"
# After:  "haversine_distance.customer_location__store_location"

# The second __ separates the two point features, NOT a chain
# This needs special handling - options:
# 1. Keep __ for point separation (different semantic meaning)
# 2. Use different separator like + or &
# 3. Change to: "haversine_distance.customer_location.store_location"
```

**Recommendation:** Change GeoDistance to also use `.` for point separation:
```python
# Proposed: "haversine_distance.customer_location.store_location"
```
This maintains consistency. The parser knows GeoDistance expects exactly 2 source features.

### 4.5 Summary: Separator Inventory

| Separator | Current Use | Proposed Change | Rationale |
|-----------|-------------|-----------------|-----------|
| `__` | Feature chaining | **Change to `.`** | Improve readability |
| `~` | Multi-column index | **Keep as-is** | Working well, distinct |
| `__` (nested in GeoDistance) | Point separation | **Change to `.`** | Consistency |

### 4.6 Visual Comparison: Before and After

**Current (hard to read):**
```
standard_scaled__mean_imputed__income
onehot_encoded__category~0
haversine_distance__customer_loc__store_loc
```

**Proposed (clear boundaries):**
```
standard_scaled.mean_imputed.income
onehot_encoded.category~0
haversine_distance.customer_loc.store_loc
```

---

## 5. Design Options Considered

### 4.1 Option A: Dot Separator (`.`) - RECOMMENDED

```python
Feature("standard_scaled.mean_imputed.income")
Feature("max_aggr.sum_7_day_window.mean_imputed.price")
```

| Pros                               | Cons                                  |
|------------------------------------|---------------------------------------|
| Excellent readability              | May look like Python attribute access |
| Familiar to all programmers        | Could confuse IDEs in some contexts   |
| Clear visual break from snake_case | N/A                                   |
| Single character (concise)         | N/A                                   |
| No shell escaping needed           | N/A                                   |

**Mitigations for cons:**
- Feature names are string literals, not actual Python attributes
- Users write `Feature("a.b.c")`, not `df.a.b.c`
- IDE impact is minimal for string literals

### 4.2 Option B: Arrow Separator (`->`)

```python
Feature("standard_scaled->mean_imputed->income")
```

| Pros                           | Cons                             |
|--------------------------------|----------------------------------|
| Shows transformation direction | Two characters (longer)          |
| Very clear meaning             | Uncommon in data frameworks      |
| No Python conflicts            | May need escaping in some shells |

### 4.3 Option C: Colon Separator (`:`)

```python
Feature("standard_scaled:mean_imputed:income")
```

| Pros             | Cons                                    |
|------------------|-----------------------------------------|
| Used by Feast    | Conflicts with Python dict/slice syntax |
| Clear separation | May confuse users                       |
| Single character | N/A                                     |

### 4.4 Option D: Double Colon (`::`)

```python
Feature("standard_scaled::mean_imputed::income")
```

| Pros                  | Cons                           |
|-----------------------|--------------------------------|
| Very clear separation | Resembles C++ scope resolution |
| No Python conflicts   | Two characters                 |

### 4.5 Option E: Pipe (`|`)

```python
Feature("standard_scaled|mean_imputed|income")
```

| Pros               | Cons                     |
|--------------------|--------------------------|
| Data pipeline feel | Shell pipe conflict      |
| Clear separation   | Requires quoting in bash |

### 4.6 Comparison Matrix

| Criteria       | `.`   | `->`  | `:`   | `::`  | `\|`  |
|----------------|-------|-------|-------|-------|-------|
| Readability    | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| Familiarity    | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| Shell safety   | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| Conciseness    | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| Python clarity | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★★ |

---

## 5. Recommendation

### 5.1 Primary Change: Use Dot (`.`) Separator

**New feature name format:**
```python
# Before
Feature("sum_aggr__sales")
Feature("max_aggr__sum_7_day_window__mean_imputed__price")

# After
Feature("sum_aggr.sales")
Feature("max_aggr.sum_7_day_window.mean_imputed.price")
```

**Rationale:**
1. Best balance of readability and familiarity
2. Clear visual distinction from snake_case underscores
3. Widely understood (object.property pattern)
4. No shell escaping issues
5. Single character keeps names concise

### 5.2 Multi-Column Features: Keep Tilde (`~`)

**No change to multi-column separator:**
```python
# Stays the same
Feature("onehot_encoded.category~0")
Feature("onehot_encoded.category~1")
```

**Rationale:**
1. Already visually distinct from underscores AND dots
2. Working well in current implementation
3. No user complaints about this separator

### 5.3 Backward Compatibility

**Clean break (no backward compatibility):**
- Simpler implementation
- Cleaner codebase
- Major version bump (breaking change)
- Clear migration path for users

---

## 6. Technical Implementation

### 6.1 Current Implementation

**Core parser** (`feature_chain_parser.py:38`):
```python
parts = _feature_name.split("__", 1)
```

**Feature group patterns** (example from `aggregated_feature_group/base.py`):
```python
PATTERN = "_aggr__"
PREFIX_PATTERN = r"^([\w]+)_aggr__"
```

### 6.2 Files Requiring Modification

#### Core Files

| File | Changes |
|------|---------|
| `mloda_core/abstract_plugins/components/feature_chainer/feature_chain_parser.py` | Change `split("__", 1)` to `split(".", 1)`, update `find("__")` |

#### Feature Group Patterns (12+ files)

| File | PATTERN Change | PREFIX_PATTERN Change |
|------|----------------|----------------------|
| `aggregated_feature_group/base.py` | `_aggr__` → `_aggr.` | `_aggr__` → `_aggr\.` |
| `data_quality/missing_value/base.py` | `_imputed__` → `_imputed.` | `_imputed__` → `_imputed\.` |
| `dimensionality_reduction/base.py` | `__` → `.` | `d__` → `d\.` |
| `time_window/base.py` | `_window__` → `_window.` | `_window__` → `_window\.` |
| `text_cleaning/base.py` | `__` → `.` | `text__` → `text\.` |
| `node_centrality/base.py` | `_centrality__` → `_centrality.` | `_centrality__` → `_centrality\.` |
| `sklearn/scaling/base.py` | `_scaled__` → `_scaled.` | `_scaled__` → `_scaled\.` |
| `sklearn/encoding/base.py` | `_encoded__` → `_encoded.` | `_encoded__` → `_encoded\.` |
| `clustering/base.py` | `__` → `.` | `__` → `\.` |
| `forecasting/base.py` | `__` → `.` | `__` → `\.` |
| `geo_distance/base.py` | `_distance__` → `_distance.` | `_distance__` → `_distance\.` |
| `sklearn/pipeline/base.py` | `__` → `.` | `__` → `\.` |

#### Tests

| File | Changes |
|------|---------|
| `tests/test_plugins/integration_plugins/chainer/test_chained_features.py` | Update all feature name strings |
| `tests/test_plugins/integration_plugins/chainer/context/test_chained_context_features.py` | Update all feature name strings |
| `tests/test_plugins/integration_plugins/chainer/chainer_test_feature.py` | Update PATTERN constant |
| `tests/test_plugins/integration_plugins/chainer/chainer_context_feature.py` | Update PATTERN constant |

#### Documentation

| File | Changes |
|------|---------|
| `docs/docs/in_depth/feature-chain-parser.md` | Update all examples |
| All feature group docstrings | Update example feature names |

### 6.3 Implementation Steps

1. **Define separator constant** (centralize for future flexibility):
```python
# In feature_chain_parser.py
FEATURE_CHAIN_SEPARATOR = "."
```

2. **Update FeatureChainParser:**
```python
# Change from:
parts = _feature_name.split("__", 1)
# To:
parts = _feature_name.split(FEATURE_CHAIN_SEPARATOR, 1)
```

3. Update all PATTERN constants in feature groups
4. Update all PREFIX_PATTERN regex (escape dot as `\.`)
5. Update all tests
6. Update all documentation
7. Run full test suite (`tox`)

---

## 7. Migration Guide (For Users)

### 7.1 Breaking Change Notice

```
BREAKING CHANGE: Feature chaining separator changed from __ to .

Before: Feature("sum_aggr__sales")
After:  Feature("sum_aggr.sales")
```

### 7.2 Migration Script (Optional)

```python
def migrate_feature_name(old_name: str) -> str:
    """Convert old __ separator to new . separator."""
    # Note: This is a simple replacement - may need adjustment
    # for edge cases where __ appears in actual feature names
    return old_name.replace("__", ".")
```

### 7.3 Search & Replace Pattern

```
Find:    __
Replace: .
Files:   *.py (in user code that defines features)
```

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Users have existing code with `__` | High | Medium | Clear migration guide, major version bump |
| Dot confused with Python attributes | Low | Low | Feature names are strings, not attributes |
| IDE autocomplete issues | Low | Low | Minimal impact on string literals |
| Regex patterns break | Medium | High | Thorough testing, escape dots as `\.` |

---

## 9. Success Metrics

1. **Readability:** User feedback confirms improved clarity
2. **No regressions:** All existing tests pass with new separator
3. **Documentation:** All examples updated consistently

---

## 10. Open Questions

1. **Separator choice confirmation:** Is dot (`.`) the preferred option, or should we consider alternatives?
2. **Version bump:** Should this be v0.3.0 (minor) or v1.0.0 (major)?
3. **Deprecation period:** Should we support both separators temporarily?

---

## 11. Appendix: Full Example Comparison

### Before (Current)

```python
from mloda_core.api import Feature

# Simple features
sales_aggregated = Feature("sum_aggr__sales")
age_imputed = Feature("mean_imputed__age")
income_scaled = Feature("standard_scaled__income")

# Chained features
complex_feature = Feature("max_aggr__sum_7_day_window__mean_imputed__price")

# Multi-column (one-hot encoded)
category_0 = Feature("onehot_encoded__category~0")
category_1 = Feature("onehot_encoded__category~1")
```

### After (Proposed)

```python
from mloda_core.api import Feature

# Simple features
sales_aggregated = Feature("sum_aggr.sales")
age_imputed = Feature("mean_imputed.age")
income_scaled = Feature("standard_scaled.income")

# Chained features
complex_feature = Feature("max_aggr.sum_7_day_window.mean_imputed.price")

# Multi-column (one-hot encoded) - unchanged
category_0 = Feature("onehot_encoded.category~0")
category_1 = Feature("onehot_encoded.category~1")
```

---

## 12. References

- [Issue #114: Feature Chaining Separator](https://github.com/mloda-ai/mloda/issues/114)
- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [dbt Stakeholder-Friendly Model Names](https://docs.getdbt.com/blog/stakeholder-friendly-model-names)
- [Feast Feature View](https://docs.feast.dev/getting-started/concepts/feature-view)
- [Spark ML Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html)
