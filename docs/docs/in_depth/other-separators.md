# Other Separators in mloda

**Date:** November 30, 2025

This document covers separators used in mloda **other than** the feature chaining separator (`__`).

---

## 1. Tilde (`~`) - Multi-Column Output Index

### Current Usage

Used when a single transformation produces multiple output columns.

**Location:** `abstract_feature_group.py:159`

```python
col_name = f"{feature_name}~{suffix}"
```

### Example

```python
# One-hot encoding produces multiple columns
Feature("onehot_encoded__category")
# Output: onehot_encoded__category~0, onehot_encoded__category~1, onehot_encoded__category~2

# PCA with custom suffixes
Feature("pca_2d__features")
# Output: pca_2d__features~dim1, pca_2d__features~dim2
```

### Analysis

| Criteria | Assessment |
|----------|------------|
| Readability | Excellent - clearly distinct from `_` and `__` |
| Regex safety | Yes - no escaping needed |
| Uniqueness | Excellent - rare in column names |
| Shell safety | Good - may need quoting in some shells |

### Decision

**Keep `~`** - Works well, no issues reported, visually distinct.

---

## 2. Ampersand (`&`) - Multi-Feature Input

### Current Usage

Used in GeoDistanceFeatureGroup to combine two input features.

**Location:** `geo_distance/base.py:123` (currently uses `__`, proposed to use `&`)

### Current Problem

GeoDistance currently reuses `__` to separate input features:

```python
# Current (confusing - same separator for different purposes)
"haversine_distance__customer_location__store_location"
```

### Proposed Improvement

Use `&` to clearly indicate "these features are combined":

```python
# Proposed (clear semantic meaning)
"haversine_distance__customer_location&store_location"
```

### Analysis

| Criteria | Assessment |
|----------|------------|
| Readability | Excellent - "A & B" clearly means combination |
| Regex safety | Yes - no escaping needed |
| Semantic clarity | Excellent - distinct from chaining (`__`) |
| Shell safety | Yes - safe in quoted strings |

### Alternatives Considered

| Symbol | Pros | Cons |
|--------|------|------|
| `&` | Clear "AND" meaning | None significant |
| `*` | sklearn uses for interactions | Implies multiplication |
| `+` | Intuitive combination | Implies addition |
| `x` | Cross-product notation | Could be part of feature name |

### Decision

**Use `&` for multi-feature inputs** - Clear semantics, no conflicts.

---

## 3. Comma (`,`) - Options Parsing

### Current Usage

Used to parse comma-separated source features in configuration options (not in feature names).

**Location:** `options.py:205`

```python
if "," in val:
    feature_names = [name.strip() for name in val.split(",")]
```

### Example

```python
# In configuration/options (not feature names)
options = Options(source_features="feature1, feature2, feature3")
```

### Analysis

| Criteria | Assessment |
|----------|------------|
| Readability | Excellent - universal list separator |
| Scope | Config only - never appears in feature names |
| Familiarity | Universal - CSV, function args, etc. |

### Decision

**Keep `,` for options parsing** - Standard convention, config-only usage.

---

## Summary

| Separator | Purpose | Location | Decision |
|-----------|---------|----------|----------|
| `~` | Multi-column output index | Feature names | **Keep** |
| `&` | Multi-feature input | Feature names | **Use** (change from `__`) |
| `,` | Source feature list | Options/config | **Keep** |

### Complete Example

```python
# All separators working together:

# __ = chaining (transformation to source)
# &  = multi-feature input (combine two features)
# ~  = multi-column output (index)
# ,  = options parsing (config only)

Feature("haversine_distance__loc1&loc2")      # Chain + multi-input
Feature("onehot_encoded__category~0")          # Chain + multi-output
Feature("pca_2d__features~dim1")               # Chain + multi-output

# In options (not feature names):
Options(source_features="age, income, score")  # Comma-separated list
```
