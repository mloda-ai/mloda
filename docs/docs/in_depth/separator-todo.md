# Separator Implementation TODO

**Date:** November 30, 2025

---

## Summary of Decisions

| Separator | Purpose | Decision | Action Required |
|-----------|---------|----------|-----------------|
| `__` | Feature chaining | Keep | None |
| `~` | Multi-column output | Keep | None |
| `&` | Multi-feature input | Use | Implement in GeoDistance, Clustering |
| `,` | Options parsing | Keep | Remove from feature names |

---

## TODO List

### 1. Add Separator Constants to FeatureChainParser

- [ ] Add constants to `feature_chain_parser.py`:
  ```python
  CHAIN_SEPARATOR = "__"
  COLUMN_SEPARATOR = "~"
  INPUT_SEPARATOR = "&"
  ```
- [ ] Replace hardcoded `"__"` with `CHAIN_SEPARATOR` (lines 38, 39, 308, 310)
- [ ] Update docstring to document all separators

**File:** `mloda_core/abstract_plugins/components/feature_chainer/feature_chain_parser.py`

---

### 2. Update GeoDistanceFeatureGroup

- [ ] Change from `__` to `&` for separating two point features
- [ ] Update parsing logic:
  ```python
  # Before
  parts = source_part.split("__", 1)

  # After
  parts = source_part.split("&", 1)
  ```
- [ ] Update tests
- [ ] Update docstrings/examples

**File:** `mloda_plugins/feature_group/experimental/geo_distance/base.py`

**Example change:**
```python
# Before
Feature("haversine_distance__customer_location__store_location")

# After
Feature("haversine_distance__customer_location&store_location")
```

---

### 3. Update ClusteringFeatureGroup

- [ ] Change from `,` to `&` for separating source features in feature name
- [ ] Update parsing logic:
  ```python
  # Before
  source_features = [feature.strip() for feature in source_features_str.split(",")]

  # After
  source_features = [feature.strip() for feature in source_features_str.split("&")]
  ```
- [ ] Update tests
- [ ] Update docstrings/examples

**File:** `mloda_plugins/feature_group/experimental/clustering/base.py`

**Example change:**
```python
# Before
Feature("cluster_kmeans_3__age,income,score")

# After
Feature("cluster_kmeans_3__age&income&score")
```

---

### 4. Keep Options Comma Parsing (No Change)

- [x] Comma in Options API stays as-is (not in feature names)

**File:** `mloda_core/abstract_plugins/components/options.py`

**Usage (unchanged):**
```python
Options(source_features="feature1, feature2, feature3")
```

---

### 5. Update Documentation

- [ ] Update `docs/docs/in_depth/feature-chain-parser.md` with separator reference
- [ ] Update feature group docstrings with new examples
- [ ] Add separator reference to main docs

---

### 6. Update Tests

- [ ] `tests/test_plugins/integration_plugins/chainer/` - verify chaining still works
- [ ] GeoDistance tests - update feature name format
- [ ] Clustering tests - update feature name format

---

## Final Separator System

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   cluster_kmeans_3__age&income&score~0                      │
│   ├──────────────┘  ├──────────────┘ ├┘                     │
│   │                 │                │                      │
│   │                 │                └─ ~ column index      │
│   │                 │                                       │
│   │                 └─ & multi-feature input                │
│   │                                                         │
│   └─ __ feature chaining                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Options API (not in feature names):
  Options(source_features="a, b, c")  ← comma for convenience
```

---

## Related Documents

- [Separator Decision](./separator-decision.md) - Why `__` was kept
- [Other Separators](./other-separators.md) - Details on `~`, `&`, `,`
- [Separator Analysis](./separator-analysis.md) - Industry research
