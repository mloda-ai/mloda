# Separator Analysis: Industry Research and Recommendations

**Related:** [Feature Chaining Separator Design](./feature-chaining-separator-design.md)
**Date:** November 30, 2025

---

## 1. Current mloda Separators

### 1.1 Separator Inventory

| Separator | Location | Purpose | Example |
|-----------|----------|---------|---------|
| `__` | `feature_chain_parser.py:38` | Feature chaining (prefix→source) | `sum_aggr__sales` |
| `~` | `abstract_feature_group.py:159` | Multi-column output index | `pca_2d__features~dim1` |
| `,` | `options.py:205` | Comma-separated source features in OPTIONS | `"feature1, feature2"` |
| `__` (nested) | `geo_distance/base.py:123` | Separating two point features | `haversine_distance__loc1__loc2` |

### 1.2 Code References

**Feature Chaining** (`feature_chain_parser.py:38`):
```python
parts = _feature_name.split("__", 1)
```

**Multi-Column Output** (`abstract_feature_group.py:159`):
```python
col_name = f"{feature_name}~{suffix}"
```

**Comma-Separated Options** (`options.py:205`):
```python
if "," in val:
    feature_names = [name.strip() for name in val.split(",")]
```

**GeoDistance Point Separation** (`geo_distance/base.py:123`):
```python
source_part = FeatureChainParser.extract_source_feature(feature_name, self.PREFIX_PATTERN)
parts = source_part.split("__", 1)  # ["customer_location", "store_location"]
```

---

## 2. Industry Research

### 2.1 Transformation Chaining (Prefix → Source)

How frameworks name transformed features:

| Framework | Separator | Format | Example |
|-----------|-----------|--------|---------|
| sklearn ColumnTransformer | `__` | `transformer__column` | `MinMaxScaler__column2` |
| dbt | `__` | `source__model` | `stripe__payments` |
| Feast | `:` | `view:feature` | `driver_hourly_stats:conv_rate` |

**Sources:**
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [dbt Naming Conventions](https://docs.getdbt.com/blog/stakeholder-friendly-model-names)
- [Feast Feature View](https://docs.feast.dev/getting-started/concepts/feature-view)

### 2.2 Multi-Column Output (One Feature → Many Columns)

How frameworks name output columns when one transformation produces multiple columns:

| Framework | Separator | Format | Example |
|-----------|-----------|--------|---------|
| sklearn OneHotEncoder | `_` | `column_category` | `Sex_female`, `AgeGroup_0` |
| pandas get_dummies | `_` (configurable via `prefix_sep`) | `column_category` | `House_Type_Detached` |
| sklearn generic | `_N` | `feature_N` | `feature_0`, `feature_1` |
| **mloda** | `~` | `feature~index` | `onehot_encoded__category~0` |

**Key Parameters:**
- sklearn OneHotEncoder: `feature_name_combiner` parameter for custom separators
- pandas get_dummies: `prefix_sep` parameter (default `_`)

**Sources:**
- [sklearn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [pandas get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

### 2.3 Multi-Feature Input (Many Features → One Output)

How frameworks name features that combine multiple input features:

| Framework | Separator | Format | Example |
|-----------|-----------|--------|---------|
| sklearn PolynomialFeatures | `*` | `feature1*feature2` | `age*income`, `A*B` |
| DSL patterns | `(:)` or `(*)` | `feature1(:)feature2` | `c_1(:)c_2` (categorical combo) |
| TensorFlow crossed_column | (hashed internally) | N/A | No visible separator |
| Feature engineering DSLs | `x` or `X` | `feature1_x_feature2` | `age_x_income` |

**sklearn PolynomialFeatures History:**

The separator was changed from whitespace to `*` to resolve ambiguity:
```python
# Old (confusing): ['A B', 'A B C', 'A B D']  - Is "A B" one feature or two?
# New (clear):     ['A B', 'A B*C', 'A B*D']  - Clearly shows A B interacting with C
```

**Sources:**
- [sklearn PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [PolynomialFeatures separator change PR](https://github.com/scikit-learn/scikit-learn/pull/10886)
- [TensorFlow crossed_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column)

---

## 3. Key Insight: Three Semantic Categories

The industry distinguishes between **three different semantic operations**, each with its own separator convention:

| Category | Semantic Meaning | Direction | Industry Separator |
|----------|------------------|-----------|-------------------|
| **Transformation Chaining** | Apply transformation to source | Vertical (composition) | `__` |
| **Multi-Column Output** | One feature produces many columns | Horizontal (decomposition) | `_` or index suffix |
| **Multi-Feature Input** | Many features combine into one | Horizontal (combination) | `*` |

### Visual Representation

```
                    TRANSFORMATION CHAINING (vertical)
                    "Apply transformation to source"
                              │
                              ▼
                    ┌─────────────────┐
                    │  sum_aggr__     │──► source feature
                    │  (prefix)       │
                    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   MULTI-FEATURE         SINGLE              MULTI-COLUMN
   INPUT                 OUTPUT              OUTPUT
   "Combine A and B"     "One result"        "Multiple results"
        │                     │                     │
        ▼                     ▼                     ▼
   feature1*feature2     feature            feature~0, feature~1
```

---

## 4. The Problem with Current GeoDistance

### 4.1 Current Implementation

GeoDistance uses `__` for **both** chaining AND multi-feature input:

```python
# Current: haversine_distance__customer_location__store_location
#          ├────────────────┘  ├───────────────┘  ├─────────────┘
#          transformation      point 1            point 2
#          prefix              └────── same __ separator ──────┘
```

This is problematic because:
1. **Ambiguous parsing**: Is the second `__` a chain or a feature combination?
2. **Different semantics**: Chaining vs. combination are fundamentally different operations
3. **Inconsistent with industry**: Multi-feature inputs typically use `*` or similar

### 4.2 With Proposed Dot Separator

If we only change `__` to `.`:

```python
# Problematic: haversine_distance.customer_location.store_location
#              ├────────────────┘ ├───────────────┘ ├─────────────┘
#              transformation     point 1           point 2
#              prefix             └─── still ambiguous ───┘
```

The `.` would mean two different things:
1. Chaining (transformation → source)
2. Feature combination (point1 + point2)

---

## 5. Recommendation: Three-Separator System

### 5.1 Proposed Separators

| Separator | Purpose | Semantic Meaning | Example |
|-----------|---------|------------------|---------|
| `.` | **Chaining** | Transformation → source | `sum_aggr.sales` |
| `~` | **Multi-column output** | Feature → indexed columns | `pca_2d.features~dim1` |
| `&` | **Multi-feature input** | Feature + feature → result | `haversine_distance.loc1&loc2` |

### 5.2 Why `&` for Multi-Feature Input?

**Options Considered:**

| Symbol | Pros | Cons |
|--------|------|------|
| `*` | sklearn convention, familiar | Could imply multiplication |
| `&` | Clear "AND" meaning, single char | Less common in ML |
| `+` | Intuitive "combination" | Could imply addition |
| `x` or `X` | Cross-product notation | Could be part of feature name |
| `_x_` | Very explicit | Three characters, verbose |

**Recommendation: `&`**

Rationale:
1. **Clear semantic meaning**: "A AND B" - distance between A and B
2. **Single character**: Concise like `.` and `~`
3. **Not used elsewhere**: No conflicts in mloda codebase
4. **Distinct**: Visually different from `.` (chaining) and `~` (index)
5. **No Python conflicts**: Safe in string literals
6. **Shell safe**: No special meaning (when quoted)

### 5.3 Complete Example Comparison

**Before (Current - Hard to Read):**
```python
# Chaining
Feature("sum_aggr__sales")
Feature("max_aggr__sum_7_day_window__mean_imputed__price")

# Multi-column
Feature("onehot_encoded__category")  # produces ~0, ~1, ~2...

# Multi-feature input (GeoDistance)
Feature("haversine_distance__customer_location__store_location")
```

**After (Proposed - Clear Semantics):**
```python
# Chaining (. separator)
Feature("sum_aggr.sales")
Feature("max_aggr.sum_7_day_window.mean_imputed.price")

# Multi-column (~ separator - unchanged)
Feature("onehot_encoded.category")  # produces ~0, ~1, ~2...

# Multi-feature input (& separator)
Feature("haversine_distance.customer_location&store_location")
```

### 5.4 Parsing Logic

```python
# Chaining: split on first "."
"sum_aggr.sales".split(".", 1)  # ["sum_aggr", "sales"]

# Multi-feature: split on "&" for source features
"customer_location&store_location".split("&")  # ["customer_location", "store_location"]

# Multi-column: split on "~" for column index
"onehot_encoded.category~0".split("~")  # ["onehot_encoded.category", "0"]
```

---

## 6. Impact on Feature Groups

### 6.1 Feature Groups Using Multi-Feature Input

| Feature Group | Current Format | Proposed Format |
|---------------|----------------|-----------------|
| GeoDistanceFeatureGroup | `haversine_distance__loc1__loc2` | `haversine_distance.loc1&loc2` |

### 6.2 Implementation Changes for GeoDistance

**Current** (`geo_distance/base.py:123`):
```python
parts = source_part.split("__", 1)
```

**Proposed**:
```python
MULTI_FEATURE_SEPARATOR = "&"
parts = source_part.split(MULTI_FEATURE_SEPARATOR, 1)
```

---

## 7. Summary: Complete Separator System

| Separator | Name | Purpose | Direction | Example |
|-----------|------|---------|-----------|---------|
| `.` | Chain separator | Transformation → source | Vertical | `sum_aggr.sales` |
| `~` | Column separator | Feature → column index | Horizontal (out) | `feature~0` |
| `&` | Input separator | Features → combined feature | Horizontal (in) | `loc1&loc2` |
| `,` | Options separator | Parse source features in config | Config only | `"feat1, feat2"` |

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  haversine_distance.customer_location&store_location~0          │
│  ├─────────────────┘├───────────────────────────────┘├─┘        │
│  │                  │                                │          │
│  │                  │                                └── column │
│  │                  │                                    index  │
│  │                  │                                    (~)    │
│  │                  │                                           │
│  │                  └── multi-feature input (&)                 │
│  │                      "combine these features"                │
│  │                                                              │
│  └── chaining (.)                                               │
│      "apply transformation to source"                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. References

### Industry Documentation
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [sklearn PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [sklearn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [pandas get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- [TensorFlow crossed_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column)
- [Feast Feature View](https://docs.feast.dev/getting-started/concepts/feature-view)
- [dbt Naming Conventions](https://docs.getdbt.com/blog/stakeholder-friendly-model-names)

### GitHub Issues & PRs
- [PolynomialFeatures separator change](https://github.com/scikit-learn/scikit-learn/pull/10886)
- [mloda Issue #114: Feature Chaining Separator](https://github.com/mloda-ai/mloda/issues/114)

### Stack Overflow & Community
- [Feature naming conventions](https://stats.stackexchange.com/questions/22766/feature-naming-conventions)
- [sklearn feature names](https://stackoverflow.com/questions/54570947/feature-names-from-onehotencoder)
