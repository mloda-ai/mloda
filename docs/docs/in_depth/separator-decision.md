# Decision: Keep Double Underscore (`__`) as Feature Chain Separator

**Issue:** [https://github.com/mloda-ai/mloda/issues/114](https://github.com/mloda-ai/mloda/issues/114)
**Date:** November 30, 2025
**Status:** Decided - Keep `__`

---

## Decision

**We will keep the double underscore (`__`) as the feature chain separator.**

```python
# Current syntax (unchanged)
Feature("sum_aggr__sales")
Feature("max_aggr__sum_7_day_window__mean_imputed__price")
```

---

## Rationale

### 1. Regex Safety

The double underscore requires no escaping in regex patterns:

```python
# Safe - no escaping needed
PREFIX_PATTERN = r"^([\w]+)_imputed__"

# Dot would require escaping (error-prone)
PREFIX_PATTERN = r"^([\w]+)_imputed\."  # Easy to forget backslash
```

With 12+ feature groups using regex patterns, the risk of subtle bugs from missed escapes is significant.

### 2. Data Source Compatibility

Many real-world data sources use dots in column names:

| Source | Example Column Names |
|--------|---------------------|
| MongoDB (flattened) | `address.city`, `user.profile.name` |
| JSON APIs | `response.data.items` |
| Sensor systems | `sensor.temp.01` |
| External databases | `api.v2.response` |

With dot separator, these would create parsing ambiguity:

```python
# Ambiguous: Is this a chain or a dotted source name?
"user.email.mean_imputed"

# Unambiguous with __
"user.email__mean_imputed"  # Clear: source="user.email", transform="mean_imputed"
```

### 3. Industry Precedent

scikit-learn uses `__` for the same purpose in `ColumnTransformer` and `Pipeline`:

```python
# sklearn convention
preprocessor.named_transformers_['MinMaxScaler__column2']
```

---

## Alternatives Considered

### Dot (`.`)

```python
Feature("sum_aggr.sales")
Feature("max_aggr.sum_7_day_window.mean_imputed.price")
```

**Pros:** Excellent readability, familiar notation (JSONPath, attributes), single character.

**Cons:** Regex metacharacter requiring `\.` escaping in all patterns. Creates parsing ambiguity with data sources that use dots in column names (MongoDB, flattened JSON, APIs). High risk of subtle bugs from missed escapes.

**Decision:** Rejected due to regex safety and data source compatibility concerns.

---

### Double Colon (`::`)

```python
Feature("sum_aggr::sales")
Feature("max_aggr::sum_7_day_window::mean_imputed::price")
```

**Pros:** Visually distinct from snake_case, no regex escaping needed, no ambiguity with typical column names, familiar from C++/Ruby namespaces.

**Cons:** Two characters (same as `__`), less common in data/ML ecosystems, could be confused with Python's slice notation in some contexts.

**Decision:** Viable alternative but offers marginal improvement over `__` while introducing unfamiliarity.

---

### Arrow (`->`)

```python
Feature("sum_aggr->sales")
Feature("max_aggr->sum_7_day_window->mean_imputed->price")
```

**Pros:** Clearly shows transformation direction, very readable, no ambiguity with column names.

**Cons:** Two characters, uncommon in data frameworks, may need escaping in some shells, regex requires escaping (`-\>`).

**Decision:** Rejected due to shell escaping concerns and lack of ecosystem precedent.

---

### Colon (`:`)

```python
Feature("sum_aggr:sales")
Feature("max_aggr:sum_7_day_window:mean_imputed:price")
```

**Pros:** Used by Feast for feature views, single character, clear separation.

**Cons:** Conflicts visually with Python dict/slice syntax, could confuse users reading code, some data sources use colons in identifiers.

**Decision:** Rejected due to Python syntax confusion and potential column name conflicts.

---

### Double Colon with Angle (`::>`)

```python
Feature("sum_aggr::>sales")
Feature("max_aggr::>sum_7_day_window::>mean_imputed::>price")
```

**Pros:** Very explicit direction, zero ambiguity, unique.

**Cons:** Three characters (verbose), visually noisy, no ecosystem precedent.

**Decision:** Rejected due to verbosity and unfamiliarity.

---

### Pipe (`|`)

```python
Feature("sum_aggr|sales")
Feature("max_aggr|sum_7_day_window|mean_imputed|price")
```

**Pros:** Data pipeline feel, single character, clear separation.

**Cons:** Shell pipe conflict (requires quoting in bash), regex metacharacter requiring escaping.

**Decision:** Rejected due to shell conflicts and regex escaping needs.

---

### Forward Slash (`/`)

```python
Feature("sum_aggr/sales")
Feature("max_aggr/sum_7_day_window/mean_imputed/price")
```

**Pros:** Familiar path notation, single character, clear hierarchy.

**Cons:** Conflicts with file paths, division operator confusion, requires shell quoting.

**Decision:** Rejected due to file path confusion and shell escaping needs.

---

## Summary

| Separator | Readability | Regex Safe | Source Compatible | Shell Safe | Decision |
|-----------|-------------|------------|-------------------|------------|----------|
| `__` | Medium | Yes | Yes | Yes | **Keep** |
| `.` | Excellent | No | No | Yes | Rejected |
| `::` | Good | Yes | Yes | Yes | Viable but marginal |
| `->` | Excellent | No | Yes | Partial | Rejected |
| `:` | Good | Yes | Partial | Yes | Rejected |
| `\|` | Good | No | Yes | No | Rejected |
| `/` | Good | Yes | Partial | Partial | Rejected |

---

## Conclusion

While the double underscore (`__`) is not the most visually elegant separator, it provides:

1. **Zero regex escaping** - No risk of pattern bugs
2. **Full data source compatibility** - Works with dotted column names
3. **Industry alignment** - Matches sklearn conventions
4. **Proven stability** - Current implementation works reliably

The readability concern from Issue #114 is acknowledged, but the technical risks of alternatives outweigh the aesthetic benefits.
