# DONE: `Deferred` data-type declaration + fail-fast reconciliation (PR #493, refs #485)

## Final design (locked)

`return_data_type_rule` returns `DataTypeDeclaration = DataType | None | Deferred`:
- bare `DataType` — concrete type, known at planning
- `None` — no fixed type / polymorphic (explicit signal; the default)
- `Deferred` — fixed but only knowable at compute time

`engine.set_data_type` is pure reconciliation of the user's declared type vs the provider's
rule, and is **fail-fast** — it does NOT catch exceptions from the rule.

| declared (user) | outcome (provider) | result |
|---|---|---|
| any | `None` | `declared` |
| `None` | `DataType Y` | `Y` |
| `X` | `DataType Y`, `X == Y` | `Y` |
| `X` | `DataType Y`, `X != Y` | raise `ValueError` (mismatch) |
| any | `Deferred` | `declared` (no planning pin) |
| any | rule raises | exception propagates (fail-fast) |

## Why this diverges from #485

#485 proposed a catch-and-degrade wrapper ("planning never raises"). But the rule runs only
*after* the feature group is selected (engine.py:138 → :172; the rule is used nowhere in
selection), so a raise is a failure of a committed component, not a non-applicable candidate.
Swallowing it would hide a real bug. Hence fail-fast, with `None` as the explicit "no type"
signal. This still removes the motivation for per-plugin `try/except` (plugins return `None`
for "can't determine", and let genuine errors raise).

## Notes
- **Meaning A** for `Deferred`: the resolved dtype is carried by the computed data's native
  dtype; nothing in mloda reads a post-compute `feature.data_type`, so NO write-back and NO
  `Feature` marker. `Deferred` reconciles like `None` today; it is a declarable intent +
  forward seam.
- **Backward compatible**: `DataType | None | Deferred` is a superset of `Optional[DataType]`;
  existing plugins returning `DataType`/`None` need no change and still typecheck.
- Core-only PR. Registry `try/except` cleanup is a separate, optional follow-up.

## History (evolution during review)
Started as the literal #485 wrapper (`Fixed`/`Open`/`Deferred`/`Broken` union, catch + classify
+ log). Simplified across review: dropped `Fixed` (bare `DataType`), dropped dead `RuleOutcome`,
renamed `RuleResult` → `DataTypeDeclaration`, replaced `Open` with `None`, and finally dropped
the `try/except` + `Broken` in favor of fail-fast once we established the rule runs
post-selection.

## Gate
`tox` green: pytest -n8 --timeout=10, ruff format/check, pip-licenses, mypy --strict, bandit.
`EXPECTED_SKIP_COUNT` default is 194 (tox.ini).
