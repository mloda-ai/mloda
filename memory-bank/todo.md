# TODO: RuleResult union for `return_data_type_rule` (issue #485)

## Decision (locked)

Break the plugin contract: `return_data_type_rule` returns a `RuleResult` tagged union
instead of `Optional[DataType]`. The never-raises guarantee is owned by core at the single
call site (`engine.set_data_type`), which classifies a raised rule into an engine-internal
`Broken` outcome (log + return no-planning-type) instead of crashing the whole graph.

Variants:
- `Fixed(DataType)` — concrete type, known at planning (plugin-returnable)
- `Open` — polymorphic by design, no fixed type ever (plugin-returnable; the new default)
- `Deferred` — fixed but only knowable at compute; type rides in the computed data
  ("Meaning A"), so it reconciles like `Open` today (no planning pin) and exists as a
  declarable intent + forward seam (plugin-returnable)
- `Broken(error)` — engine-only; produced when a rule raises

Scope notes:
- **Meaning A**: a resolved Deferred type is carried by the output data's native dtype and
  the existing `DataTypeValidator` path; there is NO mloda-level reader of a post-compute
  `feature.data_type`, so there is NO write-back step and NO `Feature` marker field.
- Hazard preserved: `Fixed` vs declared-type mismatch still raises loudly at planning;
  a rule that raises while the user declared a type logs a WARNING (not silent) and falls
  back to the declared type.
- Clean break, no compatibility shim. Migration surface: base default, factory passthrough,
  two test files.
- This PR is **core-only**. Registry `try/except` cleanup is a separate follow-up (issue DoD).

## Phases

- [x] **Phase 1 — `RuleResult` union module.** New module
  `mloda/core/abstract_plugins/components/data_type_rule.py` with frozen-dataclass variants
  `Fixed`/`Open`/`Deferred`/`Broken`, type aliases `RuleResult` (plugin-returnable) and
  `RuleOutcome` (incl. `Broken`). Unit tests for construction, equality, `Fixed` payload.
  Standalone — tox green on its own.

- [ ] **Phase 2 — Contract flip + engine reconciliation (atomic).**
  - `FeatureGroup.return_data_type_rule` default returns `Open()`; annotation → `RuleResult`.
  - `engine.set_data_type`: call rule in a guard; on raise → classify
    (`debug` for `ValueError`/`IndexError`/`TypeError`/`KeyError`; `warning` otherwise) + log
    feature name + FG class + exc → `Broken`. Match `RuleResult`:
    - `Fixed(Y)`: reconcile with declared type (`==` or raise mismatch); return `Y`.
    - `Open`: return declared type or `None`.
    - `Deferred`: return declared type or `None` (no planning pin).
    - `Broken`: if declared type → WARN (raise-while-declared) + return declared type; else
      return `None`.
    - `assert_never` default for mypy `--strict` exhaustiveness.
  - Migrate factory passthrough (`dynamic_feature_group_factory.py`) to `RuleResult`.
  - Migrate the two test override files to the new contract.
  - Tests: raising rule does not crash planning; mismatch still raises; raise-while-declared
    warns + keeps declared type; classify levels; `Deferred`/`Open` → no pin.
  - tox green.

## Gate
`tox` must pass after each phase (pytest -n8 --timeout=10, ruff format/check, pip-licenses,
mypy --strict, bandit). Update `EXPECTED_SKIP_COUNT` if skip count changes. Tick the box only
when tox is green.
