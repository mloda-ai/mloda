# Epic #518: cross-engine comparison contract

One PR (`feat/comparison-contract-518`). Strict tz model (naive vs aware distinct).
TDD: Red writes failing tests, Green implements, `tox` gates each phase.

## Phases

- [x] Phase 0 - Foundation: `comparison_contract.py` (SemanticDimension, ComparisonContract,
      ColumnSemantics, validate). Pure-logic unit tests. No behavior change. tox GREEN.
- [x] Phase 1 - Per-framework `column_semantics()` for pandas, polars (+lazy), pyarrow,
      sql-family (duckdb/sqlite via arrow), spark, python_dict. Single arrow source of truth
      in sql_type_semantics. tox GREEN (skip count 171).
- [x] Phase 2 - Port as-of joins onto the contract. `_column_semantics` hook (default
      ordered-only; pandas/polars/pyarrow/sqlite override for tz). require_compatible
      adds strict naive-vs-aware guard. Ordered-error + coercion preserved. tox GREEN.
- [x] Phase 3 - Equi-joins on temporal keys (INNER/LEFT/RIGHT/OUTER) via require_compatible
      on key pairs; string/ID joins stay legal. `_column_semantics` now a MANDATORY hook
      (base raises actionable error); wired all 7 built-ins incl new duckdb reader +
      python_dict + spark. tox GREEN.
- [ ] Phase 4 - Range/min/max filters. Validate order-compatibility with native datetime
      bound in `BaseFilterEngine.do_filter`.
- [ ] Phase 5 - Time windows + forecasting via `TimeReferenceMixin` (tz/unit aware).
- [ ] Phase 6 - Docs (per-engine capability matrix) + reconcile EXPECTED_SKIP_COUNT.

## Deferred (follow-up issues, per scoping decision)

- Value-inspection fallback for sqlite / python_dict (untrusted-schema scan).
- pandas nullable-vs-promotion coercion (needs dedicated research pass).
- Physical unit ENFORCEMENT beyond carrying it as a parameter.
