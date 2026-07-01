# Epic #518: cross-engine comparison contract

One PR (`feat/comparison-contract-518`). Strict tz model (naive vs aware distinct).
TDD: Red writes failing tests, Green implements, `tox` gates each phase.

## Phases

- [x] Phase 0 - Foundation: `comparison_contract.py` (SemanticDimension, ComparisonContract,
      ColumnSemantics, validate). Pure-logic unit tests. No behavior change. tox GREEN.
- [ ] Phase 1 - Per-framework `column_semantics()` for pandas, polars (+lazy), pyarrow,
      duckdb, sqlite, spark, python_dict, iceberg. Consolidate `is_ordered_arrow_type`.
- [ ] Phase 2 - Port as-of joins onto the contract (behavior-preserving). Keep coercion
      seam + sqlite julianday path. Add tz/unit dimension tests.
- [ ] Phase 3 - Equi-joins on temporal keys. Enforce tz/unit compatibility across sides
      ONLY when both keys temporal; string/ID equi-joins stay legal.
- [ ] Phase 4 - Range/min/max filters. Validate order-compatibility with native datetime
      bound in `BaseFilterEngine.do_filter`.
- [ ] Phase 5 - Time windows + forecasting via `TimeReferenceMixin` (tz/unit aware).
- [ ] Phase 6 - Docs (per-engine capability matrix) + reconcile EXPECTED_SKIP_COUNT.

## Deferred (follow-up issues, per scoping decision)

- Value-inspection fallback for sqlite / python_dict (untrusted-schema scan).
- pandas nullable-vs-promotion coercion (needs dedicated research pass).
- Physical unit ENFORCEMENT beyond carrying it as a parameter.
