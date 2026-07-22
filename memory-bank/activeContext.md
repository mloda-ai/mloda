# Active Context

## Current Work Focus

The active initiative is the **PROPERTY_MAPPING / PropertySpec consumption hardening** epic (from the mloda#750 audit): make the typed option contract safe from authoring through compute, migrate downstream plugins, and retire transitional code.

Core hardening is complete and the full `tox` gate is green (170 skipped, plus format, lint, licenses, strict mypy, and bandit).

## Status

- ✅ Typed `PropertySpec` is the sole `PROPERTY_MAPPING` contract; raw-dict specs are a hard break.
- ✅ Defaults are materialized at feature intake (framework-enforced), with opt-in explicit-`None` semantics; default-equivalent twin requests canonicalize during planning, and the compute boundary remains an idempotent safety net (os-008).
- ✅ Structured name parsing and explicit capture-to-spec binding replace reverse lookup and fabricated captureless tokens.
- ✅ Required-presence enforcement on the string-named match path; all-optional universal-matcher guard.
- ✅ Resolution failures carry per-candidate elimination facts (`EvaluationResult`).
- ✅ Post-hardening cleanup: retired transitional parser seams (os-005) and consolidated the `PROPERTY_MAPPING` test suite around a public behavior matrix (os-006).

## What Works

- **Compute frameworks (9)**: PythonDict (dependency-free) and SQLite (PyArrow only); optional extras cover Pandas, PyArrow, Polars (eager and lazy), DuckDB, Iceberg, Spark. DuckDB and SQLite share a common SQL base.
- **Feature groups**: core transforms (Aggregated, TimeWindow, MissingValue), analytics (Clustering, DimensionalityReduction, Forecasting, NodeCentrality), sklearn family (Pipeline, Encoding, Scaling), processing (TextCleaning, GeoDistance), LLM feature groups, and infrastructure (environment introspection, dynamic factory, source-input composition, input-data reader suite).
- **Extenders**: OtelExtender (OpenTelemetry).

## Next Steps / Known Issues

- **Phase 6 downstream** (owning repos): migrate mloda-registry raw mappings and captureless patterns, align declared value spaces with runtime behavior, scaffold the plugin-template example, and refresh mloda.ai samples before raising the registry `>=0.10,<0.11` cap.
- **Local FeatureGroup subclass GC race** (mloda#868): shared-fixture test parametrization is deferred until this flake is resolved.

## Architecture Snapshot

The typed option lifecycle (author -> parse -> bind -> match -> resolve -> materialize defaults -> compute) is documented in `systemPatterns.md`.
