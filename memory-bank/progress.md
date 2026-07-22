# Progress

## Working Features

### Compute Frameworks (9)

PythonDict (dependency-free) needs no extra and SQLite needs only PyArrow; the rest are optional extras: Pandas, PyArrow, Polars (eager and lazy), DuckDB, Iceberg, Spark. DuckDB and SQLite share a common SQL base.

### Feature Groups

- **Core transforms**: Aggregated, TimeWindow, MissingValue (under data quality).
- **Analytics**: Clustering, DimensionalityReduction, Forecasting (scikit-learn); NodeCentrality (numpy graph centrality).
- **Sklearn family**: Pipeline, Encoding, Scaling (shared artifact storage).
- **Processing**: TextCleaning (nltk), GeoDistance.
- **LLM**: API-backed LLM feature groups, CLI features, and tool integrations.
- **Infrastructure**: environment introspection (InstalledPackages, ListDirectory), dynamic feature-group factory, source-input composition, and the input-data reader suite (CSV / Parquet / Feather / ORC / JSON file readers, document readers, SQLite DB reader).

### Extenders

- OtelExtender (OpenTelemetry).

## Recent Achievements

- ✅ FeatureGroup resolution unified around one matcher: `resolve_feature` adapts the same non-raising `EvaluationResult` seam a run uses.
- ✅ Typed resolution failures (`FeatureResolutionError`) with per-candidate elimination facts, `mlodaAPI.diagnose(...)`, and `session.resolution_report()`; resolution debug tooling exported from `mloda.provider`.
- ✅ Post-epic cleanup (os-015): removed remaining dead code and documented the new resolution surfaces.
- ✅ `PROPERTY_MAPPING` hardening: typed `PropertySpec` contract, raw-dict hard break, `property_spec()` authoring path.
- ✅ Framework-enforced default materialization at the compute boundary; opt-in explicit-`None`.
- ✅ Structured parsed-name bindings; required-presence enforcement; all-optional universal-matcher guard.
- ✅ Post-hardening cleanup: retired transitional parser seams; `PROPERTY_MAPPING` test suite consolidated around a public behavior matrix.

## Remaining / Known Issues

- **os-008**: effective-options materialization placement and default-equivalent twin canonicalization.
- **os-016 / os-017 / os-018**: optional resolution refactors (shared `resolve_or_raise` helper, columnwise framework-hook triple, author-time guard split).
- **Phase 6 downstream**: raw-mapping migration and doc/sample refresh in mloda-registry, plugin-template, and mloda.ai.
- **Local FeatureGroup subclass GC race** (mloda#868): shared-fixture test parametrization is deferred until this flake is resolved.
