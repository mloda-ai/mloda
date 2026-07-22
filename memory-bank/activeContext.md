# Active Context

## Current Work Focus

Two core epics have landed their delivery; the workspace is in a post-epic cleanup phase.

- **FeatureGroup resolution** (mloda#722, board epic `feature-group-resolution`): unified around the engine's `IdentifyFeatureGroupClass` matcher, adapted in place, with no replacement resolver package.
- **PROPERTY_MAPPING / PropertySpec consumption hardening** (mloda#750 audit): the typed option contract is safe from authoring through compute.

The full `tox` gate is green (tests, format, lint, licenses, strict mypy, bandit).

## Status

### FeatureGroup resolution (delivered)

- ✅ One authoritative matcher: `resolve_feature` is a thin adapter over the same non-raising `EvaluationResult` seam a run uses; raising versus returning is presentation-only.
- ✅ Typed failures: `FeatureResolutionError` (a `ValueError` subclass) carries `.feature_name`, `.result`, and `.partial_records`.
- ✅ Non-raising diagnostics: `mlodaAPI.diagnose(...)` returns a `ResolutionDiagnosis`; `session.resolution_report()` returns per-feature `ResolutionRecord`s.
- ✅ Per-candidate elimination facts captured in `EvaluationResult`; resolution debug tooling exported from `mloda.provider`; a blessed, contract-pinned resolution test seam (os-014).
- ✅ Post-epic cleanup: removed the last dead symbols and documented the new surfaces (troubleshooting page, `mloda-api.md`).

### PROPERTY_MAPPING hardening (complete)

- ✅ Typed `PropertySpec` is the sole `PROPERTY_MAPPING` contract; raw-dict specs are a hard break.
- ✅ Defaults are materialized once at the central compute boundary, with opt-in explicit-`None` semantics.
- ✅ Structured name parsing and explicit capture-to-spec binding; required-presence enforcement; all-optional universal-matcher guard.
- ✅ Post-hardening cleanup: retired transitional parser seams (os-005); consolidated the test suite around a public behavior matrix (os-006).

## Next Steps

- **os-008**: decide effective-options materialization placement and default-equivalent twin canonicalization.
- **os-016 / os-017 / os-018**: optional resolution refactors (shared `resolve_or_raise` helper, columnwise framework-hook triple, splitting the author-time guard subsystem out of `feature_chain_parser.py`).
- **Phase 6 downstream** (owning repos): migrate mloda-registry raw mappings and captureless patterns, scaffold the plugin-template example, refresh mloda.ai samples before raising the registry `>=0.10,<0.11` cap.

## Architecture Snapshot

The typed option lifecycle (author -> parse -> bind -> match -> resolve -> materialize defaults -> compute) is documented in `systemPatterns.md`. Resolution runs one matcher, builds one environment, evaluates once, and renders failures as a pure projection of the captured `EvaluationResult`.
