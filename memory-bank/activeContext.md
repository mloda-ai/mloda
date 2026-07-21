# Active Context

## Current Work Focus

The active initiative is the **PROPERTY_MAPPING / PropertySpec consumption hardening** epic (from the mloda#750 audit): make the typed option contract safe from authoring through compute, migrate downstream plugins, and retire transitional code.

Core hardening is complete and the full `tox` gate is green (170 skipped, plus format, lint, licenses, strict mypy, and bandit).

## Status

- ✅ Typed `PropertySpec` is the sole `PROPERTY_MAPPING` contract; raw-dict specs are a hard break.
- ✅ Defaults are materialized once at the central compute boundary (framework-enforced), with opt-in explicit-`None` semantics.
- ✅ Structured name parsing and explicit capture-to-spec binding replace reverse lookup and fabricated captureless tokens.
- ✅ Required-presence enforcement on the string-named match path; all-optional universal-matcher guard.
- ✅ Resolution failures carry per-candidate elimination facts (`EvaluationResult`).
- ✅ Post-hardening cleanup: retired transitional parser seams (os-005) and consolidated the `PROPERTY_MAPPING` test suite around a public behavior matrix (os-006).

## Next Steps

- **os-008**: decide effective-options materialization placement and default-equivalent twin canonicalization (follow-up from the #796 / #803 reviews).
- **Phase 6 downstream** (owning repos): migrate mloda-registry raw mappings and captureless patterns, align declared value spaces with runtime behavior, scaffold the plugin-template example, and refresh mloda.ai samples before raising the registry `>=0.10,<0.11` cap.

## Architecture Snapshot

The typed option lifecycle (author -> parse -> bind -> match -> resolve -> materialize defaults -> compute) is documented in `systemPatterns.md`.
