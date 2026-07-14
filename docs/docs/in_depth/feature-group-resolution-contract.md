# Feature Group Resolution Contract

Status: this contract was decided under issue #722 (Stage 1). The engine matcher (`IdentifyFeatureGroupClass`) and the debug matcher (`resolve_feature`) still diverge today; the known divergences are pinned by `tests/test_core/test_resolution_parity/`. Stages 2-4 of #722 implement this contract; nothing below is implemented yet unless a parity test says otherwise. Input-data reader selection (issue #727) and GlobalFilter (issue #728) are out-of-scope follow-ups.

## Boundary model

One authoritative pipeline:

```text
run configuration
  -> EnvironmentBuildOutcome            (environment construction, may fail structurally)
  -> ResolutionEnvironmentSnapshot
  -> FeatureGroupResolver.resolve(request, environment)
  -> ResolutionOutcome
  -> projections
```

`FeatureGroupResolver` is the only component allowed to decide whether a FeatureGroup wins. It never raises for resolution failures; it returns structured state. The engine validates the outcome and raises a typed `FeatureResolutionError` carrying it. Diagnostic surfaces consume the same outcome; they never re-match and never call provider hooks.

## Candidate universe

The candidate universe is exactly the accessible mapping of the environment snapshot: discovered FeatureGroup subclasses, minus collector-disabled, minus registry-strict-mode rejections, after redefinition dedup, each with frameworks = declared intersect run-enabled intersect available. The resolver never calls `get_all_subclasses` and never re-reads registry state.

Environment provenance distinguishes: not discovered, disabled by collector, rejected by policy or strict mode, unavailable, not enabled for this run, redefinition conflict, invalid provider framework declaration, accessible.

## Statuses and invariants

- `ResolutionStatus`: RESOLVED, NOT_FOUND, AMBIGUOUS, FAILED.
- `CandidateResolutionStatus`: WINNER, SURVIVOR (passed every filter, did not win: ambiguous or failed outcome), SHADOWED, REJECTED, FAILED.
- `FrameworkResolutionStatus`: SUPPORTED, NOT_ENABLED, UNAVAILABLE, PIN_EXCLUDED, CAPABILITY_REJECTED, HOOK_FAILED.

Invariants:

- A winner is present if and only if the status is RESOLVED.
- A decision-relevant fatal provider or environment error cannot coexist with RESOLVED.
- Candidate and framework ordering is deterministic by stable plugin identity (`module:qualname`), never set iteration.
- Shadowed candidates stay visible, with who shadowed them and the affected framework set.
- Public records serialize without Python class objects.
- Every rejection carries a structured reason (criteria, domain, scope, abstract class, no accessible framework, capability rejection, framework-pin exclusion, link/index mismatch, provider-hook failure, policy exclusion, subclass shadowing), never only an error string.

## Filter and hook order

The decided order:

1. Normalize and validate the request.
2. Apply environment accessibility and policy.
3. Structural scope filter.
4. Abstract classification (abstract candidates can never win).
5. Domain.
6. Criteria (`match_feature_group_criteria`).
7. Effective frameworks from the environment.
8. Framework pin.
9. Link/index compatibility.
10. Per-feature capability hook (`supports_compute_framework`), on remaining frameworks only.
11. Framework-aware subclass preference.
12. Final uniqueness validation.

Two deliberate changes from current engine behavior:

- Today criteria runs before scope, so irrelevant provider code can observe, and even abort, a scoped-out request.
- Today capability hooks run before the framework pin, so a broken hook on a pinned-away framework is engine-fatal.

The new order runs cheap structural exclusions first and never invokes hooks for candidates or frameworks that are already excluded. Each provider hook is invoked at most once per candidate or per candidate/framework pair. Renderers and diagnostics never invoke hooks. Diagnostic enrichment cannot change the resolution status.

## Subclass policy

Framework-aware preference, the engine's current semantics: a child shadows its parent only when both survived all prior steps and their supported framework sets are equal. Shadowing records the shadowed candidate as SHADOWED. Pure-issubclass collapse (what `resolve_feature` does today) is rejected because it hides real ambiguity that the engine reports.

## Failure precedence

FAILED (a decision-relevant provider or environment error) beats AMBIGUOUS, which beats NOT_FOUND. An otherwise successful candidate never hides a decision-relevant failure that would abort the engine. Failures of candidates or frameworks excluded earlier by policy are not decision-relevant, and their hooks are not invoked at all.

## Provider failure semantics

| Signal | Engine semantics | Diagnostic semantics |
| --- | --- | --- |
| Ordinary non-match (`False`) | Candidate REJECTED | Silent |
| Typed value rejection (`PropertyValueRejection`) | Candidate REJECTED with structured reason | Surfaced identically on both paths |
| Provider contract violation (malformed framework declaration and similar) | FAILED, fail closed | FAILED, fail closed |
| Unexpected provider exception | FAILED, fail closed, never degraded open, never converted to a non-match or an empty framework list | FAILED, fail closed |
| Environment or policy failure | FAILED at environment construction | FAILED at environment construction |

Recorded per failure: plugin identity, hook or stage, exception category, safe message. Never retained: traceback objects. The current `resolve_feature` behavior of degrading a raising capability hook open is explicitly rejected.

## Projections

The three personas from issue #722:

| Persona | Projection |
| --- | --- |
| Data user | Concise actionable message: typed failure, dependency path, suggested fixes, explicit standalone vs exact-run labeling. |
| Data provider | Candidate, hook, and framework-level decisions: the failing hook and its category, deterministic ordering, exactly-once hook invocation. |
| Data steward | Redacted serializable report: stable plugin identity, registry and strict-mode provenance, policy exclusions, effective framework set, environment fingerprint, deterministic output. |

Human-readable strings are projections of structured state. They are never matching inputs and never the only retained representation.

## Serialization and redaction

Stable plugin identity is `module:qualname`. Public records are plain data (strings, numbers, tuples) and serialize without importing plugin classes.

Redaction: credentials, raw data-access objects, secret option values, and unsafe plugin exception content never appear in public or steward projections. The internal request may carry opaque provider values, but the resolver never mutates them.

## Diagnostics modes

Two labelled modes:

- Standalone (default): a fresh default environment, exploratory. It records its origin and does not claim to reproduce a run.
- Exact-run (preflight): the same configuration as `prepare()`, or the stored outcome of an actual planning pass.

Standalone mode runs the authoritative resolver exactly once against its fresh default environment to build its captured outcome. The hook-free guarantee applies to inspecting and projecting an already captured outcome, not to that single build pass.

A session exposes its captured report; inspecting a report never re-matches. `mlodaAPI.explain` and `session.resolved_plan` stay plan-based.
