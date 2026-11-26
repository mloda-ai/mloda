# PROPERTY_MAPPING test behavior matrix

The map for the PROPERTY_MAPPING / PropertySpec hardening suite.
Each behavior is owned by a public-boundary test; private-helper tests survive only where the
helper is itself an intentional contract (listed at the end). Keep this file in step when tests move.

Public boundaries: `FeatureGroup.match_feature_group_criteria`, `FeatureChainParser.parse_feature_name`
/ `extract_property_values` / `build_effective_options`, `FeatureGroup.options_with_defaults`,
`option_key_is_present`, `validate_property_values`, the identify engine (`identify_winner` / `evaluate_or_raise`), and
`mloda.run_all` / compute.

## Axes

| Axis | Values | Owning tests |
|---|---|---|
| Match path | config (options only), name (string-named), bare name, plain group (no mixin) | `test_name_path_validates_option_values.py` (config / name / bare name), `test_property_mapping_enforced_validators.py` (config + name per key), `test_name_path_required_presence.py` / `test_strict_validation_returns_false.py` / `test_required_when_enforced_on_override.py` (plain group) |
| Option source | group, context | `test_options_with_defaults.py` (group vs context fill), `test_retire_transitional_seams.py::TestOptionKeyIsPresentContextStorage` |
| Presence | absent, present-as-None (flagless -> absent), explicit None opt-in -> present, present | `test_explicit_none_opt_in.py`, `test_retire_transitional_seams.py::TestOptionKeyIsPresent*`, `test_property_mapping_enforced_validators.py::TestStrictValidationDoesNotMakeKeysRequired` |
| Required / conditional | required, `required_when` predicate, `default`, optional | `test_property_mapping_required_when.py`, `test_property_spec_builder.py` (default-based `_can_skip_required_check` verdicts), `test_universal_optional_matcher.py` (`required_when` / conditional verdicts) |
| Value shape | scalar, list/tuple/set/frozenset, empty container, string (scalar), dict (composite) | `test_property_mapping_sequence_unpacking.py` (core), `test_property_mapping_container_invariance.py` (shipped plugins) |
| Value space | `allowed_values` membership, `element_validator` (per element), `match_guard` (raw value) | `test_property_mapping_allowed_values.py`, `test_property_mapping_type_constraints.py`, `test_property_mapping_unified_model.py`, `test_strict_validation_returns_false.py::TestPlainGroupStrictValidation` / `TestPlainGroupMatchGuard` (plain group) |
| Callback outcome | returns True/False, raises (expected TypeError, broken-looking KeyError, AttributeError) | `test_option_value_rejection_never_escapes.py` (containment + DEBUG/WARNING tiers + cause chaining), `test_strict_validation_returns_false.py` (plain group: contained ValueError, marked abort, raising guard) |
| Diagnostics | rejection reason surfaced, recorded on first pass, no phantom reason for unrelated groups, `expected` naming a non-strict guard's rejection | `test_option_value_rejection_never_escapes.py` (identify carries reason), `test_property_mapping_enforced_validators.py::TestRejectedValueNamedInResolutionError`, render layer under `test_prepare/`, `test_first_pass_rejection_recording.py::TestExpectedGuardRejectionRecording`, `test_first_pass_rejection_recording.py::TestPlainGroupRejectionRecording` and `test_name_path_required_presence.py::TestPlainGroupPresence` (plain group) |
| Defaults lifecycle | pure `options_with_defaults` view; central materialization at compute | `test_options_with_defaults.py` (view), `test_materialize_defaults_boundary.py` (`calculate_feature` boundary) |
| Construction time | constructor is the schema, shape rules fire in the class body, one default check | `test_property_mapping_spec_schema.py`, `test_property_mapping_spec_shape.py`, `test_property_mapping_unified_model.py::TestOneDefaultCheckImplementation` |
| Definition-time diagnostics | universal all-optional matcher warning, probe containment | `test_universal_optional_matcher.py` |

## Intentional private-helper contracts (kept on purpose)

- `_strict_validation_rejection_reason` (diagnostic facade): asserted directly where the public
  identify path cannot reach the value range, because some invalid values collide with feature-name
  tokens (see the `reported_invalid_value` note in `test_property_mapping_enforced_validators.py`).
  Kept for the collision-free reason content and for the "no phantom reason for an unrelated group"
  scoping contract.
- `option_key_is_present`: module-level, public; the one presence helper
  (`test_retire_transitional_seams.py`).
- `_can_skip_required_check` truth table: owned by `test_property_spec_builder.py`.
- `_UNIVERSAL_MATCHER_PROBE_NAME` and the probe DEBUG record: definition-time machinery
  (`test_universal_optional_matcher.py`).

## Retired seams (absence pinned, in one place)

`test_retire_transitional_seams.py::TestTransitionalSeamsAreGone` pins that the deleted private
wrappers stay gone: `_is_context_parameter`, `_is_strict_validation`, `_get_element_validator`,
`_get_validation_function`, the `_extract_property_values` alias, the `_validate_type_validators`
rename, and the mixin `_build_effective_options` pass-through.

## Deferred

- FeatureGroup-shape parametrization across `test_property_mapping_sequence_unpacking.py`, `test_property_mapping_unified_model.py` and `test_property_mapping_type_constraints.py` stays organizational duplication: moving these classes into shared modules needs its own change (autouse registry-leak guards, local-subclass GC flake).
- Cross-layer `_strict_validation_rejection_reason` coverage in the `test_prepare/` render layer is the public reason surface and stays there.

## Where each invariant is pinned

| Invariant | Test |
| --- | --- |
| `PropertySpec` construction: defaults, immutability, `allowed_values` normalization, flag and callable rules, strict needs a value space | `tests/.../feature_chainer/test_property_spec_type.py` |
| `name_converter` applies to named captures, runs before validation, and a raising one is a rejected value | `tests/.../feature_chainer/test_parsed_name_bindings.py` |
| Unknown or mistyped field caught by mypy at author time | `tests/.../feature_chainer/test_property_spec_author_time.py` |
| `PROPERTY_MAPPING` accepts nothing but `PropertySpec`, and the pipeline behaves as before | `tests/.../feature_chainer/test_property_spec_hard_break.py` |
| The constructor IS the schema (no key set left to curate) | `tests/.../feature_chainer/test_property_mapping_spec_schema.py` |
| Shape rules, relocated from class definition to construction | `tests/.../feature_chainer/test_property_mapping_spec_shape.py` |
| Declared-default invariant | `tests/test_core/test_abstract_plugins/test_property_mapping_default_invariant.py` |
| Declared defaults materialized into runtime options | `tests/test_core/test_abstract_plugins/test_options_with_defaults.py` |
| A declared handle stays on the declaring feature (a non-declaring consumer forwards it like any group option) and is never a named capture | `tests/test_core/test_abstract_plugins/test_components/test_credential_handle_spec.py` |
| A credential value in a non-pointer option is refused at plan time | `tests/test_core/test_abstract_plugins/test_components/test_credential_handle_spec.py` |
| `consumer_local_keys` and the explicit-allowlist override in `Options.inherit_from` | `tests/test_core/test_abstract_plugins/test_components/test_options_inherit_from.py` |
| Declared defaults materialized at the compute boundary | `tests/test_core/test_abstract_plugins/test_materialize_defaults_boundary.py` |
| Declared defaults materialized at intake, canonicalizing default-equivalent twins | `tests/test_core/test_abstract_plugins/test_intake_default_canonicalization.py` |
| A context key that does not reach an input group declaring it as a group option logs one plan warning | `tests/test_core/test_core/test_uncascaded_context_warning.py` |
| Filter criteria matching observes effective options | `tests/test_core/test_filter/test_filter_criteria_effective_options.py` |
| The shared declaration mechanics: the per-key validator | `tests/.../test_components/test_declaration_surface.py` |
| A pattern-less group: enforced `required_when`, and defaults it applies itself | `tests/.../feature_chainer/test_required_when_enforced_on_override.py`, `tests/test_core/test_abstract_plugins/test_intake_default_canonicalization.py` |
| A plain group enforces required keys, strict values and `match_guard`, also through an override | `tests/.../feature_chainer/test_name_path_required_presence.py`, `test_strict_validation_returns_false.py`, `test_required_when_enforced_on_override.py` |
| A plain group's rejection reasons reach the failure report | `tests/test_core/test_prepare/test_first_pass_rejection_recording.py` |
| Container invariance, no stringification, str-as-scalar, dict-as-composite, empty containers | `tests/.../feature_chainer/test_property_mapping_sequence_unpacking.py` |
| `scalar=True` refuses containers at match time, in `run(parameters=...)` and in `validate_property_values` | `test_property_mapping_sequence_unpacking.py`, `tests/test_core/test_api/test_prepare_run.py`, `tests/.../test_components/test_validate_property_values.py` |
| Present option values validated on the string-named path too | `tests/.../feature_chainer/test_name_path_validates_option_values.py` |
| Required presence on the string-named path: the mandatory non-match, the retired env var stays ignored, and the `deferred_binding` / `in_features` exemptions | `tests/.../feature_chainer/test_name_path_required_presence.py` |
| A declared option or `in_features` that contradicts an owned name aborts the match, and the env var downgrades it | `tests/test_core/test_abstract_plugins/test_components/test_forwarded_name_mismatch.py` |
| Empty operands and the source count on the name path | `tests/.../feature_chainer/test_in_feature_count_gate_name_sources.py` |
| `required_when` survives an overridden matcher, runs exactly once, and demands a classmethod | `tests/.../feature_chainer/test_required_when_enforced_on_override.py` |
| A `required_when` non-match records its reason, so the failure report names the key and its owner | `tests/test_core/test_prepare/test_required_when_rejection_recording.py` |
| `required_when` end to end, including the context-key remedy in the failure report | `tests/.../feature_chainer/test_property_mapping_required_when.py` |
| Plugin specs behave identically across containers | `tests/test_plugins/feature_group/experimental/test_property_mapping_container_invariance.py` |
| `property_spec` builder surface | `tests/.../feature_chainer/test_property_spec_builder.py` |
| A `group_only` key in the context is refused at plan time | `tests/test_core/test_api/test_prepare_run.py` |
| Rejection reasons surfaced to the end user | `tests/test_core/test_prepare/test_identify_feature_group_error_message.py` |
| The all-optional universal-matcher diagnostic and its `ALLOW_UNIVERSAL_MATCHER` escape hatch | `tests/.../feature_chainer/test_universal_optional_matcher.py` |
| `NAME_ONLY_MATCH` (name-owned match only, no configuration path) | `tests/.../feature_chainer/test_feature_chain_parser_mixin.py` (`TestNameOnlyMatch`) |
| The missing-`in_features` definition-time diagnostic and its zero-source rejection | `tests/.../feature_chainer/test_missing_in_features_declaration.py` (the diagnostic), `tests/.../feature_chainer/test_in_feature_count_gate_name_sources.py` (the option-path zero-source non-match records no rejection) |
| `expected` reports a `match_guard` rejection on a non-strict spec, with the capped value echo | `tests/test_core/test_prepare/test_first_pass_rejection_recording.py` |
