# PROPERTY_MAPPING test behavior matrix

The map for the PROPERTY_MAPPING / PropertySpec hardening suite (mloda#750 epic, phase 7).
Each behavior is owned by a public-boundary test; private-helper tests survive only where the
helper is itself an intentional contract (listed at the end). Keep this file in step when tests move.

Public boundaries: `FeatureGroup.match_feature_group_criteria`, `FeatureChainParser.parse_feature_name`
/ `extract_property_values` / `build_effective_options`, `FeatureGroup.options_with_defaults`,
`option_key_is_present`, the identify engine (`identify_winner` / `evaluate_or_raise`), and
`mloda.run_all` / compute.

## Axes

| Axis | Values | Owning tests |
|---|---|---|
| Match path | config (options only), name (string-named), bare name | `test_name_path_validates_option_values.py` (config / name / bare name), `test_property_mapping_enforced_validators.py` (config + name per key) |
| Option source | group, context | `test_options_with_defaults.py` (group vs context fill), `test_retire_transitional_seams.py::TestOptionKeyIsPresentContextStorage` |
| Presence | absent, present-as-None (flagless -> absent), explicit None opt-in -> present, present | `test_explicit_none_opt_in.py`, `test_retire_transitional_seams.py::TestOptionKeyIsPresent*`, `test_property_mapping_enforced_validators.py::TestStrictValidationDoesNotMakeKeysRequired` |
| Required / conditional | required, `required_when` predicate, `default`, optional | `test_property_mapping_required_when.py`, `test_property_spec_builder.py` (default-based `_can_skip_required_check` verdicts), `test_universal_optional_matcher.py` (`required_when` / conditional verdicts) |
| Value shape | scalar, list/tuple/set/frozenset, empty container, string (scalar), dict (composite) | `test_property_mapping_sequence_unpacking.py` (core), `test_property_mapping_container_invariance.py` (shipped plugins) |
| Value space | `allowed_values` membership, `element_validator` (per element), `match_guard` (raw value) | `test_property_mapping_allowed_values.py`, `test_property_mapping_type_constraints.py`, `test_property_mapping_unified_model.py` |
| Callback outcome | returns True/False, raises (expected TypeError, broken-looking KeyError, AttributeError) | `test_option_value_rejection_never_escapes.py` (containment + DEBUG/WARNING tiers + cause chaining) |
| Diagnostics | rejection reason surfaced, recorded on first pass, no phantom reason for unrelated groups | `test_option_value_rejection_never_escapes.py` (identify carries reason), `test_property_mapping_enforced_validators.py::TestRejectedValueNamedInResolutionError`, render layer under `test_prepare/` |
| Defaults lifecycle | pure `options_with_defaults` view; central materialization at compute | `test_options_with_defaults.py` (view), `test_materialize_defaults_boundary.py` (`calculate_feature` boundary) |
| Construction time | constructor is the schema, shape rules fire in the class body, one default check | `test_property_mapping_spec_schema.py`, `test_property_mapping_spec_shape.py`, `test_property_mapping_unified_model.py::TestOneDefaultCheckImplementation` |
| Definition-time diagnostics | universal all-optional matcher warning, probe containment | `test_universal_optional_matcher.py` |

## Intentional private-helper contracts (kept on purpose)

- `_strict_validation_rejection_reason` (diagnostic facade): asserted directly where the public
  identify path cannot reach the value range, because some invalid values collide with feature-name
  tokens (see the `reported_invalid_value` note in `test_property_mapping_enforced_validators.py`).
  Kept for the collision-free reason content and for the "no phantom reason for an unrelated group"
  scoping contract.
- `option_key_is_present`: module-level, public; the one #768 presence helper
  (`test_retire_transitional_seams.py`).
- `_can_skip_required_check` truth table: owned by `test_property_spec_builder.py`.
- `_UNIVERSAL_MATCHER_PROBE_NAME` and the probe DEBUG record: #771 definition-time machinery
  (`test_universal_optional_matcher.py`).

## Retired seams (absence pinned, in one place)

`test_retire_transitional_seams.py::TestTransitionalSeamsAreGone` pins that the deleted private
wrappers stay gone: `_is_context_parameter`, `_is_strict_validation`, `_get_element_validator`,
`_get_validation_function`, the `_extract_property_values` alias, the `_validate_type_validators`
rename, and the mixin `_build_effective_options` pass-through.

## Deferred (documented, not silently dropped)

- FeatureGroup-shape parametrization: single-key `element_validator` / `match_guard` / membership /
  `required_when` shapes recur across `test_property_mapping_sequence_unpacking.py`,
  `test_property_mapping_unified_model.py`, and `test_property_mapping_type_constraints.py`. Not
  collapsed here: several sibling files run autouse registry-leak guards and the local-subclass GC
  race is a known flake source (mloda#868), so moving these classes into shared modules needs its
  own change. The behavior is already owned above; the remaining duplication is organizational.
- Cross-layer `_strict_validation_rejection_reason` coverage in the `test_prepare/` render layer is
  the public reason surface and stays where it is.
