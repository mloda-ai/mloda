"""Failing tests pinning the Phase 1 declarative subtype universe contract (issue #639)."""

from abc import abstractmethod
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, property_spec


SUBTYPE_KEY_NAME = "sustatu_window_function"
WINDOW_UNIVERSE = frozenset({"median", "sum", "lag", "ntile"})
BETA_SUPPORTED = frozenset({"sum", "lag"})
FLATTENED_UNIVERSE = frozenset({"rows_7", "range_7", "rows_all"})
PLAIN_FEATURE = "sustatu_plain_feature"
FLATTENED_FEATURE = "sustatu_flattened_feature"


def _sustatu_is_positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


class SubtypeUFwAlpha(ComputeFramework):
    """First dummy compute framework for subtype-universe tests."""


class SubtypeUFwBeta(ComputeFramework):
    """Second dummy compute framework for subtype-universe tests."""


class SubtypeUWindowBaseFG(FeatureChainParserMixin, FeatureGroup):
    """Shape A declaration: SUBTYPE_KEY with allowed values plus a parametric family."""

    SUBTYPE_KEY = SUBTYPE_KEY_NAME
    PARAMETRIC_SUBTYPE_FAMILIES = {"ntile": "N-tile bucketing"}
    PREFIX_PATTERN = r".*__([\w]+)_sustatw$"
    PROPERTY_MAPPING = {
        SUBTYPE_KEY_NAME: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum", "lag": "Lag"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeUFwAlpha, SubtypeUFwBeta}


class SubtypeUWindowFG(SubtypeUWindowBaseFG):
    """Concrete window family member narrowing subtype support on SubtypeUFwBeta."""

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is SubtypeUFwBeta:
            return BETA_SUPPORTED
        return WINDOW_UNIVERSE


class SubtypeUAbstractWindowFG(SubtypeUWindowBaseFG):
    """Abstract member of the window family; declares the universe, not support."""

    @abstractmethod
    def _sustatu_backend_marker(self) -> None: ...


class SubtypeUOverreachFG(SubtypeUWindowBaseFG):
    """Declares support for a subtype outside its universe; the audit must flag it."""

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        return WINDOW_UNIVERSE | {"sustatu_bogus"}


class SubtypeUExplicitOverrideFG(SubtypeUWindowFG):
    """Explicit supports_compute_framework override; must win over the derived gate."""

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return True


class SubtypeUPlainFG(FeatureGroup):
    """Family without any subtype dimension."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeUFwAlpha, SubtypeUFwBeta}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PLAIN_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubtypeUFlattenedFG(FeatureGroup):
    """Shape B declaration: flattened frame_type x frame_unit subtype, SUBTYPE_KEY stays None."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeUFwAlpha, SubtypeUFwBeta}

    @classmethod
    def subtype_universe(cls) -> frozenset[str]:
        return FLATTENED_UNIVERSE

    @classmethod
    def resolve_subtype(cls, feature_name: FeatureName | str, options: Options) -> Optional[str]:
        frame_type = options.get("sustatu_frame_type")
        frame_unit = options.get("sustatu_frame_unit")
        if frame_type is None or frame_unit is None:
            return None
        return f"{frame_type}_{frame_unit}"

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == FLATTENED_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubtypeUPredicateUniverseFG(FeatureChainParserMixin, FeatureGroup):
    """Shape A with a predicate-only key made legal by an explicit subtype_universe override."""

    SUBTYPE_KEY = "sustatu_pred_key"
    PREFIX_PATTERN = r".*__([\w]+)_sustatp$"
    PROPERTY_MAPPING = {
        "sustatu_pred_key": property_spec(
            "Predicate-validated subtype.",
            strict=True,
            validation_function=_sustatu_is_positive_int,
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeUFwAlpha, SubtypeUFwBeta}

    @classmethod
    def subtype_universe(cls) -> frozenset[str]:
        return frozenset({"p1", "p2"})


class SubtypeUValueSpaceFG(FeatureGroup):
    """PROPERTY_MAPPING with spec-form, list-form, legacy flattened and predicate-only keys."""

    PROPERTY_MAPPING = {
        "sustatu_algo": property_spec("Algorithm.", strict=True, allowed_values={"a1": "A one", "a2": "A two"}),
        "sustatu_kind": property_spec("Kind.", strict=True, allowed_values=["x1", "x2"]),
        "sustatu_bucket": property_spec("Bucket count.", strict=True, allowed_values={2: "two", 7: "seven"}),
        "sustatu_mode": {
            "fast": "Fast mode",
            "slow": "Slow mode",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "sustatu_n": property_spec("Predicate-only key.", strict=True, validation_function=_sustatu_is_positive_int),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestContract01SubtypeKeyDefault:
    """Contract 1: SUBTYPE_KEY class attribute with default None."""

    def test_base_default_is_none(self) -> None:
        assert FeatureGroup.SUBTYPE_KEY is None

    def test_untouched_subclass_default_is_none(self) -> None:
        assert SubtypeUPlainFG.SUBTYPE_KEY is None

    def test_declaring_subclass_carries_key(self) -> None:
        assert FeatureGroup.SUBTYPE_KEY is None
        assert SubtypeUWindowBaseFG.SUBTYPE_KEY == SUBTYPE_KEY_NAME


class TestContract02ParametricSubtypeFamiliesDefault:
    """Contract 2: PARAMETRIC_SUBTYPE_FAMILIES class attribute with default None."""

    def test_base_default_is_none(self) -> None:
        assert FeatureGroup.PARAMETRIC_SUBTYPE_FAMILIES is None

    def test_untouched_subclass_default_is_none(self) -> None:
        assert SubtypeUPlainFG.PARAMETRIC_SUBTYPE_FAMILIES is None

    def test_declaring_subclass_carries_families(self) -> None:
        assert FeatureGroup.PARAMETRIC_SUBTYPE_FAMILIES is None
        assert SubtypeUWindowBaseFG.PARAMETRIC_SUBTYPE_FAMILIES == {"ntile": "N-tile bucketing"}


class TestContract03DeclaredOptionValues:
    """Contract 3: declared_option_values(key) returns the declared value space of a key."""

    def test_property_spec_dict_allowed_values(self) -> None:
        assert SubtypeUValueSpaceFG.declared_option_values("sustatu_algo") == frozenset({"a1", "a2"})

    def test_property_spec_list_allowed_values(self) -> None:
        assert SubtypeUValueSpaceFG.declared_option_values("sustatu_kind") == frozenset({"x1", "x2"})

    def test_non_string_values_are_stringified(self) -> None:
        values = SubtypeUValueSpaceFG.declared_option_values("sustatu_bucket")
        assert values == frozenset({"2", "7"})
        assert all(isinstance(value, str) for value in values)

    def test_legacy_flattened_form_excludes_reserved_keys(self) -> None:
        assert SubtypeUValueSpaceFG.declared_option_values("sustatu_mode") == frozenset({"fast", "slow"})

    def test_predicate_only_key_has_empty_value_space(self) -> None:
        assert SubtypeUValueSpaceFG.declared_option_values("sustatu_n") == frozenset()

    def test_absent_key_returns_empty(self) -> None:
        assert SubtypeUValueSpaceFG.declared_option_values("sustatu_absent") == frozenset()

    def test_property_mapping_none_returns_empty(self) -> None:
        assert SubtypeUPlainFG.declared_option_values("anything") == frozenset()

    def test_result_is_frozenset(self) -> None:
        assert isinstance(SubtypeUValueSpaceFG.declared_option_values("sustatu_algo"), frozenset)
        assert isinstance(SubtypeUPlainFG.declared_option_values("anything"), frozenset)


class TestContract04SubtypeUniverse:
    """Contract 4: subtype_universe() = declared values of SUBTYPE_KEY plus parametric family names."""

    def test_universe_unions_declared_values_and_families(self) -> None:
        assert SubtypeUWindowBaseFG.subtype_universe() == WINDOW_UNIVERSE

    def test_universe_empty_without_subtype_key(self) -> None:
        assert SubtypeUPlainFG.subtype_universe() == frozenset()
        assert FeatureGroup.subtype_universe() == frozenset()

    def test_universe_is_frozenset(self) -> None:
        assert isinstance(SubtypeUWindowBaseFG.subtype_universe(), frozenset)

    def test_universe_override_is_honored(self) -> None:
        # supported_subtypes is the base-provided default; red today via AttributeError.
        assert SubtypeUFlattenedFG.supported_subtypes(SubtypeUFwAlpha) == FLATTENED_UNIVERSE
        assert SubtypeUFlattenedFG.subtype_universe() == FLATTENED_UNIVERSE


class TestContract05SupportedSubtypes:
    """Contract 5: supported_subtypes(cfw) defaults to the full universe; subclasses narrow."""

    def test_default_is_full_universe(self) -> None:
        assert SubtypeUWindowBaseFG.supported_subtypes(SubtypeUFwAlpha) == WINDOW_UNIVERSE
        assert SubtypeUWindowBaseFG.supported_subtypes(SubtypeUFwBeta) == WINDOW_UNIVERSE

    def test_subclass_narrows_per_framework(self) -> None:
        assert SubtypeUWindowFG.supported_subtypes(SubtypeUFwBeta) < SubtypeUWindowFG.subtype_universe()
        assert SubtypeUWindowFG.supported_subtypes(SubtypeUFwBeta) == BETA_SUPPORTED
        assert SubtypeUWindowFG.supported_subtypes(SubtypeUFwAlpha) == WINDOW_UNIVERSE

    def test_default_follows_universe_override(self) -> None:
        assert SubtypeUFlattenedFG.supported_subtypes(SubtypeUFwBeta) == FLATTENED_UNIVERSE

    def test_predicate_only_key_with_universe_override(self) -> None:
        assert SubtypeUPredicateUniverseFG.supported_subtypes(SubtypeUFwAlpha) == frozenset({"p1", "p2"})


class TestContract06ResolveSubtype:
    """Contract 6: resolve_subtype resolves the raw subtype from the name, then options; never raises."""

    def test_resolves_from_feature_name(self) -> None:
        assert SubtypeUWindowBaseFG.resolve_subtype("value__median_sustatw", Options()) == "median"

    def test_accepts_feature_name_object(self) -> None:
        assert SubtypeUWindowBaseFG.resolve_subtype(FeatureName("value__lag_sustatw"), Options()) == "lag"

    def test_parametric_instance_resolves_raw(self) -> None:
        assert SubtypeUWindowBaseFG.resolve_subtype("value__ntile_2_sustatw", Options()) == "ntile_2"

    def test_bare_chained_name_does_not_resolve_from_name_and_never_raises(self) -> None:
        assert SubtypeUWindowBaseFG.resolve_subtype("__sum_sustatw", Options()) is None

    def test_bare_chained_name_falls_back_to_options(self) -> None:
        options = Options(group={SUBTYPE_KEY_NAME: "median"})
        assert SubtypeUWindowBaseFG.resolve_subtype("__sum_sustatw", options) == "median"

    def test_resolves_from_options_when_name_does_not_parse(self) -> None:
        options = Options(group={SUBTYPE_KEY_NAME: "median"})
        assert SubtypeUWindowBaseFG.resolve_subtype("sustatu_unchained", options) == "median"

    def test_options_value_is_stringified(self) -> None:
        options = Options(group={SUBTYPE_KEY_NAME: 7})
        assert SubtypeUWindowBaseFG.resolve_subtype("sustatu_unchained", options) == "7"

    def test_name_parsing_takes_precedence_over_options(self) -> None:
        options = Options(group={SUBTYPE_KEY_NAME: "median"})
        assert SubtypeUWindowBaseFG.resolve_subtype("value__sum_sustatw", options) == "sum"

    def test_none_when_neither_resolves(self) -> None:
        assert SubtypeUWindowBaseFG.resolve_subtype("sustatu_unchained", Options()) is None

    def test_none_when_subtype_key_is_none(self) -> None:
        options = Options(group={SUBTYPE_KEY_NAME: "median"})
        assert SubtypeUPlainFG.resolve_subtype("value__sum_sustatw", options) is None

    def test_shape_b_override_resolves_flattened_subtype(self) -> None:
        # Base default must exist and stay None-returning; red today via AttributeError.
        assert FeatureGroup.resolve_subtype("anything", Options()) is None
        options = Options(group={"sustatu_frame_type": "rows", "sustatu_frame_unit": "7"})
        assert SubtypeUFlattenedFG.resolve_subtype(FLATTENED_FEATURE, options) == "rows_7"


class TestContract07CanonicalSubtype:
    """Contract 7: canonical_subtype collapses <family>_<digits> to the family, else identity."""

    def test_parametric_instance_collapses_to_family(self) -> None:
        assert SubtypeUWindowBaseFG.canonical_subtype("ntile_2") == "ntile"

    def test_stem_that_is_not_a_family_stays_identity(self) -> None:
        assert SubtypeUWindowBaseFG.canonical_subtype("ntile_2_3") == "ntile_2_3"

    def test_plain_subtype_stays_identity(self) -> None:
        assert SubtypeUWindowBaseFG.canonical_subtype("median") == "median"

    def test_declared_value_is_not_a_family(self) -> None:
        assert SubtypeUWindowBaseFG.canonical_subtype("sum_2") == "sum_2"

    def test_non_digit_suffix_stays_identity(self) -> None:
        assert SubtypeUWindowBaseFG.canonical_subtype("ntile_x") == "ntile_x"

    def test_without_families_everything_is_identity(self) -> None:
        assert SubtypeUPlainFG.canonical_subtype("ntile_2") == "ntile_2"
        assert FeatureGroup.canonical_subtype("anything_3") == "anything_3"


class TestContract08DerivedSupportsComputeFramework:
    """Contract 8: the default supports_compute_framework is derived from the declaration."""

    def test_base_keeps_returning_true(self) -> None:
        assert FeatureGroup.subtype_universe() == frozenset()
        assert FeatureGroup.supports_compute_framework("anything", Options(), SubtypeUFwAlpha) is True

    def test_empty_universe_is_open(self) -> None:
        assert SubtypeUPlainFG.subtype_universe() == frozenset()
        assert SubtypeUPlainFG.supports_compute_framework(PLAIN_FEATURE, Options(), SubtypeUFwBeta) is True

    def test_unresolved_subtype_is_open(self) -> None:
        assert SubtypeUWindowFG.resolve_subtype("sustatu_unchained", Options()) is None
        assert SubtypeUWindowFG.supports_compute_framework("sustatu_unchained", Options(), SubtypeUFwBeta) is True

    def test_unknown_subtype_is_open(self) -> None:
        assert SubtypeUWindowFG.canonical_subtype("zzz") == "zzz"
        assert SubtypeUWindowFG.supports_compute_framework("value__zzz_sustatw", Options(), SubtypeUFwBeta) is True

    def test_declared_subtype_gated_by_supported_subtypes(self) -> None:
        assert SubtypeUWindowFG.supports_compute_framework("value__median_sustatw", Options(), SubtypeUFwBeta) is False
        assert SubtypeUWindowFG.supports_compute_framework("value__median_sustatw", Options(), SubtypeUFwAlpha) is True
        assert SubtypeUWindowFG.supports_compute_framework("value__sum_sustatw", Options(), SubtypeUFwBeta) is True

    def test_parametric_instance_gated_via_canonical_family(self) -> None:
        assert SubtypeUWindowFG.supports_compute_framework("value__ntile_4_sustatw", Options(), SubtypeUFwBeta) is False
        assert SubtypeUWindowFG.supports_compute_framework("value__ntile_4_sustatw", Options(), SubtypeUFwAlpha) is True

    def test_options_resolved_subtype_is_gated_too(self) -> None:
        options = Options(group={SUBTYPE_KEY_NAME: "median"})
        assert SubtypeUWindowFG.supports_compute_framework("sustatu_unchained", options, SubtypeUFwBeta) is False

    def test_explicit_override_wins(self) -> None:
        # The derived gate on the parent rejects the subtype, the child's explicit override wins.
        assert SubtypeUWindowFG.supports_compute_framework("value__median_sustatw", Options(), SubtypeUFwBeta) is False
        assert SubtypeUExplicitOverrideFG.supported_subtypes(SubtypeUFwBeta) == BETA_SUPPORTED
        result = SubtypeUExplicitOverrideFG.supports_compute_framework(
            "value__median_sustatw", Options(), SubtypeUFwBeta
        )
        assert result is True


class TestContract09SubtypeSupportMatrix:
    """Contract 9: subtype_support_matrix() audit surface per compute framework."""

    def test_matrix_maps_every_declared_framework(self) -> None:
        matrix = SubtypeUWindowFG.subtype_support_matrix()
        assert matrix == {
            SubtypeUFwAlpha.get_class_name(): WINDOW_UNIVERSE,
            SubtypeUFwBeta.get_class_name(): BETA_SUPPORTED,
        }

    def test_matrix_empty_for_abstract_class(self) -> None:
        assert SubtypeUAbstractWindowFG.subtype_support_matrix() == {}

    def test_matrix_empty_for_empty_universe(self) -> None:
        assert SubtypeUPlainFG.subtype_support_matrix() == {}

    def test_matrix_rejects_support_outside_universe(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SubtypeUOverreachFG.subtype_support_matrix()
        message = str(exc_info.value)
        assert "sustatu_bogus" in message
        assert "SubtypeUOverreachFG" in message


class TestContract10DeclarationValidation:
    """Contract 10: exactly two legal declaration shapes, enforced at class definition time."""

    def test_shape_a_declaration_is_legal(self) -> None:
        # SubtypeUWindowBaseFG was defined at module scope without error; pin its derived universe.
        assert SubtypeUWindowBaseFG.subtype_universe() == WINDOW_UNIVERSE

    def test_subtype_key_absent_from_property_mapping_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubtypeUBrokenAbsentKeyFG(FeatureGroup):
                SUBTYPE_KEY = "sustatu_missing_key"
                PROPERTY_MAPPING = {
                    "sustatu_other": property_spec("Other.", strict=True, allowed_values={"v1": "V one"}),
                }

        message = str(exc_info.value)
        assert "SubtypeUBrokenAbsentKeyFG" in message
        assert "sustatu_missing_key" in message

    def test_subtype_key_without_property_mapping_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubtypeUBrokenNoMappingFG(FeatureGroup):
                SUBTYPE_KEY = "sustatu_missing_key"

        message = str(exc_info.value)
        assert "SubtypeUBrokenNoMappingFG" in message
        assert "sustatu_missing_key" in message

    def test_predicate_only_value_space_without_universe_override_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubtypeUBrokenPredicateOnlyFG(FeatureGroup):
                SUBTYPE_KEY = "sustatu_pred_only_key"
                PROPERTY_MAPPING = {
                    "sustatu_pred_only_key": property_spec(
                        "Predicate-only subtype key.",
                        strict=True,
                        validation_function=_sustatu_is_positive_int,
                    ),
                }

        message = str(exc_info.value)
        assert "SubtypeUBrokenPredicateOnlyFG" in message
        assert "sustatu_pred_only_key" in message

    def test_predicate_only_key_with_universe_override_is_legal(self) -> None:
        # SubtypeUPredicateUniverseFG was defined at module scope without error. Its key has no
        # enumerable value space, which is exactly why the explicit universe override is required.
        assert SubtypeUPredicateUniverseFG.declared_option_values("sustatu_pred_key") == frozenset()
        assert SubtypeUPredicateUniverseFG.subtype_universe() == frozenset({"p1", "p2"})

    def test_universe_override_alone_without_subtype_key_raises(self) -> None:
        with pytest.raises(ValueError):

            class SubtypeUBrokenUniverseOnlyFG(FeatureGroup):
                @classmethod
                def subtype_universe(cls) -> frozenset[str]:
                    return frozenset({"rows_7"})

    def test_shape_b_with_both_overrides_is_legal(self) -> None:
        # SubtypeUFlattenedFG was defined at module scope without error; pin the derived gate.
        assert SubtypeUFlattenedFG.supported_subtypes(SubtypeUFwAlpha) == FLATTENED_UNIVERSE

    def test_parametric_families_without_subtype_dimension_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubtypeUBrokenFamiliesOnlyFG(FeatureGroup):
                PARAMETRIC_SUBTYPE_FAMILIES = {"ntile": "N-tile bucketing"}

        assert "SubtypeUBrokenFamiliesOnlyFG" in str(exc_info.value)

    def test_parametric_family_colliding_with_declared_value_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubtypeUBrokenCollidingFamilyFG(FeatureGroup):
                SUBTYPE_KEY = "sustatu_colliding_key"
                PARAMETRIC_SUBTYPE_FAMILIES = {"median": "Collides with a declared value"}
                PROPERTY_MAPPING = {
                    "sustatu_colliding_key": property_spec(
                        "Colliding subtype key.",
                        strict=True,
                        allowed_values={"median": "Median", "sum": "Sum"},
                    ),
                }

        message = str(exc_info.value)
        assert "SubtypeUBrokenCollidingFamilyFG" in message
        assert "median" in message


class TestContract11MatchTimeIntegration:
    """Contract 11: the declaration is enforced through IdentifyFeatureGroupClass."""

    def test_unsupported_subtype_is_routed_to_capable_framework(self) -> None:
        feature = Feature("value__median_sustatw")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubtypeUWindowFG: {SubtypeUFwAlpha, SubtypeUFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is SubtypeUWindowFG
        assert compute_frameworks == {SubtypeUFwAlpha}, (
            f"'median' is unsupported on SubtypeUFwBeta and must be routed around; got {compute_frameworks}"
        )

    def test_pin_to_incapable_framework_raises_capability_error(self) -> None:
        feature = Feature("value__median_sustatw")
        feature.compute_frameworks = {SubtypeUFwBeta}
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubtypeUWindowFG: {SubtypeUFwAlpha, SubtypeUFwBeta},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        message = str(exc_info.value)
        lowered = message.lower()
        assert "unsupported" in lowered or "not supported" in lowered, (
            f"Error must signal an unsupported framework, but got: {message}"
        )
        assert SubtypeUFwBeta.get_class_name() in message
        assert SubtypeUFwAlpha.get_class_name() in message
        assert "Did you mean" not in message

    def test_parametric_instance_is_routed_around(self) -> None:
        feature = Feature("value__ntile_2_sustatw")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubtypeUWindowFG: {SubtypeUFwAlpha, SubtypeUFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is SubtypeUWindowFG
        assert compute_frameworks == {SubtypeUFwAlpha}, (
            f"Family 'ntile' is unsupported on SubtypeUFwBeta, so 'ntile_2' must be routed around; "
            f"got {compute_frameworks}"
        )

    def test_unknown_subtype_keeps_every_framework(self) -> None:
        assert SubtypeUWindowFG.canonical_subtype("zzz") == "zzz"

        feature = Feature("value__zzz_sustatw")
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubtypeUWindowFG: {SubtypeUFwAlpha, SubtypeUFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        _, compute_frameworks = identified.get()
        assert compute_frameworks == {SubtypeUFwAlpha, SubtypeUFwBeta}, (
            f"An undeclared subtype must stay open on every framework; got {compute_frameworks}"
        )

    def test_family_without_subtype_dimension_is_unaffected(self) -> None:
        assert SubtypeUPlainFG.subtype_universe() == frozenset()

        feature = Feature(PLAIN_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SubtypeUPlainFG: {SubtypeUFwAlpha, SubtypeUFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        _, compute_frameworks = identified.get()
        assert compute_frameworks == {SubtypeUFwAlpha, SubtypeUFwBeta}

    def test_matching_stays_orthogonal_to_subtype_support(self) -> None:
        assert SubtypeUWindowFG.resolve_subtype("value__median_sustatw", Options()) == "median"
        assert "median" not in SubtypeUWindowFG.supported_subtypes(SubtypeUFwBeta)
        assert SubtypeUWindowFG.match_feature_group_criteria("value__median_sustatw", Options()) is True


R2_SUBTYPE_KEY = "sustatr2_op"
R2_UNIVERSE = frozenset({"lag_1", "sum", "lag"})
R2_BETA_SUPPORTED = frozenset({"lag_1", "sum"})


class SubtypeULagLiteralBaseFG(FeatureChainParserMixin, FeatureGroup):
    """Universe containing the literal declared value 'lag_1' alongside the parametric family 'lag'."""

    SUBTYPE_KEY = R2_SUBTYPE_KEY
    PARAMETRIC_SUBTYPE_FAMILIES = {"lag": "Lag by N rows"}
    PREFIX_PATTERN = r".*__([\w]+)_sustatr2$"
    PROPERTY_MAPPING = {
        R2_SUBTYPE_KEY: property_spec(
            "Operation subtype with a parametric-looking declared member.",
            strict=True,
            allowed_values={"lag_1": "Lag by one row", "sum": "Sum"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeUFwAlpha, SubtypeUFwBeta}


class SubtypeULagLiteralNarrowFG(SubtypeULagLiteralBaseFG):
    """Supports the literal 'lag_1' but not the 'lag' family on SubtypeUFwBeta."""

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is SubtypeUFwBeta:
            return R2_BETA_SUPPORTED
        return R2_UNIVERSE


class SubtypeUHookMatrixAbstractFG(SubtypeUExplicitOverrideFG):
    """Abstract hook-overriding family member; the matrix audit must stay empty."""

    @abstractmethod
    def _sustatu_hook_abstract_marker(self) -> None: ...


class SubtypeUHookMatrixPlainFG(FeatureGroup):
    """Hook override without a subtype dimension; the matrix audit must stay empty."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeUFwAlpha, SubtypeUFwBeta}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestR1ResolverOnlyDeclarationRaises:
    """R1: overriding only resolve_subtype without a subtype dimension is inert and must raise."""

    def test_resolver_only_override_without_universe_raises_at_definition(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubtypeUBrokenResolverOnlyFG(FeatureGroup):
                @classmethod
                def resolve_subtype(cls, feature_name: FeatureName | str, options: Options) -> Optional[str]:
                    return "rows_7"

        assert "SubtypeUBrokenResolverOnlyFG" in str(exc_info.value)


class TestR2DeclaredUniverseMemberNeverCollapses:
    """R2: canonical_subtype keeps a declared universe member even when it looks parametric."""

    def test_class_shape_is_legal_and_declared_member_stays_identity(self) -> None:
        assert SubtypeULagLiteralBaseFG.subtype_universe() == R2_UNIVERSE
        assert SubtypeULagLiteralBaseFG.canonical_subtype("lag_1") == "lag_1"
        assert SubtypeULagLiteralBaseFG.canonical_subtype("lag_7") == "lag"

    def test_derived_gate_distinguishes_declared_member_from_family_instance(self) -> None:
        gate = SubtypeULagLiteralNarrowFG.supports_compute_framework
        assert gate("value__lag_1_sustatr2", Options(), SubtypeUFwBeta) is True
        assert gate("value__lag_7_sustatr2", Options(), SubtypeUFwBeta) is False
        assert gate("value__lag_7_sustatr2", Options(), SubtypeUFwAlpha) is True


class TestR3HookOverrideMakesMatrixUnverifiable:
    """R3: a hand-written supports_compute_framework makes the declared matrix non-authoritative."""

    def test_matrix_raises_for_hook_overrider_and_spares_undimensioned_classes(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SubtypeUExplicitOverrideFG.subtype_support_matrix()
        message = str(exc_info.value)
        assert "supports_compute_framework" in message
        assert "SubtypeUExplicitOverrideFG" in message
        assert SubtypeUHookMatrixAbstractFG.subtype_support_matrix() == {}
        assert SubtypeUHookMatrixPlainFG.subtype_support_matrix() == {}
