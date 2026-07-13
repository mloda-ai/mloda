"""Pinning tests for the composed SubtypeDeclaration value-object contract (issue #639)."""

import dataclasses
from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, SubtypeDeclaration, property_spec


SUBDECL_KEY = "subdecl_window_function"
WINDOW_UNIVERSE = frozenset({"median", "sum", "lag", "ntile"})
BETA_SUPPORTED = frozenset({"sum", "lag"})
FLATTENED_LITERALS = frozenset({"rows_7", "range_7", "rows_all"})
FLATTENED_UNIVERSE = FLATTENED_LITERALS | {"roll"}
R2_KEY = "subdecl_op"
R2_UNIVERSE = frozenset({"lag_1", "sum", "lag"})
R2_BETA_SUPPORTED = frozenset({"lag_1", "sum"})
PLAIN_FEATURE = "subdecl_plain_feature"


def _subdecl_is_positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _subdecl_noop_resolver(feature_name: str, options: Options) -> Optional[str]:
    return None


def _subdecl_frame_resolver(feature_name: str, options: Options) -> Optional[str]:
    frame_type = options.get("subdecl_frame_type")
    frame_unit = options.get("subdecl_frame_unit")
    if frame_type is None or frame_unit is None:
        return None
    return f"{frame_type}_{frame_unit}"


def _subdecl_echo_resolver(feature_name: str, options: Options) -> Optional[str]:
    return feature_name


def _subdecl_pred_resolver(feature_name: str, options: Options) -> Optional[str]:
    value = options.get("subdecl_pred_key")
    if value is None:
        return None
    return f"p{value}"


class SubDeclFwAlpha(ComputeFramework):
    """First dummy compute framework for subtype-declaration tests."""


class SubDeclFwBeta(ComputeFramework):
    """Second dummy compute framework for subtype-declaration tests."""


class SubDeclWindowBaseFG(FeatureChainParserMixin, FeatureGroup):
    """Shape A: keyed declaration with a parametric family."""

    SUBTYPES = SubtypeDeclaration(
        key=SUBDECL_KEY,
        parametric_families={"ntile": "N-tile bucketing"},
    )
    PREFIX_PATTERN = r".*__([\w]+)_subdeclw$"
    PROPERTY_MAPPING = {
        SUBDECL_KEY: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum", "lag": "Lag"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclFwAlpha, SubDeclFwBeta}


class SubDeclWindowFG(SubDeclWindowBaseFG):
    """Narrows subtype support on SubDeclFwBeta through the declaration."""

    SUBTYPES = SubtypeDeclaration(
        key=SUBDECL_KEY,
        parametric_families={"ntile": "N-tile bucketing"},
        supported={SubDeclFwBeta.get_class_name(): BETA_SUPPORTED},
    )


class SubDeclAbstractWindowFG(SubDeclWindowBaseFG):
    """Abstract member of the window family; the matrix must stay empty."""

    @abstractmethod
    def _subdecl_backend_marker(self) -> None: ...


class SubDeclExplicitOverrideFG(SubDeclWindowFG):
    """Hand-written supports_compute_framework; wins for matching, voids the matrix."""

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return True


class SubDeclPlainFG(FeatureGroup):
    """Family without any subtype dimension."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclFwAlpha, SubDeclFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclFlattenedFG(FeatureGroup):
    """Shape B: flattened universe with a resolver and a parametric family."""

    SUBTYPES = SubtypeDeclaration(
        universe=FLATTENED_LITERALS,
        resolver=_subdecl_frame_resolver,
        parametric_families={"roll": "Rolling window"},
    )

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclFwAlpha, SubDeclFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclEchoFG(FeatureGroup):
    """Shape B whose resolver echoes the received name; pins stringification and pass-through."""

    SUBTYPES = SubtypeDeclaration(universe={"echo"}, resolver=_subdecl_echo_resolver)

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclPredicateUniverseFG(FeatureGroup):
    """Predicate-only PROPERTY_MAPPING key made legal via a shape B declaration."""

    PROPERTY_MAPPING = {
        "subdecl_pred_key": property_spec(
            "Predicate-validated subtype.",
            strict=True,
            element_validator=_subdecl_is_positive_int,
        ),
    }
    SUBTYPES = SubtypeDeclaration(universe={"p1", "p2"}, resolver=_subdecl_pred_resolver)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclFwAlpha, SubDeclFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclValueSpaceFG(FeatureGroup):
    """PROPERTY_MAPPING with mapping-form, list-form, numeric and predicate-only keys."""

    PROPERTY_MAPPING = {
        "subdecl_algo": property_spec("Algorithm.", strict=True, allowed_values={"a1": "A one", "a2": "A two"}),
        "subdecl_kind": property_spec("Kind.", strict=True, allowed_values=["x1", "x2"]),
        "subdecl_bucket": property_spec("Bucket count.", strict=True, allowed_values={2: "two", 7: "seven"}),
        "subdecl_mode": property_spec(
            "Mode.",
            strict=True,
            context=True,
            allowed_values={"fast": "Fast mode", "slow": "Slow mode"},
        ),
        "subdecl_n": property_spec("Predicate-only key.", strict=True, element_validator=_subdecl_is_positive_int),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclLagLiteralBaseFG(FeatureChainParserMixin, FeatureGroup):
    """Keyed universe containing the literal 'lag_1' alongside the parametric family 'lag'."""

    SUBTYPES = SubtypeDeclaration(key=R2_KEY, parametric_families={"lag": "Lag by N rows"})
    PREFIX_PATTERN = r".*__([\w]+)_subdeclr2$"
    PROPERTY_MAPPING = {
        R2_KEY: property_spec(
            "Operation subtype with a parametric-looking declared member.",
            strict=True,
            allowed_values={"lag_1": "Lag by one row", "sum": "Sum"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclFwAlpha, SubDeclFwBeta}


class SubDeclLagLiteralNarrowFG(SubDeclLagLiteralBaseFG):
    """Supports the literal 'lag_1' but not the 'lag' family on SubDeclFwBeta."""

    SUBTYPES = SubtypeDeclaration(
        key=R2_KEY,
        parametric_families={"lag": "Lag by N rows"},
        supported={SubDeclFwBeta.get_class_name(): R2_BETA_SUPPORTED},
    )


class SubDeclHookMatrixAbstractFG(SubDeclExplicitOverrideFG):
    """Abstract hook-overriding family member; the matrix audit must stay empty."""

    @abstractmethod
    def _subdecl_hook_abstract_marker(self) -> None: ...


class SubDeclHookMatrixPlainFG(FeatureGroup):
    """Hook override without a subtype dimension; the matrix audit must stay empty."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubDeclFwAlpha, SubDeclFwBeta}

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


class TestSubtypeDeclarationValueObject:
    """SubtypeDeclaration is a keyword-only frozen dataclass with normalized fields."""

    def test_public_export_matches_module_path(self) -> None:
        from mloda.core.abstract_plugins.components.subtype_declaration import SubtypeDeclaration as ModuleDecl

        assert ModuleDecl is SubtypeDeclaration

    def test_fields_are_keyword_only(self) -> None:
        ctor: Any = SubtypeDeclaration
        with pytest.raises(TypeError):
            ctor("subdecl_positional_key")

    def test_universe_iterable_is_normalized_to_frozenset(self) -> None:
        decl = SubtypeDeclaration(universe=["rows_7", "rows_all", "rows_7"], resolver=_subdecl_noop_resolver)
        assert decl.universe == frozenset({"rows_7", "rows_all"})
        assert isinstance(decl.universe, frozenset)

    def test_supported_is_normalized_to_frozensets(self) -> None:
        decl = SubtypeDeclaration(key="subdecl_any_key", supported={"SubDeclFwBeta": ["sum", "lag", "sum"]})
        assert decl.supported == {"SubDeclFwBeta": frozenset({"sum", "lag"})}
        assert decl.supported is not None
        assert all(isinstance(values, frozenset) for values in decl.supported.values())

    def test_declaration_is_frozen(self) -> None:
        decl = SubtypeDeclaration(key="subdecl_any_key")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(decl, "key", "subdecl_other_key")


class TestSubtypeDeclarationShapeValidation:
    """__post_init__ enforces exactly two legal declaration shapes."""

    def test_key_together_with_universe_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration(key="subdecl_any_key", universe={"a"}, resolver=_subdecl_noop_resolver)

    def test_key_together_with_resolver_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration(key="subdecl_any_key", resolver=_subdecl_noop_resolver)

    def test_universe_without_resolver_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration(universe={"a"})

    def test_resolver_without_universe_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration(resolver=_subdecl_noop_resolver)

    def test_empty_declaration_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration()

    def test_families_only_declaration_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration(parametric_families={"ntile": "N-tile bucketing"})

    def test_supported_only_declaration_raises(self) -> None:
        with pytest.raises(ValueError):
            SubtypeDeclaration(supported={"SubDeclFwBeta": {"sum"}})

    def test_shape_b_family_colliding_with_universe_literal_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SubtypeDeclaration(
                universe={"ntile", "plain"},
                resolver=_subdecl_noop_resolver,
                parametric_families={"ntile": "Collides with a declared literal"},
            )
        assert "ntile" in str(exc_info.value)

    def test_shape_b_supported_outside_universe_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SubtypeDeclaration(
                universe={"a"},
                resolver=_subdecl_noop_resolver,
                supported={"SubDeclFwBeta": {"subdecl_bogus"}},
            )
        assert "subdecl_bogus" in str(exc_info.value)

    def test_shape_b_supported_family_name_is_inside_universe(self) -> None:
        decl = SubtypeDeclaration(
            universe={"a"},
            resolver=_subdecl_noop_resolver,
            parametric_families={"fam": "A family"},
            supported={"SubDeclFwBeta": {"fam"}},
        )
        assert decl.supported == {"SubDeclFwBeta": frozenset({"fam"})}


class TestSubtypesDefault:
    """SUBTYPES class attribute with default None; one declaration object per family."""

    def test_base_default_is_none(self) -> None:
        assert FeatureGroup.SUBTYPES is None

    def test_untouched_subclass_default_is_none(self) -> None:
        assert SubDeclPlainFG.SUBTYPES is None

    def test_declaring_subclass_carries_declaration(self) -> None:
        assert FeatureGroup.SUBTYPES is None
        decl = SubDeclWindowBaseFG.SUBTYPES
        assert decl is not None
        assert decl.key == SUBDECL_KEY
        assert decl.parametric_families == {"ntile": "N-tile bucketing"}


class TestDeclaredOptionValues:
    """declared_option_values(key) returns the declared value space of a key."""

    def test_property_spec_dict_allowed_values(self) -> None:
        assert SubDeclValueSpaceFG.declared_option_values("subdecl_algo") == frozenset({"a1", "a2"})

    def test_property_spec_list_allowed_values(self) -> None:
        assert SubDeclValueSpaceFG.declared_option_values("subdecl_kind") == frozenset({"x1", "x2"})

    def test_non_string_values_are_stringified(self) -> None:
        values = SubDeclValueSpaceFG.declared_option_values("subdecl_bucket")
        assert values == frozenset({"2", "7"})
        assert all(isinstance(value, str) for value in values)

    def test_spec_flags_are_not_part_of_the_value_space(self) -> None:
        # subdecl_mode carries context/strict_validation flags; only allowed_values is the value space.
        values = SubDeclValueSpaceFG.declared_option_values("subdecl_mode")
        assert values == frozenset({"fast", "slow"})
        assert not values & {str(DefaultOptionKeys.context), str(DefaultOptionKeys.strict_validation)}

    def test_predicate_only_key_has_empty_value_space(self) -> None:
        assert SubDeclValueSpaceFG.declared_option_values("subdecl_n") == frozenset()

    def test_absent_key_returns_empty(self) -> None:
        assert SubDeclValueSpaceFG.declared_option_values("subdecl_absent") == frozenset()

    def test_property_mapping_none_returns_empty(self) -> None:
        assert SubDeclPlainFG.declared_option_values("anything") == frozenset()

    def test_result_is_frozenset(self) -> None:
        assert isinstance(SubDeclValueSpaceFG.declared_option_values("subdecl_algo"), frozenset)
        assert isinstance(SubDeclPlainFG.declared_option_values("anything"), frozenset)


class TestSubtypeUniverse:
    """subtype_universe() derives from the declaration; empty without one."""

    def test_keyed_universe_unions_declared_values_and_families(self) -> None:
        assert SubDeclWindowBaseFG.subtype_universe() == WINDOW_UNIVERSE

    def test_shape_b_universe_unions_literals_and_families(self) -> None:
        assert SubDeclFlattenedFG.subtype_universe() == FLATTENED_UNIVERSE

    def test_universe_empty_without_declaration(self) -> None:
        assert SubDeclPlainFG.subtype_universe() == frozenset()
        assert FeatureGroup.subtype_universe() == frozenset()

    def test_universe_is_frozenset(self) -> None:
        assert isinstance(SubDeclWindowBaseFG.subtype_universe(), frozenset)
        assert isinstance(SubDeclFlattenedFG.subtype_universe(), frozenset)


class TestSupportedSubtypes:
    """supported_subtypes(cfw) defaults to the full universe; supported entries narrow per framework."""

    def test_default_is_full_universe(self) -> None:
        assert SubDeclWindowBaseFG.supported_subtypes(SubDeclFwAlpha) == WINDOW_UNIVERSE
        assert SubDeclWindowBaseFG.supported_subtypes(SubDeclFwBeta) == WINDOW_UNIVERSE

    def test_supported_entry_narrows_only_the_named_framework(self) -> None:
        assert SubDeclWindowFG.supported_subtypes(SubDeclFwBeta) < SubDeclWindowFG.subtype_universe()
        assert SubDeclWindowFG.supported_subtypes(SubDeclFwBeta) == BETA_SUPPORTED
        assert SubDeclWindowFG.supported_subtypes(SubDeclFwAlpha) == WINDOW_UNIVERSE

    def test_shape_b_default_is_full_universe(self) -> None:
        assert SubDeclFlattenedFG.supported_subtypes(SubDeclFwBeta) == FLATTENED_UNIVERSE

    def test_shape_b_predicate_escape_default(self) -> None:
        assert SubDeclPredicateUniverseFG.supported_subtypes(SubDeclFwAlpha) == frozenset({"p1", "p2"})

    def test_empty_without_declaration(self) -> None:
        assert FeatureGroup.supported_subtypes(SubDeclFwAlpha) == frozenset()
        assert SubDeclPlainFG.supported_subtypes(SubDeclFwBeta) == frozenset()


class TestResolveSubtype:
    """resolve_subtype resolves the raw subtype from the name, then options; never raises."""

    def test_resolves_from_feature_name(self) -> None:
        assert SubDeclWindowBaseFG.resolve_subtype("value__median_subdeclw", Options()) == "median"

    def test_accepts_feature_name_object(self) -> None:
        assert SubDeclWindowBaseFG.resolve_subtype(FeatureName("value__lag_subdeclw"), Options()) == "lag"

    def test_parametric_instance_resolves_raw(self) -> None:
        assert SubDeclWindowBaseFG.resolve_subtype("value__ntile_2_subdeclw", Options()) == "ntile_2"

    def test_bare_chained_name_does_not_resolve_from_name_and_never_raises(self) -> None:
        assert SubDeclWindowBaseFG.resolve_subtype("__sum_subdeclw", Options()) is None

    def test_bare_chained_name_falls_back_to_options(self) -> None:
        options = Options(group={SUBDECL_KEY: "median"})
        assert SubDeclWindowBaseFG.resolve_subtype("__sum_subdeclw", options) == "median"

    def test_resolves_from_options_when_name_does_not_parse(self) -> None:
        options = Options(group={SUBDECL_KEY: "median"})
        assert SubDeclWindowBaseFG.resolve_subtype("subdecl_unchained", options) == "median"

    def test_options_value_is_stringified(self) -> None:
        options = Options(group={SUBDECL_KEY: 7})
        assert SubDeclWindowBaseFG.resolve_subtype("subdecl_unchained", options) == "7"

    def test_name_parsing_takes_precedence_over_options(self) -> None:
        options = Options(group={SUBDECL_KEY: "median"})
        assert SubDeclWindowBaseFG.resolve_subtype("value__sum_subdeclw", options) == "sum"

    def test_none_when_neither_resolves(self) -> None:
        assert SubDeclWindowBaseFG.resolve_subtype("subdecl_unchained", Options()) is None

    def test_none_without_declaration(self) -> None:
        options = Options(group={SUBDECL_KEY: "median"})
        assert SubDeclPlainFG.resolve_subtype("value__sum_subdeclw", options) is None
        assert FeatureGroup.resolve_subtype("anything", Options()) is None

    def test_shape_b_resolver_result_is_returned(self) -> None:
        options = Options(group={"subdecl_frame_type": "rows", "subdecl_frame_unit": "7"})
        assert SubDeclFlattenedFG.resolve_subtype("subdecl_flattened_feature", options) == "rows_7"
        assert SubDeclFlattenedFG.resolve_subtype("subdecl_flattened_feature", Options()) is None

    def test_shape_b_resolver_receives_stringified_name(self) -> None:
        assert SubDeclEchoFG.resolve_subtype(FeatureName("subdecl_echo_name"), Options()) == "subdecl_echo_name"


class TestCanonicalSubtype:
    """canonical_subtype collapses <family>_<digits> to the family, else identity."""

    def test_parametric_instance_collapses_to_family(self) -> None:
        assert SubDeclWindowBaseFG.canonical_subtype("ntile_2") == "ntile"

    def test_stem_that_is_not_a_family_stays_identity(self) -> None:
        assert SubDeclWindowBaseFG.canonical_subtype("ntile_2_3") == "ntile_2_3"

    def test_plain_subtype_stays_identity(self) -> None:
        assert SubDeclWindowBaseFG.canonical_subtype("median") == "median"

    def test_declared_value_is_not_a_family(self) -> None:
        assert SubDeclWindowBaseFG.canonical_subtype("sum_2") == "sum_2"

    def test_non_digit_suffix_stays_identity(self) -> None:
        assert SubDeclWindowBaseFG.canonical_subtype("ntile_x") == "ntile_x"

    def test_without_families_everything_is_identity(self) -> None:
        assert SubDeclPlainFG.canonical_subtype("ntile_2") == "ntile_2"
        assert FeatureGroup.canonical_subtype("anything_3") == "anything_3"

    def test_declared_literal_never_collapses(self) -> None:
        assert SubDeclLagLiteralBaseFG.subtype_universe() == R2_UNIVERSE
        assert SubDeclLagLiteralBaseFG.canonical_subtype("lag_1") == "lag_1"
        assert SubDeclLagLiteralBaseFG.canonical_subtype("lag_7") == "lag"

    def test_shape_b_family_collapse_and_literal_identity(self) -> None:
        assert SubDeclFlattenedFG.canonical_subtype("roll_3") == "roll"
        assert SubDeclFlattenedFG.canonical_subtype("rows_7") == "rows_7"


class TestDerivedSupportsComputeFramework:
    """The default supports_compute_framework is derived from the declaration."""

    def test_base_keeps_returning_true(self) -> None:
        assert FeatureGroup.SUBTYPES is None
        assert FeatureGroup.subtype_universe() == frozenset()
        assert FeatureGroup.supports_compute_framework("anything", Options(), SubDeclFwAlpha) is True

    def test_empty_universe_is_open(self) -> None:
        assert SubDeclPlainFG.subtype_universe() == frozenset()
        assert SubDeclPlainFG.supports_compute_framework(PLAIN_FEATURE, Options(), SubDeclFwBeta) is True

    def test_unresolved_subtype_is_open(self) -> None:
        assert SubDeclWindowFG.resolve_subtype("subdecl_unchained", Options()) is None
        assert SubDeclWindowFG.supports_compute_framework("subdecl_unchained", Options(), SubDeclFwBeta) is True

    def test_unknown_subtype_is_open(self) -> None:
        assert SubDeclWindowFG.canonical_subtype("zzz") == "zzz"
        assert SubDeclWindowFG.supports_compute_framework("value__zzz_subdeclw", Options(), SubDeclFwBeta) is True

    def test_declared_subtype_gated_by_supported_subtypes(self) -> None:
        assert SubDeclWindowFG.supports_compute_framework("value__median_subdeclw", Options(), SubDeclFwBeta) is False
        assert SubDeclWindowFG.supports_compute_framework("value__median_subdeclw", Options(), SubDeclFwAlpha) is True
        assert SubDeclWindowFG.supports_compute_framework("value__sum_subdeclw", Options(), SubDeclFwBeta) is True

    def test_parametric_instance_gated_via_canonical_family(self) -> None:
        assert SubDeclWindowFG.supports_compute_framework("value__ntile_4_subdeclw", Options(), SubDeclFwBeta) is False
        assert SubDeclWindowFG.supports_compute_framework("value__ntile_4_subdeclw", Options(), SubDeclFwAlpha) is True

    def test_options_resolved_subtype_is_gated_too(self) -> None:
        options = Options(group={SUBDECL_KEY: "median"})
        assert SubDeclWindowFG.supports_compute_framework("subdecl_unchained", options, SubDeclFwBeta) is False

    def test_declared_literal_gates_independently_of_its_family(self) -> None:
        gate = SubDeclLagLiteralNarrowFG.supports_compute_framework
        assert gate("value__lag_1_subdeclr2", Options(), SubDeclFwBeta) is True
        assert gate("value__lag_7_subdeclr2", Options(), SubDeclFwBeta) is False
        assert gate("value__lag_7_subdeclr2", Options(), SubDeclFwAlpha) is True

    def test_explicit_override_wins(self) -> None:
        assert SubDeclWindowFG.supports_compute_framework("value__median_subdeclw", Options(), SubDeclFwBeta) is False
        assert SubDeclExplicitOverrideFG.supported_subtypes(SubDeclFwBeta) == BETA_SUPPORTED
        result = SubDeclExplicitOverrideFG.supports_compute_framework(
            "value__median_subdeclw", Options(), SubDeclFwBeta
        )
        assert result is True


class TestSubtypeSupportMatrix:
    """subtype_support_matrix() audit surface per compute framework."""

    def test_matrix_maps_every_declared_framework(self) -> None:
        matrix = SubDeclWindowFG.subtype_support_matrix()
        assert matrix == {
            SubDeclFwAlpha.get_class_name(): WINDOW_UNIVERSE,
            SubDeclFwBeta.get_class_name(): BETA_SUPPORTED,
        }

    def test_matrix_empty_for_abstract_class(self) -> None:
        assert SubDeclAbstractWindowFG.subtype_support_matrix() == {}

    def test_matrix_empty_for_empty_universe(self) -> None:
        assert SubDeclPlainFG.subtype_support_matrix() == {}

    def test_matrix_raises_for_hook_overrider_and_spares_undimensioned_classes(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SubDeclExplicitOverrideFG.subtype_support_matrix()
        message = str(exc_info.value)
        assert "supports_compute_framework" in message
        assert "SubDeclExplicitOverrideFG" in message
        assert SubDeclHookMatrixAbstractFG.subtype_support_matrix() == {}
        assert SubDeclHookMatrixPlainFG.subtype_support_matrix() == {}


class TestClassDefinitionValidation:
    """Shape A class-dependent checks are enforced at class definition time."""

    def test_legal_keyed_declaration_does_not_raise(self) -> None:
        # SubDeclWindowBaseFG was defined at module scope without error; pin its derived universe.
        assert SubDeclWindowBaseFG.subtype_universe() == WINDOW_UNIVERSE

    def test_legal_shape_b_declaration_does_not_raise(self) -> None:
        # SubDeclFlattenedFG was defined at module scope without error; pin the derived default.
        assert SubDeclFlattenedFG.supported_subtypes(SubDeclFwAlpha) == FLATTENED_UNIVERSE

    def test_key_absent_from_property_mapping_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubDeclBrokenAbsentKeyFG(FeatureGroup):
                SUBTYPES = SubtypeDeclaration(key="subdecl_missing_key")
                PROPERTY_MAPPING = {
                    "subdecl_other": property_spec("Other.", strict=True, allowed_values={"v1": "V one"}),
                }

        message = str(exc_info.value)
        assert "SubDeclBrokenAbsentKeyFG" in message
        assert "subdecl_missing_key" in message

    def test_key_without_property_mapping_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubDeclBrokenNoMappingFG(FeatureGroup):
                SUBTYPES = SubtypeDeclaration(key="subdecl_missing_key")

        message = str(exc_info.value)
        assert "SubDeclBrokenNoMappingFG" in message
        assert "subdecl_missing_key" in message

    def test_predicate_only_key_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubDeclBrokenPredicateOnlyFG(FeatureGroup):
                SUBTYPES = SubtypeDeclaration(key="subdecl_pred_only_key")
                PROPERTY_MAPPING = {
                    "subdecl_pred_only_key": property_spec(
                        "Predicate-only subtype key.",
                        strict=True,
                        element_validator=_subdecl_is_positive_int,
                    ),
                }

        message = str(exc_info.value)
        assert "SubDeclBrokenPredicateOnlyFG" in message
        assert "subdecl_pred_only_key" in message

    def test_predicate_only_key_can_declare_shape_b(self) -> None:
        # SubDeclPredicateUniverseFG was defined at module scope without error; shape B is the escape.
        assert SubDeclPredicateUniverseFG.declared_option_values("subdecl_pred_key") == frozenset()
        assert SubDeclPredicateUniverseFG.subtype_universe() == frozenset({"p1", "p2"})

    def test_family_colliding_with_declared_value_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubDeclBrokenCollidingFamilyFG(FeatureGroup):
                SUBTYPES = SubtypeDeclaration(
                    key="subdecl_colliding_key",
                    parametric_families={"median": "Collides with a declared value"},
                )
                PROPERTY_MAPPING = {
                    "subdecl_colliding_key": property_spec(
                        "Colliding subtype key.",
                        strict=True,
                        allowed_values={"median": "Median", "sum": "Sum"},
                    ),
                }

        message = str(exc_info.value)
        assert "SubDeclBrokenCollidingFamilyFG" in message
        assert "median" in message

    def test_supported_outside_universe_raises_at_class_definition(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class SubDeclBrokenOverreachFG(FeatureGroup):
                SUBTYPES = SubtypeDeclaration(
                    key="subdecl_overreach_key",
                    supported={"SubDeclFwBeta": {"subdecl_bogus"}},
                )
                PROPERTY_MAPPING = {
                    "subdecl_overreach_key": property_spec(
                        "Overreaching subtype key.",
                        strict=True,
                        allowed_values={"v1": "V one"},
                    ),
                }

        message = str(exc_info.value)
        assert "SubDeclBrokenOverreachFG" in message
        assert "subdecl_bogus" in message


def _subdecl_setitem(mapping: Any, key: str, value: Any) -> None:
    mapping[key] = value


def _subdecl_raising_resolver(feature_name: str, options: Options) -> Optional[str]:
    raise RuntimeError("subdecl resolver boom")


def _subdecl_int_resolver(feature_name: str, options: Options) -> Any:
    return 7


class SubDeclRaisingResolverFG(FeatureGroup):
    """Shape B whose resolver raises; resolution must degrade instead of propagating."""

    SUBTYPES = SubtypeDeclaration(
        universe={"boom"},
        resolver=_subdecl_raising_resolver,
        supported={SubDeclFwBeta.get_class_name(): frozenset()},
    )

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SubDeclIntResolverFG(FeatureGroup):
    """Shape B whose resolver returns a raw int; the result must be stringified."""

    SUBTYPES = SubtypeDeclaration(
        universe={"7"},
        resolver=_subdecl_int_resolver,
        supported={SubDeclFwBeta.get_class_name(): frozenset()},
    )

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestSubDeclDeepImmutability:
    """supported and parametric_families are read-only mappings; universe stays a frozenset."""

    def test_shape_a_mappings_reject_item_assignment(self) -> None:
        decl = SubtypeDeclaration(
            key="subdecl_immut_key",
            parametric_families={"fam": "A family"},
            supported={"SubDeclFwBeta": {"fam"}},
        )
        with pytest.raises(TypeError):
            _subdecl_setitem(decl.supported, "SubDeclFwAlpha", frozenset({"fam"}))
        with pytest.raises(TypeError):
            _subdecl_setitem(decl.parametric_families, "fam2", "Another family")

    def test_shape_b_mappings_reject_item_assignment_and_universe_is_frozenset(self) -> None:
        decl = SubtypeDeclaration(
            universe={"lit"},
            resolver=_subdecl_noop_resolver,
            parametric_families={"fam": "A family"},
            supported={"SubDeclFwBeta": {"lit", "fam"}},
        )
        with pytest.raises(TypeError):
            _subdecl_setitem(decl.supported, "SubDeclFwAlpha", frozenset({"lit"}))
        with pytest.raises(TypeError):
            _subdecl_setitem(decl.parametric_families, "fam2", "Another family")
        assert isinstance(decl.universe, frozenset)


class TestSubDeclStringNormalizationOnIngest:
    """Every declared value is stringified when the declaration is constructed."""

    def test_numeric_universe_members_are_stringified(self) -> None:
        raw_universe: set[Any] = {2, 7}
        decl = SubtypeDeclaration(universe=raw_universe, resolver=_subdecl_noop_resolver)
        assert decl.universe == frozenset({"2", "7"})

    def test_numeric_supported_values_are_stringified(self) -> None:
        raw_universe: set[Any] = {2, 7}
        raw_supported: dict[str, set[Any]] = {"SubDeclSomeFw": {2}}
        decl = SubtypeDeclaration(universe=raw_universe, resolver=_subdecl_noop_resolver, supported=raw_supported)
        assert decl.supported == {"SubDeclSomeFw": frozenset({"2"})}

    def test_numeric_parametric_family_keys_are_stringified(self) -> None:
        raw_families: dict[Any, str] = {5: "Lag by N rows"}
        decl = SubtypeDeclaration(universe={"lit"}, resolver=_subdecl_noop_resolver, parametric_families=raw_families)
        assert decl.family_names() == frozenset({"5"})


class TestSubDeclKeyedNumericValueSpace:
    """A keyed declaration over a numeric allowed_values space is definable and gates."""

    def test_numeric_value_space_family_is_definable_and_gates(self) -> None:
        numeric_supported: dict[str, set[Any]] = {
            SubDeclFwAlpha.get_class_name(): {2},
            SubDeclFwBeta.get_class_name(): {7},
        }

        class SubDeclNumericBucketFG(FeatureGroup):
            SUBTYPES = SubtypeDeclaration(key="subdecl_num_bucket", supported=numeric_supported)
            PROPERTY_MAPPING = {
                "subdecl_num_bucket": property_spec(
                    "Bucket count.",
                    strict=True,
                    allowed_values={2: "two", 7: "seven"},
                ),
            }

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
                return {SubDeclFwAlpha, SubDeclFwBeta}

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
                return None

        options = Options(group={"subdecl_num_bucket": 2})
        gate = SubDeclNumericBucketFG.supports_compute_framework
        assert gate("subdecl_num_feature", options, SubDeclFwAlpha) is True
        assert gate("subdecl_num_feature", options, SubDeclFwBeta) is False


class TestSubDeclEmptyUniverseRejection:
    """Shape B must declare a non-empty universe."""

    def test_empty_universe_raises(self) -> None:
        empty_universe: set[str] = set()
        with pytest.raises(ValueError):
            SubtypeDeclaration(universe=empty_universe, resolver=_subdecl_noop_resolver)


class TestSubDeclRaisingResolverDegrades:
    """A raising resolver degrades to unresolved instead of propagating."""

    def test_resolve_subtype_returns_none(self) -> None:
        assert SubDeclRaisingResolverFG.resolve_subtype("subdecl_raising_feature", Options()) is None

    def test_supports_compute_framework_stays_open(self) -> None:
        gate = SubDeclRaisingResolverFG.supports_compute_framework
        assert gate("subdecl_raising_feature", Options(), SubDeclFwBeta) is True


class TestSubDeclResolverResultStringified:
    """A non-string resolver result is stringified before gating."""

    def test_resolve_subtype_stringifies_int_result(self) -> None:
        assert SubDeclIntResolverFG.resolve_subtype("subdecl_int_feature", Options()) == "7"

    def test_stringified_result_gates_by_supported_subtypes(self) -> None:
        gate = SubDeclIntResolverFG.supports_compute_framework
        assert gate("subdecl_int_feature", Options(), SubDeclFwBeta) is False
        assert gate("subdecl_int_feature", Options(), SubDeclFwAlpha) is True
