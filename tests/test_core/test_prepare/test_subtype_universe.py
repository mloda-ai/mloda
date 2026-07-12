"""Tests for the declarative subtype universe on FeatureGroup (issue #639).

Contract under test (to be implemented by the Green agent):

1. ``FeatureGroup.SUBTYPE_KEY: ClassVar[str | None] = None`` names the
   PROPERTY_MAPPING key carrying the family's subtype discriminator.
2. ``declared_option_values(key)`` returns a key's declared value space
   (``allowed_values`` form and legacy flattened form).
3. ``subtype_universe()`` returns the subtypes the family defines.
4. ``supported_subtypes(compute_framework)`` returns the subtypes supported on
   one framework, defaulting to the full universe.
5. ``resolve_subtype(feature_name, options)`` returns a concrete feature's subtype.
6. ``supports_compute_framework`` default is DERIVED from 3/4/5.
7. ``subtype_support_matrix()`` (``@final``) enumerates framework -> subtypes and
   raises on a capability declared outside the universe.
8. ``__init_subclass__`` rejects a SUBTYPE_KEY absent from PROPERTY_MAPPING unless
   the class overrides ``subtype_universe()``.

All tests fail until the feature exists.
"""

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


class SubtypeFwAlpha(ComputeFramework):
    """First compute framework used in subtype tests."""


class SubtypeFwBeta(ComputeFramework):
    """Second compute framework used in subtype tests."""


class NoSubtypeFeatureGroup(FeatureGroup):
    """Family without a subtype dimension: SUBTYPE_KEY stays None."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class LegacyValuesFeatureGroup(FeatureGroup):
    """Legacy flattened PROPERTY_MAPPING: value space is the non-metadata keys."""

    PROPERTY_MAPPING = {
        "stat_type": {
            "sum": "Sum of values",
            "median": "Median value",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "meta_only": {
            "explanation": "A key that declares no value space",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }


class AllowedValuesFeatureGroup(FeatureGroup):
    """property_spec form: value space comes from ``allowed_values``."""

    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value"},
        ),
        "frame_unit": property_spec(
            "frame unit",
            strict=True,
            allowed_values=["rows", "range"],
        ),
    }


class SplitStatFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Family with a subtype dimension and a per-framework capability subset.

    SubtypeFwBeta supports only "sum"; SubtypeFwAlpha supports the full universe.
    """

    SUBTYPE_KEY = "stat_type"
    PREFIX_PATTERN = r".*__([\w]+)_splitstat$"

    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value", "mode": "Most frequent value"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeFwAlpha, SubtypeFwBeta}

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is SubtypeFwBeta:
            return frozenset({"sum"})
        return cls.subtype_universe()


class FullUniverseStatFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Same shape as SplitStatFeatureGroup but without a supported_subtypes override."""

    SUBTYPE_KEY = "stat_type"
    PREFIX_PATTERN = r".*__([\w]+)_fullstat$"

    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value", "mode": "Most frequent value"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeFwAlpha, SubtypeFwBeta}


class ExplicitOverrideStatFeatureGroup(SplitStatFeatureGroup):
    """An explicit supports_compute_framework override beats the derived default."""

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return True


class MisdeclaredCapabilityFeatureGroup(SplitStatFeatureGroup):
    """Declares a capability outside the universe: the matrix must reject it."""

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        return frozenset({"sum", "bogus_stat"})


class FrameSpecFeatureGroup(FeatureGroup):
    """Two-axis family: flattens frame_type x frame_unit into compound subtypes.

    SUBTYPE_KEY names no PROPERTY_MAPPING key, which is allowed because
    subtype_universe() is overridden.
    """

    SUBTYPE_KEY = "frame_spec"

    PROPERTY_MAPPING = {
        "frame_type": property_spec("frame type", strict=True, allowed_values=["rows", "range"]),
        "frame_unit": property_spec("frame unit", strict=True, allowed_values=["1", "7"]),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeFwAlpha}

    @classmethod
    def subtype_universe(cls) -> frozenset[str]:
        return frozenset({"rows_1", "rows_7", "range_1", "range_7"})

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestDeclaredOptionValues:
    """Contract 2: declared_option_values(key) -> frozenset[str]."""

    def test_returns_empty_frozenset_when_property_mapping_is_none(self) -> None:
        assert NoSubtypeFeatureGroup.declared_option_values("stat_type") == frozenset()

    def test_returns_empty_frozenset_when_key_is_absent(self) -> None:
        assert AllowedValuesFeatureGroup.declared_option_values("not_declared") == frozenset()

    def test_legacy_flattened_form_subtracts_reserved_metadata_keys(self) -> None:
        result = LegacyValuesFeatureGroup.declared_option_values("stat_type")

        assert result == frozenset({"sum", "median"})
        assert str(DefaultOptionKeys.context) not in result
        assert str(DefaultOptionKeys.strict_validation) not in result

    def test_legacy_spec_without_value_space_returns_empty_frozenset(self) -> None:
        assert LegacyValuesFeatureGroup.declared_option_values("meta_only") == frozenset()

    def test_allowed_values_mapping_form(self) -> None:
        assert AllowedValuesFeatureGroup.declared_option_values("stat_type") == frozenset({"sum", "median"})

    def test_allowed_values_iterable_form(self) -> None:
        assert AllowedValuesFeatureGroup.declared_option_values("frame_unit") == frozenset({"rows", "range"})

    def test_returns_frozenset_of_strings(self) -> None:
        result = AllowedValuesFeatureGroup.declared_option_values("stat_type")

        assert isinstance(result, frozenset)
        assert all(isinstance(value, str) for value in result)


class TestSubtypeKeyAndUniverse:
    """Contracts 1 and 3: SUBTYPE_KEY default and subtype_universe()."""

    def test_subtype_key_defaults_to_none(self) -> None:
        assert FeatureGroup.SUBTYPE_KEY is None
        assert NoSubtypeFeatureGroup.SUBTYPE_KEY is None

    def test_universe_is_empty_without_subtype_key(self) -> None:
        assert NoSubtypeFeatureGroup.subtype_universe() == frozenset()

    def test_universe_derives_from_declared_option_values(self) -> None:
        assert SplitStatFeatureGroup.SUBTYPE_KEY == "stat_type"
        assert SplitStatFeatureGroup.subtype_universe() == frozenset({"sum", "median", "mode"})

    def test_universe_override_wins_for_multi_axis_family(self) -> None:
        assert FrameSpecFeatureGroup.subtype_universe() == frozenset({"rows_1", "rows_7", "range_1", "range_7"})


class TestSupportedSubtypes:
    """Contract 4: supported_subtypes(compute_framework)."""

    def test_default_is_the_full_universe(self) -> None:
        universe = FullUniverseStatFeatureGroup.subtype_universe()

        assert FullUniverseStatFeatureGroup.supported_subtypes(SubtypeFwAlpha) == universe
        assert FullUniverseStatFeatureGroup.supported_subtypes(SubtypeFwBeta) == universe

    def test_override_returns_a_subset_per_framework(self) -> None:
        assert SplitStatFeatureGroup.supported_subtypes(SubtypeFwBeta) == frozenset({"sum"})
        assert SplitStatFeatureGroup.supported_subtypes(SubtypeFwAlpha) == frozenset({"sum", "median", "mode"})

    def test_default_is_empty_without_subtype_dimension(self) -> None:
        assert NoSubtypeFeatureGroup.supported_subtypes(SubtypeFwAlpha) == frozenset()


class TestResolveSubtype:
    """Contract 5: resolve_subtype(feature_name, options)."""

    def test_resolves_from_the_feature_name_pattern(self) -> None:
        assert SplitStatFeatureGroup.resolve_subtype("revenue__median_splitstat", Options()) == "median"

    def test_accepts_a_feature_name_object(self) -> None:
        resolved = SplitStatFeatureGroup.resolve_subtype(FeatureName("revenue__median_splitstat"), Options())

        assert resolved == "median"

    def test_resolves_from_options_when_the_name_does_not_parse(self) -> None:
        options = Options(context={"stat_type": "mode"})

        assert SplitStatFeatureGroup.resolve_subtype("placeholder", options) == "mode"

    def test_options_value_is_stringified(self) -> None:
        options = Options(context={"stat_type": 7})

        assert SplitStatFeatureGroup.resolve_subtype("placeholder", options) == "7"

    def test_name_parsing_takes_precedence_over_options(self) -> None:
        options = Options(context={"stat_type": "sum"})

        assert SplitStatFeatureGroup.resolve_subtype("revenue__median_splitstat", options) == "median"

    def test_returns_none_when_neither_path_resolves(self) -> None:
        assert SplitStatFeatureGroup.resolve_subtype("plain_feature", Options()) is None

    def test_returns_none_without_subtype_key(self) -> None:
        options = Options(context={"stat_type": "sum"})

        assert NoSubtypeFeatureGroup.resolve_subtype("plain_feature", options) is None


class TestDerivedSupportsComputeFramework:
    """Contract 6: supports_compute_framework derived from the universe."""

    def test_true_without_subtype_key(self) -> None:
        options = Options(context={"stat_type": "median"})

        assert NoSubtypeFeatureGroup.supports_compute_framework("anything", options, SubtypeFwBeta) is True

    def test_true_when_the_subtype_does_not_resolve(self) -> None:
        assert SplitStatFeatureGroup.supports_compute_framework("plain_feature", Options(), SubtypeFwBeta) is True

    def test_true_for_a_parametric_subtype_outside_the_universe(self) -> None:
        """Parametric subtypes (ntile_2, top_3) must stay open: this hook must not double-gate."""
        resolved = SplitStatFeatureGroup.resolve_subtype("revenue__ntile_2_splitstat", Options())
        assert resolved == "ntile_2"
        assert resolved not in SplitStatFeatureGroup.subtype_universe()

        assert (
            SplitStatFeatureGroup.supports_compute_framework("revenue__ntile_2_splitstat", Options(), SubtypeFwBeta)
            is True
        )

    def test_true_when_the_subtype_is_supported_on_the_framework(self) -> None:
        assert (
            SplitStatFeatureGroup.supports_compute_framework("revenue__sum_splitstat", Options(), SubtypeFwBeta) is True
        )
        assert (
            SplitStatFeatureGroup.supports_compute_framework("revenue__median_splitstat", Options(), SubtypeFwAlpha)
            is True
        )

    def test_false_when_the_subtype_is_unsupported_on_the_framework(self) -> None:
        assert (
            SplitStatFeatureGroup.supports_compute_framework("revenue__median_splitstat", Options(), SubtypeFwBeta)
            is False
        )

    def test_false_for_the_config_based_path(self) -> None:
        options = Options(context={"stat_type": "median"})

        assert SplitStatFeatureGroup.supports_compute_framework("placeholder", options, SubtypeFwBeta) is False

    def test_explicit_override_beats_the_derived_default(self) -> None:
        assert ExplicitOverrideStatFeatureGroup.supported_subtypes(SubtypeFwBeta) == frozenset({"sum"})
        assert (
            ExplicitOverrideStatFeatureGroup.supports_compute_framework(
                "revenue__median_splitstat", Options(), SubtypeFwBeta
            )
            is True
        )

    def test_base_feature_group_default_stays_true(self) -> None:
        assert FeatureGroup.supports_compute_framework("anything", Options(), SubtypeFwAlpha) is True


class TestSubtypeSupportMatrix:
    """Contract 7: subtype_support_matrix() enumerates support without probing."""

    def test_maps_framework_class_name_to_supported_subtypes(self) -> None:
        matrix = SplitStatFeatureGroup.subtype_support_matrix()

        assert matrix == {
            "SubtypeFwAlpha": frozenset({"sum", "median", "mode"}),
            "SubtypeFwBeta": frozenset({"sum"}),
        }

    def test_default_matrix_gives_every_framework_the_full_universe(self) -> None:
        matrix = FullUniverseStatFeatureGroup.subtype_support_matrix()

        assert matrix == {
            "SubtypeFwAlpha": frozenset({"sum", "median", "mode"}),
            "SubtypeFwBeta": frozenset({"sum", "median", "mode"}),
        }

    def test_empty_dict_without_subtype_dimension(self) -> None:
        assert NoSubtypeFeatureGroup.subtype_support_matrix() == {}

    def test_raises_when_a_capability_lies_outside_the_universe(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            MisdeclaredCapabilityFeatureGroup.subtype_support_matrix()

        message = str(exc_info.value)
        assert "bogus_stat" in message, f"Error must name the offending subtype, but got: {message}"

    def test_multi_axis_family_matrix_uses_the_overridden_universe(self) -> None:
        matrix = FrameSpecFeatureGroup.subtype_support_matrix()

        assert matrix == {"SubtypeFwAlpha": frozenset({"rows_1", "rows_7", "range_1", "range_7"})}


class TestSubtypeKeyClassDefinitionValidation:
    """Contract 8: a SUBTYPE_KEY absent from PROPERTY_MAPPING is a class-definition error."""

    def test_raises_when_subtype_key_is_not_a_property_mapping_key(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class BrokenSubtypeFeatureGroup(FeatureGroup):
                SUBTYPE_KEY = "missing_key"

                PROPERTY_MAPPING = {
                    "stat_type": property_spec("statistic", strict=True, allowed_values=["sum"]),
                }

        message = str(exc_info.value)
        assert "BrokenSubtypeFeatureGroup" in message, f"Error must name the class, but got: {message}"
        assert "missing_key" in message, f"Error must name the bad key, but got: {message}"

    def test_raises_when_property_mapping_is_none(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class NoMappingSubtypeFeatureGroup(FeatureGroup):
                SUBTYPE_KEY = "stat_type"

        message = str(exc_info.value)
        assert "NoMappingSubtypeFeatureGroup" in message, f"Error must name the class, but got: {message}"
        assert "stat_type" in message, f"Error must name the bad key, but got: {message}"

    def test_no_raise_when_subtype_universe_is_overridden(self) -> None:
        class OverriddenUniverseFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = "compound_key"

            @classmethod
            def subtype_universe(cls) -> frozenset[str]:
                return frozenset({"a_1", "b_2"})

        assert OverriddenUniverseFeatureGroup.subtype_universe() == frozenset({"a_1", "b_2"})

    def test_no_raise_for_a_declared_subtype_key(self) -> None:
        class ValidSubtypeFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = "stat_type"

            PROPERTY_MAPPING = {
                "stat_type": property_spec("statistic", strict=True, allowed_values=["sum", "median"]),
            }

        assert ValidSubtypeFeatureGroup.subtype_universe() == frozenset({"sum", "median"})


class TestDerivedCapabilityAtMatchTime:
    """The derived hook drives resolution: route-around and the dedicated pin error."""

    def test_route_around_the_unsupported_framework(self) -> None:
        feature = Feature("revenue__median_splitstat")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SplitStatFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is SplitStatFeatureGroup
        assert compute_frameworks == {SubtypeFwAlpha}, (
            f"'median' is unsupported on SubtypeFwBeta and must be narrowed out; got {compute_frameworks}"
        )

    def test_supported_subtype_keeps_every_framework(self) -> None:
        feature = Feature("revenue__sum_splitstat")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SplitStatFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        _, compute_frameworks = identified.get()
        assert compute_frameworks == {SubtypeFwAlpha, SubtypeFwBeta}

    def test_pin_to_unsupported_framework_raises_the_capability_error(self) -> None:
        feature = Feature("revenue__median_splitstat")
        feature.compute_frameworks = {SubtypeFwBeta}

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SplitStatFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
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
            f"Capability error must signal an unsupported framework, but got: {message}"
        )
        assert SubtypeFwBeta.get_class_name() in message, f"Error must name the rejected framework, but got: {message}"
        assert SubtypeFwAlpha.get_class_name() in message, (
            f"Error must name the supported framework, but got: {message}"
        )
        assert "Did you mean" not in message, (
            f"Capability error must skip the fuzzy suggestion path, but got: {message}"
        )

    def test_parametric_subtype_is_not_gated_at_match_time(self) -> None:
        """An unknown/parametric subtype stays open on every framework."""
        feature = Feature("revenue__ntile_2_splitstat")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SplitStatFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        _, compute_frameworks = identified.get()
        assert compute_frameworks == {SubtypeFwAlpha, SubtypeFwBeta}

    def test_feature_group_without_subtype_dimension_is_unaffected(self) -> None:
        feature = Feature(NoSubtypeFeatureGroup.get_class_name())

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            NoSubtypeFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is NoSubtypeFeatureGroup
        assert compute_frameworks == {SubtypeFwAlpha, SubtypeFwBeta}


def test_match_feature_group_criteria_still_accepts_the_unsupported_subtype() -> None:
    """Capability is orthogonal to matching: the feature group still matches the name."""
    matched = SplitStatFeatureGroup.match_feature_group_criteria(
        FeatureName("revenue__median_splitstat"), Options(), DataAccessCollection()
    )

    assert matched is True
