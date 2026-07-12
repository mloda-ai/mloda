"""Tests for the declarative subtype universe on FeatureGroup (issue #639).

Contract under test:

1. ``FeatureGroup.SUBTYPE_KEY: ClassVar[str | None] = None`` names the
   PROPERTY_MAPPING key carrying the family's subtype discriminator.
2. ``declared_option_values(key)`` returns a key's declared value space
   (``allowed_values`` form and legacy flattened form).
3. ``subtype_universe()`` returns the subtypes the family defines.
4. ``supported_subtypes(compute_framework)`` returns the subtypes supported on
   one framework, defaulting to the full universe.
5. ``resolve_subtype(feature_name, options)`` returns a concrete feature's subtype
   and NEVER raises.
6. ``supports_compute_framework`` default is DERIVED from 3/4/5, for both
   declaration shapes.
7. ``subtype_support_matrix()`` (``@final``) enumerates framework -> subtypes,
   returns ``{}`` for an ABSTRACT class (an abstract base declares the universe,
   not the support), and raises on a capability declared outside the universe.
8. Class-definition validation. Exactly two declaration shapes are legal:

   Shape A: ``SUBTYPE_KEY`` names a declared PROPERTY_MAPPING key whose declared
   value space is non-empty (or the class overrides ``subtype_universe()``).

   Shape B: ``SUBTYPE_KEY`` stays ``None`` and the class overrides BOTH
   ``subtype_universe()`` and ``resolve_subtype()`` (flattened multi-axis families).

   Anything else is a ValueError at class-definition time.
"""

import inspect
from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DefaultOptionKeys, FeatureChainParser, FeatureChainParserMixin, FeatureGroup, property_spec


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
    """Shape A family with a per-framework capability subset.

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


class AbstractStatFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Abstract base: declares the subtype universe, but no per-framework support.

    ``compute_framework_rule()`` is None (all frameworks), which is exactly the shape
    that must NOT be turned into a fabricated "every framework supports everything"
    capability claim.
    """

    SUBTYPE_KEY = "stat_type"
    PREFIX_PATTERN = r".*__([\w]+)_abstractstat$"

    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value"},
        ),
    }

    @classmethod
    @abstractmethod
    def _perform_stat(cls, data: Any, stat_type: str) -> Any:
        """Framework-specific implementation, supplied by the concrete subclass."""


class OpenConcreteStatFeatureGroup(AbstractStatFeatureGroup):
    """Concrete subclass that keeps ``compute_framework_rule() -> None``.

    Rule None genuinely means "all frameworks", so this class DOES report every
    declared framework. Abstractness, not the rule, is what suppresses the matrix.
    """

    @classmethod
    def _perform_stat(cls, data: Any, stat_type: str) -> Any:
        return None


class FrameSpecFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Shape B family: flattens frame_type x frame_unit into one compound subtype.

    SUBTYPE_KEY stays None; the class overrides BOTH subtype_universe() and
    resolve_subtype(), so the flattened subtype is enforceable.
    SubtypeFwBeta has no range frames.
    """

    SUBTYPE_KEY = None
    PREFIX_PATTERN = r".*__([\w]+)_framespec$"

    PROPERTY_MAPPING = {
        "frame_type": property_spec("frame type", strict=True, allowed_values=["rows", "range"]),
        "frame_unit": property_spec("frame unit", strict=True, allowed_values=["1", "7"]),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SubtypeFwAlpha, SubtypeFwBeta}

    @classmethod
    def subtype_universe(cls) -> frozenset[str]:
        return frozenset({"rows_1", "rows_7", "range_1", "range_7"})

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is SubtypeFwBeta:
            return frozenset({"rows_1", "rows_7"})
        return cls.subtype_universe()

    @classmethod
    def resolve_subtype(cls, feature_name: FeatureName | str, options: Options) -> Optional[str]:
        parsed, _ = FeatureChainParser.parse_feature_name(str(feature_name), [cls.PREFIX_PATTERN])
        if parsed is not None:
            return parsed

        frame_type = options.get("frame_type")
        frame_unit = options.get("frame_unit")
        if frame_type is None or frame_unit is None:
            return None
        return f"{frame_type}_{frame_unit}"


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

    def test_universe_override_wins_for_the_flattened_family(self) -> None:
        assert FrameSpecFeatureGroup.SUBTYPE_KEY is None
        assert FrameSpecFeatureGroup.subtype_universe() == frozenset({"rows_1", "rows_7", "range_1", "range_7"})


class TestSupportedSubtypes:
    """Contract 4: supported_subtypes(compute_framework)."""

    def test_default_is_the_full_universe(self) -> None:
        assert FullUniverseStatFeatureGroup.supported_subtypes(SubtypeFwAlpha) == frozenset({"sum", "median", "mode"})
        assert FullUniverseStatFeatureGroup.supported_subtypes(SubtypeFwBeta) == frozenset({"sum", "median", "mode"})

    def test_override_returns_a_subset_per_framework(self) -> None:
        assert SplitStatFeatureGroup.supported_subtypes(SubtypeFwBeta) == frozenset({"sum"})
        assert SplitStatFeatureGroup.supported_subtypes(SubtypeFwAlpha) == frozenset({"sum", "median", "mode"})

    def test_default_is_empty_without_subtype_dimension(self) -> None:
        assert NoSubtypeFeatureGroup.supported_subtypes(SubtypeFwAlpha) == frozenset()


class TestResolveSubtype:
    """Contract 5: resolve_subtype(feature_name, options) resolves, and never raises."""

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

    def test_never_raises_on_a_pattern_match_without_a_source_feature(self) -> None:
        """A name matching PREFIX_PATTERN with an empty source segment must not blow up."""
        assert SplitStatFeatureGroup.resolve_subtype("__median_splitstat", Options()) is None

    def test_capability_hook_stays_a_bool_on_a_source_less_name(self) -> None:
        assert (
            SplitStatFeatureGroup.supports_compute_framework("__median_splitstat", Options(), SubtypeFwBeta) is True
        ), "A match-time capability hook must return a bool, never raise"


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


class TestFlattenedSubtypeIsEnforced:
    """Contract 6, Shape B: the flattened subtype gates capability, it is not decorative."""

    def test_false_for_an_unsupported_flattened_subtype_on_the_name_path(self) -> None:
        assert (
            FrameSpecFeatureGroup.supports_compute_framework("revenue__range_7_framespec", Options(), SubtypeFwBeta)
            is False
        ), "SubtypeFwBeta declares no range frames, so 'range_7' must be rejected"

    def test_true_for_a_supported_flattened_subtype_on_the_name_path(self) -> None:
        assert (
            FrameSpecFeatureGroup.supports_compute_framework("revenue__rows_1_framespec", Options(), SubtypeFwBeta)
            is True
        )
        assert (
            FrameSpecFeatureGroup.supports_compute_framework("revenue__range_7_framespec", Options(), SubtypeFwAlpha)
            is True
        )

    def test_false_for_an_unsupported_flattened_subtype_on_the_config_path(self) -> None:
        options = Options(context={"frame_type": "range", "frame_unit": "7"})

        assert FrameSpecFeatureGroup.resolve_subtype("placeholder", options) == "range_7"
        assert FrameSpecFeatureGroup.supports_compute_framework("placeholder", options, SubtypeFwBeta) is False

    def test_identify_routes_around_the_framework_without_the_flattened_subtype(self) -> None:
        feature = Feature("revenue__range_7_framespec")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            FrameSpecFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is FrameSpecFeatureGroup
        assert compute_frameworks == {SubtypeFwAlpha}, (
            f"'range_7' is unsupported on SubtypeFwBeta and must be narrowed out; got {compute_frameworks}"
        )

    def test_supported_flattened_subtype_keeps_every_framework(self) -> None:
        feature = Feature("revenue__rows_1_framespec")

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            FrameSpecFeatureGroup: {SubtypeFwAlpha, SubtypeFwBeta},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        _, compute_frameworks = identified.get()
        assert compute_frameworks == {SubtypeFwAlpha, SubtypeFwBeta}


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

    def test_abstract_class_reports_no_support_matrix(self) -> None:
        """An abstract base declares the universe; only concrete classes carry the support."""
        assert inspect.isabstract(AbstractStatFeatureGroup) is True
        assert AbstractStatFeatureGroup.subtype_universe() == frozenset({"sum", "median"})
        assert AbstractStatFeatureGroup.subtype_support_matrix() == {}, (
            "An abstract base with compute_framework_rule() -> None must not claim that every "
            "installed framework supports every subtype"
        )

    def test_concrete_class_with_rule_none_reports_every_framework(self) -> None:
        """Rule None genuinely means 'all frameworks': abstractness, not the rule, gates the matrix."""
        assert inspect.isabstract(OpenConcreteStatFeatureGroup) is False
        assert OpenConcreteStatFeatureGroup.compute_framework_rule() is None

        matrix = OpenConcreteStatFeatureGroup.subtype_support_matrix()

        expected_frameworks = {cfw.get_class_name() for cfw in get_all_subclasses(ComputeFramework)}
        assert set(matrix) == expected_frameworks
        assert "SubtypeFwAlpha" in matrix
        assert all(supported == frozenset({"sum", "median"}) for supported in matrix.values())

    def test_raises_when_a_capability_lies_outside_the_universe(self) -> None:
        class MisdeclaredCapabilityFeatureGroup(SplitStatFeatureGroup):
            """Declares a capability outside the universe: the matrix must reject it."""

            @classmethod
            def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
                return frozenset({"sum", "bogus_stat"})

        with pytest.raises(ValueError) as exc_info:
            MisdeclaredCapabilityFeatureGroup.subtype_support_matrix()

        message = str(exc_info.value)
        assert "bogus_stat" in message, f"Error must name the offending subtype, but got: {message}"

    def test_flattened_family_matrix_uses_the_overridden_universe(self) -> None:
        matrix = FrameSpecFeatureGroup.subtype_support_matrix()

        assert matrix == {
            "SubtypeFwAlpha": frozenset({"rows_1", "rows_7", "range_1", "range_7"}),
            "SubtypeFwBeta": frozenset({"rows_1", "rows_7"}),
        }


class TestSubtypeKeyClassDefinitionValidation:
    """Contract 8, Shape A: SUBTYPE_KEY must name a declared key with a value space."""

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

    def test_raises_for_an_undeclared_subtype_key_even_with_a_universe_override(self) -> None:
        """A universe override is no longer an escape hatch for a bogus SUBTYPE_KEY: use Shape B."""
        with pytest.raises(ValueError) as exc_info:

            class HalfFlattenedFeatureGroup(FeatureGroup):
                SUBTYPE_KEY = "frame_spec"

                PROPERTY_MAPPING = {
                    "frame_type": property_spec("frame type", strict=True, allowed_values=["rows", "range"]),
                }

                @classmethod
                def subtype_universe(cls) -> frozenset[str]:
                    return frozenset({"rows_1", "range_7"})

        message = str(exc_info.value)
        assert "HalfFlattenedFeatureGroup" in message, f"Error must name the class, but got: {message}"
        assert "frame_spec" in message, f"Error must name the bad key, but got: {message}"

    def test_no_raise_for_a_declared_subtype_key(self) -> None:
        class ValidSubtypeFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = "stat_type"

            PROPERTY_MAPPING = {
                "stat_type": property_spec("statistic", strict=True, allowed_values=["sum", "median"]),
            }

        assert ValidSubtypeFeatureGroup.subtype_universe() == frozenset({"sum", "median"})

    def test_no_raise_when_a_declared_key_is_paired_with_a_universe_override(self) -> None:
        class NarrowedUniverseFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = "stat_type"

            PROPERTY_MAPPING = {
                "stat_type": property_spec("statistic", strict=True, allowed_values=["sum", "median"]),
            }

            @classmethod
            def subtype_universe(cls) -> frozenset[str]:
                return frozenset({"sum"})

        assert NarrowedUniverseFeatureGroup.subtype_universe() == frozenset({"sum"})


class TestSubtypeKeyValueSpaceValidation:
    """Contract 8: a SUBTYPE_KEY with no enumerable value space is a class-definition error."""

    def test_raises_for_a_predicate_only_subtype_key(self) -> None:
        """strict_validation by predicate accepts values that the universe cannot enumerate."""
        with pytest.raises(ValueError) as exc_info:

            class PredicateOnlySubtypeFeatureGroup(FeatureGroup):
                SUBTYPE_KEY = "stat_type"

                PROPERTY_MAPPING = {
                    "stat_type": property_spec(
                        "statistic",
                        strict=True,
                        validation_function=lambda value: isinstance(value, str),
                    ),
                }

        message = str(exc_info.value)
        assert "PredicateOnlySubtypeFeatureGroup" in message, f"Error must name the class, but got: {message}"
        assert "stat_type" in message, f"Error must name the key, but got: {message}"
        assert "allowed_values" in message or "subtype_universe" in message, (
            f"Error must point at allowed_values or a subtype_universe() override, but got: {message}"
        )

    def test_raises_for_a_key_without_any_declared_value_space(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class EmptySpaceSubtypeFeatureGroup(FeatureGroup):
                SUBTYPE_KEY = "stat_type"

                PROPERTY_MAPPING = {
                    "stat_type": {
                        "explanation": "A key that declares no value space",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: False,
                    },
                }

        message = str(exc_info.value)
        assert "EmptySpaceSubtypeFeatureGroup" in message, f"Error must name the class, but got: {message}"
        assert "stat_type" in message, f"Error must name the key, but got: {message}"

    def test_no_raise_when_the_predicate_only_key_pairs_with_a_universe_override(self) -> None:
        class PredicateWithUniverseFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = "stat_type"

            PROPERTY_MAPPING = {
                "stat_type": property_spec(
                    "statistic",
                    strict=True,
                    validation_function=lambda value: isinstance(value, str),
                ),
            }

            @classmethod
            def subtype_universe(cls) -> frozenset[str]:
                return frozenset({"sum", "median"})

        assert PredicateWithUniverseFeatureGroup.subtype_universe() == frozenset({"sum", "median"})

    def test_no_raise_for_allowed_values(self) -> None:
        class AllowedValuesSubtypeFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = "stat_type"

            PROPERTY_MAPPING = {
                "stat_type": property_spec("statistic", strict=True, allowed_values=["sum", "median"]),
            }

        assert AllowedValuesSubtypeFeatureGroup.subtype_universe() == frozenset({"sum", "median"})


class TestFlattenedShapeClassDefinitionValidation:
    """Contract 8, Shape B: a universe override without SUBTYPE_KEY must pair with resolve_subtype."""

    def test_raises_when_the_universe_override_is_unpaired(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class UnpairedFlattenedFeatureGroup(FeatureGroup):
                SUBTYPE_KEY = None

                PROPERTY_MAPPING = {
                    "frame_type": property_spec("frame type", strict=True, allowed_values=["rows", "range"]),
                    "frame_unit": property_spec("frame unit", strict=True, allowed_values=["1", "7"]),
                }

                @classmethod
                def subtype_universe(cls) -> frozenset[str]:
                    return frozenset({"rows_1", "range_7"})

        message = str(exc_info.value)
        assert "UnpairedFlattenedFeatureGroup" in message, f"Error must name the class, but got: {message}"
        assert "subtype_universe" in message, f"Error must name subtype_universe(), but got: {message}"
        assert "resolve_subtype" in message, f"Error must name resolve_subtype(), but got: {message}"

    def test_no_raise_when_both_overrides_are_present(self) -> None:
        class PairedFlattenedFeatureGroup(FeatureGroup):
            SUBTYPE_KEY = None

            PROPERTY_MAPPING = {
                "frame_type": property_spec("frame type", strict=True, allowed_values=["rows", "range"]),
                "frame_unit": property_spec("frame unit", strict=True, allowed_values=["1", "7"]),
            }

            @classmethod
            def subtype_universe(cls) -> frozenset[str]:
                return frozenset({"rows_1", "range_7"})

            @classmethod
            def resolve_subtype(cls, feature_name: FeatureName | str, options: Options) -> Optional[str]:
                frame_type = options.get("frame_type")
                frame_unit = options.get("frame_unit")
                if frame_type is None or frame_unit is None:
                    return None
                return f"{frame_type}_{frame_unit}"

        options = Options(context={"frame_type": "range", "frame_unit": "7"})
        assert PairedFlattenedFeatureGroup.subtype_universe() == frozenset({"rows_1", "range_7"})
        assert PairedFlattenedFeatureGroup.resolve_subtype("placeholder", options) == "range_7"

    def test_no_raise_without_any_subtype_declaration(self) -> None:
        class PlainFeatureGroup(FeatureGroup):
            PROPERTY_MAPPING = {
                "frame_type": property_spec("frame type", strict=True, allowed_values=["rows", "range"]),
            }

        assert PlainFeatureGroup.subtype_universe() == frozenset()


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
