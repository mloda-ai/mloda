"""Pins the shared PROPERTY_MAPPING declaration-surface module: the merge/validation machinery
both ``BaseInputData`` (READER surface) and ``FeatureGroup`` (FEATURE_GROUP surface) build on.
Plain classes exercise the generic mechanics; ``BaseInputData`` subclasses exercise the READER
surface wiring.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.property_mapping import (
    CACHE_ATTR,
    SURFACE_ATTR,
    DeclarationSurface,
    configuration_property_mapping,
    declares_property_mapping,
    merged_property_mapping,
    own_property_mapping,
    reject_merge_cache_assignment,
    validate_property_mapping,
    validate_property_spec,
)
from mloda.core.abstract_plugins.feature_group import FeatureGroup


def _pmsurf_match_guard(value: Any) -> bool:
    """A match_guard predicate; only its presence on a spec matters here."""
    return True


def _pmsurf_required_when(options: Any) -> bool:
    """A required_when predicate; only its presence on a spec matters here."""
    return False


class TestDeclarationSurfaceEnum:
    """The enum names the two declaration surfaces by their base class."""

    def test_feature_group_member_value(self) -> None:
        assert DeclarationSurface.FEATURE_GROUP.value == "FeatureGroup"

    def test_reader_member_value(self) -> None:
        assert DeclarationSurface.READER.value == "BaseInputData"

    def test_exactly_two_members(self) -> None:
        assert {member.name for member in DeclarationSurface} == {"FEATURE_GROUP", "READER"}


class TestSurfaceAndCacheAttrConstants:
    """The attribute-name constants, and BaseInputData's own use of ``SURFACE_ATTR``."""

    def test_surface_attr_constant(self) -> None:
        assert SURFACE_ATTR == "PROPERTY_MAPPING_SURFACE"

    def test_cache_attr_constant(self) -> None:
        assert CACHE_ATTR == "_property_mapping_cache"

    def test_base_input_data_declares_the_reader_surface_in_its_own_dict(self) -> None:
        """The surface base sets the attribute in its own class body, not merely inherits it."""
        assert BaseInputData.__dict__[SURFACE_ATTR] is DeclarationSurface.READER

    def test_base_input_data_surface_attribute_reads_the_same_value(self) -> None:
        assert BaseInputData.PROPERTY_MAPPING_SURFACE is DeclarationSurface.READER


class TestOwnPropertyMapping:
    """``own_property_mapping`` reads exactly the class's OWN ``__dict__``, never the merge."""

    def test_no_own_declaration_returns_empty_dict(self) -> None:
        class PmsurfNoOwnDecl:
            pass

        assert own_property_mapping(PmsurfNoOwnDecl) == {}

    def test_own_dict_declaration_is_returned(self) -> None:
        spec = PropertySpec("x", default=None)

        class PmsurfOwnDictDecl:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_a": spec}

        assert own_property_mapping(PmsurfOwnDictDecl) == {"pmsurf_a": spec}

    def test_none_own_declaration_raises_naming_the_class_with_the_merge_remedy(self) -> None:
        class PmsurfNoneOwnDecl:
            PROPERTY_MAPPING = None

        with pytest.raises(ValueError) as exc_info:
            own_property_mapping(PmsurfNoneOwnDecl)

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfNoneOwnDecl" in message
        assert "hierarchy" in message
        assert "inherited" in message
        assert "None" in message

    def test_non_dict_own_declaration_raises_naming_the_type(self) -> None:
        class PmsurfListOwnDecl:
            PROPERTY_MAPPING = ["not", "a", "dict"]

        with pytest.raises(ValueError) as exc_info:
            own_property_mapping(PmsurfListOwnDecl)

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfListOwnDecl.PROPERTY_MAPPING is a list, not a dict." in message

    def test_non_dict_int_own_declaration_raises_naming_the_type(self) -> None:
        class PmsurfIntOwnDecl:
            PROPERTY_MAPPING = 42

        with pytest.raises(ValueError) as exc_info:
            own_property_mapping(PmsurfIntOwnDecl)

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfIntOwnDecl.PROPERTY_MAPPING is a int, not a dict." in message


class TestMergedPropertyMapping:
    """Reversed-MRO merge, most-derived winning, cached in the class's own ``__dict__``."""

    def test_parent_only_declaration_merges_alone(self) -> None:
        a_spec = PropertySpec("a", default="parent_a")

        class PmsurfMergeParentOnly:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_a": a_spec}

        assert merged_property_mapping(PmsurfMergeParentOnly) == {"pmsurf_a": a_spec}

    def test_child_merges_parent_and_own_keys(self) -> None:
        a_spec = PropertySpec("a", default="parent_a")
        b_spec = PropertySpec("b", default="child_b")

        class PmsurfMergeParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_a": a_spec}

        class PmsurfMergeChild(PmsurfMergeParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_b": b_spec}

        assert merged_property_mapping(PmsurfMergeChild) == {"pmsurf_a": a_spec, "pmsurf_b": b_spec}

    def test_most_derived_declaration_wins_on_a_key_collision(self) -> None:
        parent_spec = PropertySpec("a", default="parent")
        child_spec = PropertySpec("a", default="child")

        class PmsurfCollideParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_key": parent_spec}

        class PmsurfCollideChild(PmsurfCollideParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_key": child_spec}

        assert merged_property_mapping(PmsurfCollideChild)["pmsurf_key"] is child_spec
        assert merged_property_mapping(PmsurfCollideParent)["pmsurf_key"] is parent_spec

    def test_child_declaration_does_not_leak_into_the_parent(self) -> None:
        class PmsurfLeakParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {}

        class PmsurfLeakChild(PmsurfLeakParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_only_child": PropertySpec("x", default=None),
            }

        assert "pmsurf_only_child" not in merged_property_mapping(PmsurfLeakParent)

    def test_cache_lives_in_the_class_own_dict(self) -> None:
        class PmsurfCacheClass:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_k": PropertySpec("x", default=None)}

        merged_property_mapping(PmsurfCacheClass)

        assert CACHE_ATTR in PmsurfCacheClass.__dict__

    def test_a_subclass_defined_after_a_warm_parent_cache_sees_its_own_declaration(self) -> None:
        class PmsurfColdParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_cache_key": PropertySpec("p", default="parent"),
            }

        assert merged_property_mapping(PmsurfColdParent)["pmsurf_cache_key"].default == "parent"

        class PmsurfLateChild(PmsurfColdParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_cache_key": PropertySpec("c", default="child"),
            }

        assert merged_property_mapping(PmsurfLateChild)["pmsurf_cache_key"].default == "child"
        assert merged_property_mapping(PmsurfColdParent)["pmsurf_cache_key"].default == "parent"

    def test_a_key_added_by_a_late_subclass_is_visible_after_a_warm_parent_cache(self) -> None:
        class PmsurfKeysParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_parent_key": PropertySpec("Parent only.", default=None),
            }

        assert set(merged_property_mapping(PmsurfKeysParent)) == {"pmsurf_parent_key"}

        class PmsurfKeysChild(PmsurfKeysParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_child_key": PropertySpec("Child only.", default=None),
            }

        assert set(merged_property_mapping(PmsurfKeysChild)) == {"pmsurf_parent_key", "pmsurf_child_key"}
        assert "pmsurf_child_key" not in merged_property_mapping(PmsurfKeysParent)

    def test_repeated_reads_return_the_same_cached_object(self) -> None:
        """Internal accessor: the SAME cached dict is handed back; a caller-facing wrapper copies it."""

        class PmsurfRepeat:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_k": PropertySpec("x", default=None)}

        first = merged_property_mapping(PmsurfRepeat)
        second = merged_property_mapping(PmsurfRepeat)

        assert first is second


class TestDeclaresPropertyMapping:
    """True when some class OTHER THAN the surface base declares its own PROPERTY_MAPPING."""

    def test_false_when_only_the_surface_base_declares(self) -> None:
        class PmsurfNoOwnReader(BaseInputData):
            pass

        assert declares_property_mapping(PmsurfNoOwnReader) is False

    def test_true_for_a_non_empty_own_declaration(self) -> None:
        class PmsurfOwnReader(BaseInputData):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_k": PropertySpec("x", default=None),
            }

        assert declares_property_mapping(PmsurfOwnReader) is True

    def test_true_for_an_explicit_empty_own_declaration(self) -> None:
        class PmsurfEmptyDeclReader(BaseInputData):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {}

        assert declares_property_mapping(PmsurfEmptyDeclReader) is True

    def test_true_for_a_plain_mixin_declaration_in_the_mro(self) -> None:
        class PmsurfDeclaresMixin:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_mixin_k": PropertySpec("x", default=None),
            }

        class PmsurfMixinReader(PmsurfDeclaresMixin, BaseInputData):
            pass

        assert declares_property_mapping(PmsurfMixinReader) is True

    def test_plain_hierarchy_without_a_surface_base_counts_any_own_declaration(self) -> None:
        class PmsurfPlainBase:
            pass

        class PmsurfPlainChild(PmsurfPlainBase):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_k": PropertySpec("x", default=None),
            }

        assert declares_property_mapping(PmsurfPlainChild) is True
        assert declares_property_mapping(PmsurfPlainBase) is False


class TestConfigurationPropertyMapping:
    """``None`` when nothing is declared beyond the surface base, else the merged mapping."""

    def test_none_when_declares_property_mapping_is_false(self) -> None:
        class PmsurfConfigNoneReader(BaseInputData):
            pass

        assert configuration_property_mapping(PmsurfConfigNoneReader) is None

    def test_merged_result_when_declares_property_mapping_is_true(self) -> None:
        spec = PropertySpec("x", default=None)

        class PmsurfConfigReader(BaseInputData):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_config_k": spec}

        result = configuration_property_mapping(PmsurfConfigReader)

        assert result is not None
        assert result["pmsurf_config_k"] is spec


class TestRejectMergeCacheAssignment:
    """The merge cache is framework-written; a class body assigning it is rejected where written."""

    def test_raises_when_cache_attr_present_in_own_dict(self) -> None:
        class PmsurfCacheAssign:
            _property_mapping_cache: ClassVar[Any] = {}

        with pytest.raises(ValueError) as exc_info:
            reject_merge_cache_assignment(PmsurfCacheAssign)

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfCacheAssign" in message
        assert "_property_mapping_cache" in message
        assert "framework" in message

    def test_no_raise_when_cache_attr_absent(self) -> None:
        class PmsurfNoCacheAssign:
            pass

        reject_merge_cache_assignment(PmsurfNoCacheAssign)


class TestValidatePropertySpec:
    """The per-key rules, parametrized by surface; ``via`` appends the reached-through suffix."""

    def test_non_property_spec_value_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", "not a spec", DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert message == (
            "PmsurfOwner.PROPERTY_MAPPING['pmsurf_key'] is a str, not a PropertySpec. "
            "Construct PropertySpec(...) or use the property_spec(...) helper."
        )

    def test_non_property_spec_value_rejected_on_feature_group_surface_too(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", 42, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfOwner.PROPERTY_MAPPING['pmsurf_key'] is a int, not a PropertySpec." in message

    def test_via_suffix_is_appended(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfMixin", "pmsurf_key", 42, DeclarationSurface.READER, via="PmsurfReader")

        message = str(exc_info.value)
        del exc_info
        assert message.endswith(" (reached defining PmsurfReader)")

    def test_no_via_suffix_when_via_is_none(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", 42, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "(reached defining" not in message

    def test_feature_group_rejects_framework_set(self) -> None:
        spec = PropertySpec("x", default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "reader-only" in message

    def test_feature_group_rejects_scalar_only(self) -> None:
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, scalar_only=True, default="a")

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "scalar_only" in message
        assert "reader-only" in message

    def test_reader_rejects_match_guard(self) -> None:
        spec = PropertySpec("x", match_guard=_pmsurf_match_guard, default=None)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "match_guard" in message

    def test_reader_rejects_deferred_binding(self) -> None:
        spec = PropertySpec("x", deferred_binding=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "deferred_binding" in message

    def test_reader_rejects_context_false(self) -> None:
        spec = PropertySpec("x", context=False, default=None)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "context" in message

    def test_reader_rejects_framework_set_with_strict_validation(self) -> None:
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, default="a", framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "strict_validation" in message

    def test_reader_rejects_framework_set_with_required_when(self) -> None:
        spec = PropertySpec("x", required_when=_pmsurf_required_when, default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "required_when" in message

    def test_reader_rejects_framework_set_with_allow_explicit_none(self) -> None:
        spec = PropertySpec("x", allow_explicit_none=True, default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "allow_explicit_none" in message

    def test_reader_rejects_framework_set_with_no_declared_default(self) -> None:
        spec = PropertySpec("x", framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "default" in message

    def test_reader_accepts_framework_set_with_none_default(self) -> None:
        spec = PropertySpec("x", default=None, framework_set=True)

        result = validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        assert result is spec

    def test_feature_group_accepts_match_guard(self) -> None:
        spec = PropertySpec("x", match_guard=_pmsurf_match_guard, default=None)

        result = validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        assert result is spec

    def test_feature_group_accepts_context_false(self) -> None:
        spec = PropertySpec("x", context=False, default=None)

        result = validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        assert result is spec

    def test_reader_accepts_scalar_only_strict_with_allowed_values(self) -> None:
        spec = PropertySpec("x", scalar_only=True, strict_validation=True, allowed_values=("a", "b"), default="a")

        result = validate_property_spec("PmsurfOwner", "pmsurf_key", spec, DeclarationSurface.READER)

        assert result is spec

    def test_no_message_ever_mentions_the_retired_reader_options_spelling(self) -> None:
        """All error text says PROPERTY_MAPPING, never the retired READER_OPTIONS spelling."""
        messages = []

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("PmsurfOwner", "pmsurf_key", "bad", DeclarationSurface.READER)
        messages.append(str(exc_info.value))
        del exc_info

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec(
                "PmsurfOwner",
                "pmsurf_key",
                PropertySpec("x", framework_set=True, default=None),
                DeclarationSurface.FEATURE_GROUP,
            )
        messages.append(str(exc_info.value))
        del exc_info

        for message in messages:
            assert "READER_OPTIONS" not in message


class TestValidatePropertyMapping:
    """Class-definition validation for a class whose MRO contains a surface base (here: BaseInputData)."""

    def test_a_mixin_reader_invalid_spec_is_rejected_with_via_naming_the_subclass(self) -> None:
        class PmsurfBadMixin:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_bad_key": PropertySpec("Guarded.", match_guard=_pmsurf_match_guard, default=None),
            }

        with pytest.raises(ValueError) as exc_info:

            class PmsurfBadMixinReader(PmsurfBadMixin, BaseInputData):
                pass

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfBadMixin" in message
        assert "pmsurf_bad_key" in message
        assert "match_guard" in message
        assert "(reached defining PmsurfBadMixinReader)" in message

    def test_a_valid_mixin_declaration_defines_fine_and_merges(self) -> None:
        spec = PropertySpec("Valid, declared on the mixin.", default="pmsurf_mixin_default")

        class PmsurfGoodMixin:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"pmsurf_good_key": spec}

        class PmsurfGoodMixinReader(PmsurfGoodMixin, BaseInputData):
            pass

        assert merged_property_mapping(PmsurfGoodMixinReader)["pmsurf_good_key"] is spec

    def test_own_none_declaration_on_cls_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class PmsurfNoneOwnReader(BaseInputData):
                PROPERTY_MAPPING = None  # type: ignore[assignment]  # invalid shape is the point

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfNoneOwnReader" in message

    def test_direct_call_on_a_clean_subclass_does_not_raise(self) -> None:
        class PmsurfCleanReader(BaseInputData):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_clean_key": PropertySpec("Valid.", default=None),
            }

        validate_property_mapping(PmsurfCleanReader)


class TestSurfaceMarkerIsFrameworkOwned:
    """PROPERTY_MAPPING_SURFACE is framework-written; a class body assigning it is rejected."""

    def test_feature_group_subclass_assigning_surface_marker_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class PmsurfFgAssignsSurface(FeatureGroup):
                PROPERTY_MAPPING_SURFACE = DeclarationSurface.FEATURE_GROUP

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfFgAssignsSurface" in message
        assert SURFACE_ATTR in message

    def test_base_input_data_subclass_assigning_surface_marker_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class PmsurfReaderAssignsSurface(BaseInputData):
                PROPERTY_MAPPING_SURFACE = "not_a_real_surface"  # type: ignore[assignment]  # invalid shape is the point

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfReaderAssignsSurface" in message
        assert SURFACE_ATTR in message

    def test_mixin_carrying_surface_marker_mixed_into_feature_group_raises(self) -> None:
        class PmsurfMixinAssignsSurface:
            PROPERTY_MAPPING_SURFACE = DeclarationSurface.FEATURE_GROUP

        with pytest.raises(ValueError) as exc_info:

            class PmsurfMixinSurfaceMixed(PmsurfMixinAssignsSurface, FeatureGroup):
                pass

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfMixinAssignsSurface" in message
        assert SURFACE_ATTR in message


class TestTwoSurfaceBasesInOneMroAreRejected:
    """A class whose MRO carries two distinct surface bases (FeatureGroup and BaseInputData) is
    an author mistake: exactly one surface may govern PROPERTY_MAPPING validation for a class."""

    def test_feature_group_then_base_input_data_raises(self) -> None:
        """Naming SURFACE_ATTR pins the dedicated two-surface-bases guard, not an unrelated,
        incidental rejection (e.g. the reader's framework_set key looking FeatureGroup-invalid)."""
        with pytest.raises(ValueError) as exc_info:

            class PmsurfTwoBasesFgFirst(FeatureGroup, BaseInputData):
                pass

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfTwoBasesFgFirst" in message
        assert "FeatureGroup" in message
        assert "BaseInputData" in message
        assert SURFACE_ATTR in message

    def test_base_input_data_then_feature_group_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class PmsurfTwoBasesReaderFirst(BaseInputData, FeatureGroup):  # type: ignore[misc]  # wrong shape is the point
                pass

        message = str(exc_info.value)
        del exc_info
        assert "PmsurfTwoBasesReaderFirst" in message
        assert "FeatureGroup" in message
        assert "BaseInputData" in message
        assert SURFACE_ATTR in message


class TestCooperativeHookReadingBeforeSuperIsNotBlamed:
    """A mixin's __init_subclass__ may read the merge before calling super(); that warm read must
    not be mistaken for the class body itself assigning the merge cache."""

    def test_feature_group_side_mixin_reading_before_super_defines_fine(self) -> None:
        class PmsurfFgEarlyReadMixin:
            def __init_subclass__(cls, **kwargs: Any) -> None:
                cls.declared_option_keys()  # type: ignore[attr-defined]
                super().__init_subclass__(**kwargs)

        class PmsurfFgEarlyReadSub(PmsurfFgEarlyReadMixin, FeatureGroup):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_early_fg_key": PropertySpec("x", default=None),
            }

        result = PmsurfFgEarlyReadSub.declared_option_keys()
        del PmsurfFgEarlyReadSub
        del PmsurfFgEarlyReadMixin

        assert "pmsurf_early_fg_key" in result

    def test_reader_side_mixin_reading_before_super_defines_fine(self) -> None:
        class PmsurfReaderEarlyReadMixin:
            def __init_subclass__(cls, **kwargs: Any) -> None:
                cls.declared_reader_option_keys()  # type: ignore[attr-defined]
                super().__init_subclass__(**kwargs)

        class PmsurfReaderEarlyReadSub(PmsurfReaderEarlyReadMixin, BaseInputData):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_early_reader_key": PropertySpec("x", default=None),
            }

        result = PmsurfReaderEarlyReadSub.declared_reader_option_keys()
        del PmsurfReaderEarlyReadSub
        del PmsurfReaderEarlyReadMixin

        assert "pmsurf_early_reader_key" in result


class TestConfigurationPropertyMappingHotPath:
    """The hot path hands back the cached merged mapping directly (``is``), not a fresh copy."""

    def test_repeated_calls_return_the_same_cached_object(self) -> None:
        class PmsurfHotPathClass(FeatureGroup):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmsurf_hot_key": PropertySpec("x", default=None),
            }

        first = configuration_property_mapping(PmsurfHotPathClass)
        second = configuration_property_mapping(PmsurfHotPathClass)
        del PmsurfHotPathClass

        assert first is second
