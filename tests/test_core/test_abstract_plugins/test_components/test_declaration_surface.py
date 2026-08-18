"""The shared mechanics behind ``FeatureGroup.PROPERTY_MAPPING`` and ``BaseInputData.READER_OPTIONS``:
one spec type, one per-key validator, and one MRO-merge helper used by the reader (``FeatureGroup``
keeps plain-attribute lookup on ``main``). Two attribute names; ``DeclarationSurface`` says which
fields are inert on which.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.declaration_surface import (
    DeclarationSurface,
    merged_declaration,
    own_declaration,
    reject_merge_cache_assignment,
    validate_declaration,
    validate_property_spec,
)


def _dsurf_match_guard(value: Any) -> bool:
    return True


def _dsurf_required_when(options: Any) -> bool:
    return False


class TestDeclarationSurfaceEnum:
    def test_feature_group_attr(self) -> None:
        assert DeclarationSurface.FEATURE_GROUP.attr == "PROPERTY_MAPPING"

    def test_feature_group_cache_attr(self) -> None:
        assert DeclarationSurface.FEATURE_GROUP.cache_attr == "_property_mapping_cache"

    def test_reader_attr(self) -> None:
        assert DeclarationSurface.READER.attr == "READER_OPTIONS"

    def test_reader_cache_attr(self) -> None:
        assert DeclarationSurface.READER.cache_attr == "_reader_option_specs_cache"

    def test_exactly_two_members(self) -> None:
        assert {member.name for member in DeclarationSurface} == {"FEATURE_GROUP", "READER"}


class TestOwnDeclaration:
    """``own_declaration`` reads exactly ``klass.__dict__[surface.attr]``, never the merge."""

    def test_no_own_declaration_returns_empty_dict_on_feature_group_surface(self) -> None:
        class DsurfNoOwnFg:
            pass

        assert own_declaration(DsurfNoOwnFg, DeclarationSurface.FEATURE_GROUP) == {}

    def test_no_own_declaration_returns_empty_dict_on_reader_surface(self) -> None:
        class DsurfNoOwnReader:
            pass

        assert own_declaration(DsurfNoOwnReader, DeclarationSurface.READER) == {}

    def test_own_property_mapping_declaration_is_returned(self) -> None:
        spec = PropertySpec("x", default=None)

        class DsurfOwnFgDecl:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_a": spec}

        assert own_declaration(DsurfOwnFgDecl, DeclarationSurface.FEATURE_GROUP) == {"dsurf_a": spec}

    def test_own_reader_options_declaration_is_returned(self) -> None:
        spec = PropertySpec("x", default=None)

        class DsurfOwnReaderDecl:
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"dsurf_a": spec}

        assert own_declaration(DsurfOwnReaderDecl, DeclarationSurface.READER) == {"dsurf_a": spec}

    def test_none_own_declaration_on_feature_group_surface_names_property_mapping(self) -> None:
        class DsurfNoneOwnFg:
            PROPERTY_MAPPING = None

        with pytest.raises(ValueError) as exc_info:
            own_declaration(DsurfNoneOwnFg, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfNoneOwnFg" in message
        assert "PROPERTY_MAPPING" in message
        assert "hierarchy" in message
        assert "inherited" in message
        assert "None" in message

    def test_none_own_declaration_on_reader_surface_names_reader_options(self) -> None:
        class DsurfNoneOwnReader:
            READER_OPTIONS = None

        with pytest.raises(ValueError) as exc_info:
            own_declaration(DsurfNoneOwnReader, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfNoneOwnReader" in message
        assert "READER_OPTIONS" in message
        assert "PROPERTY_MAPPING" not in message

    def test_non_dict_own_declaration_on_feature_group_surface_names_the_type(self) -> None:
        class DsurfListOwnFg:
            PROPERTY_MAPPING = ["not", "a", "dict"]

        with pytest.raises(ValueError) as exc_info:
            own_declaration(DsurfListOwnFg, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfListOwnFg.PROPERTY_MAPPING is a list, not a dict." in message

    def test_non_dict_own_declaration_on_reader_surface_names_the_type(self) -> None:
        class DsurfIntOwnReader:
            READER_OPTIONS = 42

        with pytest.raises(ValueError) as exc_info:
            own_declaration(DsurfIntOwnReader, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfIntOwnReader.READER_OPTIONS is a int, not a dict." in message


class TestMergedDeclaration:
    """Reversed-MRO merge, most-derived winning, cached per class under ``surface.cache_attr``."""

    def test_parent_only_declaration_merges_alone_on_feature_group_surface(self) -> None:
        a_spec = PropertySpec("a", default="parent_a")

        class DsurfMergeFgParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_a": a_spec}

        assert merged_declaration(DsurfMergeFgParent, DeclarationSurface.FEATURE_GROUP) == {"dsurf_a": a_spec}

    def test_parent_only_declaration_merges_alone_on_reader_surface(self) -> None:
        a_spec = PropertySpec("a", default="parent_a")

        class DsurfMergeReaderParent:
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"dsurf_a": a_spec}

        assert merged_declaration(DsurfMergeReaderParent, DeclarationSurface.READER) == {"dsurf_a": a_spec}

    def test_child_merges_parent_and_own_keys_on_feature_group_surface(self) -> None:
        a_spec = PropertySpec("a", default="parent_a")
        b_spec = PropertySpec("b", default="child_b")

        class DsurfMergeFgParent2:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_a": a_spec}

        class DsurfMergeFgChild2(DsurfMergeFgParent2):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_b": b_spec}

        assert merged_declaration(DsurfMergeFgChild2, DeclarationSurface.FEATURE_GROUP) == {
            "dsurf_a": a_spec,
            "dsurf_b": b_spec,
        }

    def test_child_merges_parent_and_own_keys_on_reader_surface(self) -> None:
        a_spec = PropertySpec("a", default="parent_a")
        b_spec = PropertySpec("b", default="child_b")

        class DsurfMergeReaderParent2:
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"dsurf_a": a_spec}

        class DsurfMergeReaderChild2(DsurfMergeReaderParent2):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"dsurf_b": b_spec}

        assert merged_declaration(DsurfMergeReaderChild2, DeclarationSurface.READER) == {
            "dsurf_a": a_spec,
            "dsurf_b": b_spec,
        }

    def test_most_derived_declaration_wins_on_a_key_collision(self) -> None:
        parent_spec = PropertySpec("a", default="parent")
        child_spec = PropertySpec("a", default="child")

        class DsurfCollideParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_key": parent_spec}

        class DsurfCollideChild(DsurfCollideParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_key": child_spec}

        assert merged_declaration(DsurfCollideChild, DeclarationSurface.FEATURE_GROUP)["dsurf_key"] is child_spec
        assert merged_declaration(DsurfCollideParent, DeclarationSurface.FEATURE_GROUP)["dsurf_key"] is parent_spec

    def test_child_declaration_does_not_leak_into_the_parent(self) -> None:
        class DsurfLeakParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {}

        class DsurfLeakChild(DsurfLeakParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_only_child": PropertySpec("x", default=None),
            }

        assert "dsurf_only_child" not in merged_declaration(DsurfLeakParent, DeclarationSurface.FEATURE_GROUP)

    def test_cache_lives_in_the_class_own_dict_under_the_surface_cache_attr(self) -> None:
        class DsurfCacheClass:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_k": PropertySpec("x", default=None)}

        merged_declaration(DsurfCacheClass, DeclarationSurface.FEATURE_GROUP)

        assert DeclarationSurface.FEATURE_GROUP.cache_attr in DsurfCacheClass.__dict__

    def test_a_subclass_defined_after_a_warm_parent_cache_sees_its_own_declaration(self) -> None:
        class DsurfColdParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_cache_key": PropertySpec("p", default="parent"),
            }

        assert merged_declaration(DsurfColdParent, DeclarationSurface.FEATURE_GROUP)["dsurf_cache_key"].default == (
            "parent"
        )

        class DsurfLateChild(DsurfColdParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_cache_key": PropertySpec("c", default="child"),
            }

        assert merged_declaration(DsurfLateChild, DeclarationSurface.FEATURE_GROUP)["dsurf_cache_key"].default == (
            "child"
        )
        assert merged_declaration(DsurfColdParent, DeclarationSurface.FEATURE_GROUP)["dsurf_cache_key"].default == (
            "parent"
        )

    def test_a_key_added_by_a_late_subclass_is_visible_after_a_warm_parent_cache(self) -> None:
        class DsurfKeysParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_parent_key": PropertySpec("Parent only.", default=None),
            }

        assert set(merged_declaration(DsurfKeysParent, DeclarationSurface.FEATURE_GROUP)) == {"dsurf_parent_key"}

        class DsurfKeysChild(DsurfKeysParent):
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_child_key": PropertySpec("Child only.", default=None),
            }

        assert set(merged_declaration(DsurfKeysChild, DeclarationSurface.FEATURE_GROUP)) == {
            "dsurf_parent_key",
            "dsurf_child_key",
        }
        assert "dsurf_child_key" not in merged_declaration(DsurfKeysParent, DeclarationSurface.FEATURE_GROUP)

    def test_a_subclass_never_answers_from_a_parents_warm_cache(self) -> None:
        class DsurfNeverParent:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_never_p": PropertySpec("p", default=None),
            }

        merged_declaration(DsurfNeverParent, DeclarationSurface.FEATURE_GROUP)

        class DsurfNeverChild(DsurfNeverParent):
            pass

        merged_declaration(DsurfNeverChild, DeclarationSurface.FEATURE_GROUP)

        assert DeclarationSurface.FEATURE_GROUP.cache_attr in DsurfNeverChild.__dict__

    def test_repeated_reads_return_the_same_cached_object(self) -> None:
        class DsurfRepeat:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"dsurf_k": PropertySpec("x", default=None)}

        first = merged_declaration(DsurfRepeat, DeclarationSurface.FEATURE_GROUP)
        second = merged_declaration(DsurfRepeat, DeclarationSurface.FEATURE_GROUP)

        assert first is second

    def test_base_input_data_subclass_merges_the_reserved_key_with_its_own(self) -> None:
        class DsurfReaderMergeSub(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_reader_merge_key": PropertySpec("x", default=None),
            }

        merged = merged_declaration(DsurfReaderMergeSub, DeclarationSurface.READER)

        assert "dsurf_reader_merge_key" in merged
        assert "BaseInputData" in merged

    def test_base_input_data_subclass_most_derived_declaration_wins(self) -> None:
        class DsurfReaderMergeParent(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_reader_merge_key": PropertySpec("parent", default="parent_v"),
            }

        class DsurfReaderMergeChild(DsurfReaderMergeParent):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_reader_merge_key": PropertySpec("child", default="child_v"),
            }

        child_merged = merged_declaration(DsurfReaderMergeChild, DeclarationSurface.READER)
        parent_merged = merged_declaration(DsurfReaderMergeParent, DeclarationSurface.READER)

        assert child_merged["dsurf_reader_merge_key"].default == "child_v"
        assert parent_merged["dsurf_reader_merge_key"].default == "parent_v"


class TestAPlainClassDeclaringBothSurfacesMergesSeparately:
    """One class declaring both ``PROPERTY_MAPPING`` and ``READER_OPTIONS`` merges each independently."""

    def test_two_distinct_cache_attributes_after_reading_both_surfaces(self) -> None:
        class DsurfBothDecl:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_fg_key": PropertySpec("fg", default=None),
            }
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_reader_key": PropertySpec("reader", default=None),
            }

        fg_merged = merged_declaration(DsurfBothDecl, DeclarationSurface.FEATURE_GROUP)
        reader_merged = merged_declaration(DsurfBothDecl, DeclarationSurface.READER)

        assert fg_merged == {"dsurf_fg_key": DsurfBothDecl.PROPERTY_MAPPING["dsurf_fg_key"]}
        assert reader_merged == {"dsurf_reader_key": DsurfBothDecl.READER_OPTIONS["dsurf_reader_key"]}
        assert DeclarationSurface.FEATURE_GROUP.cache_attr in DsurfBothDecl.__dict__
        assert DeclarationSurface.READER.cache_attr in DsurfBothDecl.__dict__
        assert (
            DsurfBothDecl.__dict__[DeclarationSurface.FEATURE_GROUP.cache_attr]
            is not DsurfBothDecl.__dict__[DeclarationSurface.READER.cache_attr]
        )


class TestRejectMergeCacheAssignment:
    """The merge cache is framework-written per surface; a class body assigning it is rejected."""

    def test_raises_naming_the_feature_group_cache_attr(self) -> None:
        class DsurfCacheAssignFg:
            _property_mapping_cache: ClassVar[Any] = {}

        with pytest.raises(ValueError) as exc_info:
            reject_merge_cache_assignment(DsurfCacheAssignFg, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfCacheAssignFg" in message
        assert "_property_mapping_cache" in message
        assert "framework" in message

    def test_raises_naming_the_reader_cache_attr(self) -> None:
        class DsurfCacheAssignReader:
            _reader_option_specs_cache: ClassVar[Any] = {}

        with pytest.raises(ValueError) as exc_info:
            reject_merge_cache_assignment(DsurfCacheAssignReader, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfCacheAssignReader" in message
        assert "_reader_option_specs_cache" in message
        assert "framework" in message

    def test_no_raise_when_cache_attr_absent(self) -> None:
        class DsurfNoCacheAssign:
            pass

        reject_merge_cache_assignment(DsurfNoCacheAssign, DeclarationSurface.FEATURE_GROUP)
        reject_merge_cache_assignment(DsurfNoCacheAssign, DeclarationSurface.READER)

    def test_a_cache_warmed_by_merged_declaration_is_not_blamed(self) -> None:
        """The cache reject guard runs on the class body only; a value the framework itself wrote
        via ``merged_declaration`` before the guard fires must never be mistaken for an authored one."""

        class DsurfWarmedCache:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_warmed_k": PropertySpec("x", default=None),
            }

        merged_declaration(DsurfWarmedCache, DeclarationSurface.FEATURE_GROUP)

        reject_merge_cache_assignment(DsurfWarmedCache, DeclarationSurface.FEATURE_GROUP)


class TestPoisonedCacheWithoutABaseHook:
    """A plain class (no ``FeatureGroup``/``BaseInputData`` base, so no ``__init_subclass__`` hook
    ever ran ``reject_merge_cache_assignment``) can plant a bogus cache value directly in its body.
    ``merged_declaration`` must refuse to trust it too, not silently hand it back or blow up on a
    missing attribute of the merge-cache-internal type."""

    def test_merged_declaration_raises_on_a_poisoned_feature_group_cache(self) -> None:
        class DsurfPoisonedFgCache:
            _property_mapping_cache: ClassVar[Any] = {"k": PropertySpec("x", default=None)}

        with pytest.raises(ValueError) as merged_exc:
            merged_declaration(DsurfPoisonedFgCache, DeclarationSurface.FEATURE_GROUP)
        merged_message = str(merged_exc.value)
        del merged_exc

        with pytest.raises(ValueError) as guard_exc:
            reject_merge_cache_assignment(DsurfPoisonedFgCache, DeclarationSurface.FEATURE_GROUP)
        guard_message = str(guard_exc.value)
        del guard_exc

        assert "DsurfPoisonedFgCache" in merged_message
        assert "_property_mapping_cache" in merged_message
        assert merged_message == guard_message

    def test_merged_declaration_raises_on_a_poisoned_reader_cache(self) -> None:
        class DsurfPoisonedReaderCache:
            _reader_option_specs_cache: ClassVar[Any] = {"k": PropertySpec("x", default=None)}

        with pytest.raises(ValueError) as merged_exc:
            merged_declaration(DsurfPoisonedReaderCache, DeclarationSurface.READER)
        merged_message = str(merged_exc.value)
        del merged_exc

        with pytest.raises(ValueError) as guard_exc:
            reject_merge_cache_assignment(DsurfPoisonedReaderCache, DeclarationSurface.READER)
        guard_message = str(guard_exc.value)
        del guard_exc

        assert "DsurfPoisonedReaderCache" in merged_message
        assert "_reader_option_specs_cache" in merged_message
        assert merged_message == guard_message


class TestValidatePropertySpec:
    """The per-key rules, parametrized by surface; ``via`` appends the reached-through suffix."""

    def test_non_property_spec_value_rejected_on_reader_surface(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", "not a spec", DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert message == (
            "DsurfOwner.READER_OPTIONS['dsurf_key'] is a str, not a PropertySpec. "
            "Construct PropertySpec(...) or use the property_spec(...) helper."
        )

    def test_non_property_spec_value_rejected_on_feature_group_surface(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfOwner.PROPERTY_MAPPING['dsurf_key'] is a int, not a PropertySpec." in message

    def test_via_suffix_is_appended(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfMixin", "dsurf_key", 42, DeclarationSurface.READER, via="DsurfReader")

        message = str(exc_info.value)
        del exc_info
        assert message.endswith(" (reached defining DsurfReader)")

    def test_no_via_suffix_when_via_is_none(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "(reached defining" not in message

    def test_feature_group_rejects_framework_set(self) -> None:
        spec = PropertySpec("x", default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "reader-only" in message

    def test_feature_group_rejects_scalar_only(self) -> None:
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, scalar_only=True, default="a")

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "scalar_only" in message
        assert "reader-only" in message

    def test_reader_rejects_match_guard(self) -> None:
        spec = PropertySpec("x", match_guard=_dsurf_match_guard, default=None)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "match_guard" in message

    def test_reader_rejects_deferred_binding(self) -> None:
        spec = PropertySpec("x", deferred_binding=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "deferred_binding" in message

    def test_reader_rejects_context_false(self) -> None:
        spec = PropertySpec("x", context=False, default=None)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "context" in message

    def test_reader_rejects_framework_set_with_strict_validation(self) -> None:
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, default="a", framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "strict_validation" in message

    def test_reader_rejects_framework_set_with_required_when(self) -> None:
        spec = PropertySpec("x", required_when=_dsurf_required_when, default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "required_when" in message

    def test_reader_rejects_framework_set_with_allow_explicit_none(self) -> None:
        spec = PropertySpec("x", allow_explicit_none=True, default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "allow_explicit_none" in message

    def test_reader_rejects_framework_set_with_no_declared_default(self) -> None:
        spec = PropertySpec("x", framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "default" in message

    def test_reader_accepts_framework_set_with_none_default(self) -> None:
        spec = PropertySpec("x", default=None, framework_set=True)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        assert result is spec

    def test_feature_group_accepts_match_guard(self) -> None:
        spec = PropertySpec("x", match_guard=_dsurf_match_guard, default=None)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        assert result is spec

    def test_feature_group_accepts_context_false(self) -> None:
        spec = PropertySpec("x", context=False, default=None)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        assert result is spec

    def test_reader_accepts_scalar_only_strict_with_allowed_values(self) -> None:
        spec = PropertySpec("x", scalar_only=True, strict_validation=True, allowed_values=("a", "b"), default="a")

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.READER)

        assert result is spec

    def test_reader_surface_message_contains_reader_options_not_property_mapping(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", "bad", DeclarationSurface.READER)

        message = str(exc_info.value)
        del exc_info
        assert "READER_OPTIONS" in message
        assert "PROPERTY_MAPPING" not in message

    def test_feature_group_surface_message_contains_property_mapping_not_reader_options(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "PROPERTY_MAPPING" in message
        assert "READER_OPTIONS" not in message

    def test_no_message_ever_mentions_the_retired_surface_marker_constant(self) -> None:
        messages = []

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", "bad", DeclarationSurface.READER)
        messages.append(str(exc_info.value))
        del exc_info

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec(
                "DsurfOwner",
                "dsurf_key",
                PropertySpec("x", framework_set=True, default=None),
                DeclarationSurface.FEATURE_GROUP,
            )
        messages.append(str(exc_info.value))
        del exc_info

        for message in messages:
            assert "PROPERTY_MAPPING_SURFACE" not in message


class TestValidateDeclaration:
    """Class-definition validation for a class whose MRO contains ``root``; a direct call passes
    ``root`` explicitly (here: ``BaseInputData``)."""

    def test_a_mixin_reader_invalid_spec_is_rejected_with_via_naming_the_subclass(self) -> None:
        class DsurfBadMixin:
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_bad_key": PropertySpec("Guarded.", match_guard=_dsurf_match_guard, default=None),
            }

        with pytest.raises(ValueError) as exc_info:

            class DsurfBadMixinReader(DsurfBadMixin, BaseInputData):
                pass

        message = str(exc_info.value)
        del exc_info
        assert "DsurfBadMixin" in message
        assert "dsurf_bad_key" in message
        assert "match_guard" in message
        assert "(reached defining DsurfBadMixinReader)" in message

    def test_a_valid_mixin_declaration_defines_fine_and_merges(self) -> None:
        spec = PropertySpec("Valid, declared on the mixin.", default="dsurf_mixin_default")

        class DsurfGoodMixin:
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"dsurf_good_key": spec}

        class DsurfGoodMixinReader(DsurfGoodMixin, BaseInputData):
            pass

        assert merged_declaration(DsurfGoodMixinReader, DeclarationSurface.READER)["dsurf_good_key"] is spec

    def test_own_none_declaration_on_cls_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class DsurfNoneOwnReader(BaseInputData):
                READER_OPTIONS = None  # type: ignore[assignment]  # invalid shape is the point

        message = str(exc_info.value)
        del exc_info
        assert "DsurfNoneOwnReader" in message

    def test_direct_call_on_a_clean_subclass_does_not_raise(self) -> None:
        class DsurfCleanReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_clean_key": PropertySpec("Valid.", default=None),
            }

        validate_declaration(DsurfCleanReader, DeclarationSurface.READER, BaseInputData)

    def test_a_class_below_the_root_is_not_re_walked_as_a_plain_mixin(self) -> None:
        """A class whose own MRO already contains ``root`` validated itself at its own definition;
        ``validate_declaration`` on a grandchild must not re-raise for it a second time under a
        fresh ``via`` and must not choke walking it as though it had no root."""

        class DsurfRealInheritanceParent(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_parent_key": PropertySpec("Valid.", default=None),
            }

        class DsurfRealInheritanceChild(DsurfRealInheritanceParent):
            pass

        validate_declaration(DsurfRealInheritanceChild, DeclarationSurface.READER, BaseInputData)


class TestValidateDeclarationWalksEveryAncestorWhenThereIsNoRoot:
    """With ``root=None``, EVERY ancestor is walked as a plain mixin, not just the immediate one;
    a defect declared two classes up the plain hierarchy is still caught."""

    def test_a_defect_declared_two_classes_up_is_caught_naming_the_declaring_ancestor_via_the_leaf(self) -> None:
        class DsurfNoRootA:
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_bad_key": PropertySpec("Guarded.", match_guard=_dsurf_match_guard, default=None),
            }

        class DsurfNoRootB(DsurfNoRootA):
            pass

        with pytest.raises(ValueError) as exc_info:
            validate_declaration(DsurfNoRootB, DeclarationSurface.READER, None)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfNoRootA" in message
        assert "(reached defining DsurfNoRootB)" in message


class TestCooperativeHookReadingBeforeSuperIsNotBlamed:
    """A mixin's warm read before ``super()`` must not be mistaken for the class body assigning the cache."""

    def test_reader_side_mixin_reading_before_super_defines_fine(self) -> None:
        class DsurfReaderEarlyReadMixin:
            def __init_subclass__(cls, **kwargs: Any) -> None:
                cls.declared_reader_option_keys()  # type: ignore[attr-defined]
                super().__init_subclass__(**kwargs)

        class DsurfReaderEarlyReadSub(DsurfReaderEarlyReadMixin, BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "dsurf_early_reader_key": PropertySpec("x", default=None),
            }

        result = DsurfReaderEarlyReadSub.declared_reader_option_keys()
        del DsurfReaderEarlyReadSub
        del DsurfReaderEarlyReadMixin

        assert "dsurf_early_reader_key" in result
