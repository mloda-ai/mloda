"""Pins Phase 2 of issue #949's follow-up: FeatureGroup.PROPERTY_MAPPING moves onto the shared
declaration-surface module (``mloda/core/abstract_plugins/components/property_mapping.py``), so a
subclass's declared options are the MRO-merged view, most-derived winning, not a plain-attribute
replacement. Covers the FEATURE_GROUP surface identity, the merge accessors, materialization
(``options_with_defaults`` / ``GlobalFilter._intake_fill``), config-path matching, guards reading
the merged view, author-time rejections, and plain-mixin declarations.

Today's bug (a child's own, incomplete PROPERTY_MAPPING masks its parent's) makes several of these
children an accidental "universal matcher" (issue #771): they'd match any unrelated feature name
resolved elsewhere in the same test session. Those helpers are therefore built fresh, function-local,
by a factory (never module-level), and every using test drops its local reference before any
assertion that may fail today, so a failing assertion's own traceback cannot keep the class registered
for the rest of the run (#845). Once Green's merge lands, the same children stop being universal.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.property_mapping import DeclarationSurface
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.filter.global_filter import GlobalFilter
from mloda.user import Options

# --- Items 2-4: MRO merge, most-derived wins; materialization and config-path matching follow it. ---

PMFG_A_KEY = "pmfg_merge_a"
PMFG_B_KEY = "pmfg_merge_b"
PMFG_C_KEY = "pmfg_merge_c"


class PmfgMergeParent(FeatureChainParserMixin, FeatureGroup):
    """Declares 'a' (concrete default) and 'b' (NO_DEFAULT, required); a pattern names-captures 'a'.

    Safe at module level: 'b' is unconditionally required, so this class never becomes a universal
    matcher on the config path (unlike the children built by the factories below).
    """

    PREFIX_PATTERN = r".*__(?P<pmfg_merge_a>\w+)_pmfg_merge_op$"
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        PMFG_A_KEY: PropertySpec("a, optional.", default="parent_a_default"),
        PMFG_B_KEY: PropertySpec("b, required."),
    }


def _make_merge_child() -> type[PmfgMergeParent]:
    """A fresh subclass declaring only 'c'; function-local (see module docstring)."""

    class PmfgMergeChildLocal(PmfgMergeParent):
        PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
            PMFG_C_KEY: PropertySpec("c, optional.", default="child_c_default"),
        }

    return PmfgMergeChildLocal


def _make_merge_child2() -> type[PmfgMergeParent]:
    """A fresh subclass re-declaring 'a' with a different default; function-local, same reason."""

    class PmfgMergeChild2Local(PmfgMergeParent):
        PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
            PMFG_A_KEY: PropertySpec("a, redeclared.", default="child2_a_default"),
        }

    return PmfgMergeChild2Local


class PmfgUndeclared(FeatureChainParserMixin, FeatureGroup):
    """Carries a PREFIX_PATTERN but declares no PROPERTY_MAPPING of its own.

    Safe at module level: with no property mapping the config path never activates, so this never
    matches an unrelated name (verified by the test below, which pins exactly that).
    """

    PREFIX_PATTERN = r".*__(?P<pmfg_undeclared_key>\w+)_pmfg_undeclared_op$"


class TestFeatureGroupDeclaresTheFeatureGroupSurface:
    """Item 1: FeatureGroup carries the FEATURE_GROUP surface and an empty-dict, not None, default."""

    def test_feature_group_surface_is_feature_group(self) -> None:
        assert FeatureGroup.PROPERTY_MAPPING_SURFACE is DeclarationSurface.FEATURE_GROUP

    def test_feature_group_property_mapping_default_is_an_empty_dict(self) -> None:
        assert isinstance(FeatureGroup.PROPERTY_MAPPING, dict)
        assert FeatureGroup.PROPERTY_MAPPING == {}


class TestDeclaredOptionMerge:
    """Item 2: MRO merge, most-derived winning; a child's declaration never leaks into its parent."""

    def test_child_declared_option_keys_includes_every_inherited_key(self) -> None:
        child = _make_merge_child()
        result = child.declared_option_keys()
        del child

        assert result == {PMFG_A_KEY, PMFG_B_KEY, PMFG_C_KEY}

    def test_child_declared_option_specs_returns_every_inherited_spec(self) -> None:
        child = _make_merge_child()
        try:
            # hasattr, not a direct call: an AttributeError on the class itself would pin it via the
            # exception's own `.obj` attribute, which outlives this test's teardown gc.collect() (#845).
            has_method = hasattr(child, "declared_option_specs")
            specs = child.declared_option_specs() if has_method else {}
        finally:
            del child

        assert has_method, "declared_option_specs is not implemented yet"
        assert set(specs) == {PMFG_A_KEY, PMFG_B_KEY, PMFG_C_KEY}
        assert specs[PMFG_A_KEY].default == "parent_a_default"
        assert specs[PMFG_C_KEY].default == "child_c_default"

    def test_declared_option_specs_returns_a_fresh_copy_each_call(self) -> None:
        child = _make_merge_child()
        try:
            has_method = hasattr(child, "declared_option_specs")
            if has_method:
                first = child.declared_option_specs()
                first["pmfg_injected"] = PropertySpec("injected", default=None)
                second = child.declared_option_specs()
            else:
                second = {}
        finally:
            del child

        assert has_method, "declared_option_specs is not implemented yet"
        assert "pmfg_injected" not in second

    def test_parent_declared_option_keys_excludes_the_child_only_key(self) -> None:
        assert PmfgMergeParent.declared_option_keys() == {PMFG_A_KEY, PMFG_B_KEY}

    def test_a_second_child_redeclaring_a_key_sees_its_own_spec(self) -> None:
        child2 = _make_merge_child2()
        try:
            has_method = hasattr(child2, "declared_option_specs")
            specs = child2.declared_option_specs() if has_method else {}
        finally:
            del child2

        assert has_method, "declared_option_specs is not implemented yet"
        assert specs[PMFG_A_KEY].default == "child2_a_default"

    def test_child_own_property_mapping_attribute_is_only_its_own_declaration(self) -> None:
        """The merge lives in the accessors; PROPERTY_MAPPING itself stays the plain declaration."""
        child = _make_merge_child()
        own_keys = set(child.__dict__["PROPERTY_MAPPING"])
        del child

        assert own_keys == {PMFG_C_KEY}


class TestMaterializationFollowsTheMerge:
    """Item 3: options_with_defaults and GlobalFilter._intake_fill read the merged view."""

    def test_options_with_defaults_fills_the_inherited_default(self) -> None:
        child = _make_merge_child()
        materialized = child.options_with_defaults(Options())
        del child

        assert materialized.get(PMFG_A_KEY) == "parent_a_default"

    def test_global_filter_intake_fill_reads_the_inherited_default(self) -> None:
        child = _make_merge_child()
        result = GlobalFilter._intake_fill(child, PMFG_A_KEY, Options())
        del child

        assert result == "parent_a_default"


class TestConfigPathFollowsTheMergeAndDeclaredBeyondBase:
    """Item 4: config-path matching consults the merged view and the "declared beyond base" rule."""

    def test_undeclared_class_with_prefix_pattern_does_not_match_an_unrelated_config_path_name(self) -> None:
        """Unchanged from today: an undeclared class stays out of scope on the config path."""
        result = PmfgUndeclared.match_feature_group_criteria("pmfg_undeclared_unrelated_probe", Options())

        assert result is False

    def test_child_config_path_non_match_when_the_inherited_required_key_is_absent(self) -> None:
        child = _make_merge_child()
        result = child.match_feature_group_criteria("pmfg_merge_child_probe", Options())
        del child

        assert result is False

    def test_child_config_path_matches_when_the_inherited_required_key_is_supplied(self) -> None:
        child = _make_merge_child()
        options = Options({PMFG_B_KEY: "present_value"})
        result = child.match_feature_group_criteria("pmfg_merge_child_probe", options)
        del child

        assert result is True


# --- Item 5: guards read the merged view at call time. ---

PMFG_RW_TRIGGER_KEY = "pmfg_rw_trigger"
PMFG_RW_TARGET_KEY = "pmfg_rw_target"
PMFG_RW_CHILD_KEY = "pmfg_rw_child_only"


def _pmfg_rw_trigger_present(options: Options) -> bool:
    """required_when predicate: the target key is required whenever the trigger key is present."""
    return options.get(PMFG_RW_TRIGGER_KEY) is not None


def _make_required_when_child() -> type[FeatureGroup]:
    """A fresh Parent+Child pair, function-local: with empty options BOTH are universal matchers
    today (the target's declared ``default=None`` makes it config-path skippable regardless of the
    predicate), so neither may live at module level (see module docstring)."""

    class PmfgRequiredWhenParentLocal(FeatureChainParserMixin, FeatureGroup):
        PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
            PMFG_RW_TRIGGER_KEY: PropertySpec("trigger", default=None),
            PMFG_RW_TARGET_KEY: PropertySpec(
                "target, required when trigger is present.",
                default=None,
                required_when=_pmfg_rw_trigger_present,
            ),
        }

    class PmfgRequiredWhenChildLocal(PmfgRequiredWhenParentLocal):
        PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
            PMFG_RW_CHILD_KEY: PropertySpec("child only.", default=None),
        }

    return PmfgRequiredWhenChildLocal


class TestGuardsReadTheMergedViewAtCallTime:
    """Item 5: the required_when guard, and the name-path presence guard, see inherited keys."""

    def test_required_when_guard_non_match_when_the_conditionally_required_key_is_absent(self) -> None:
        child = _make_required_when_child()
        options = Options({PMFG_RW_TRIGGER_KEY: "on"})
        result = child.match_feature_group_criteria("pmfg_rw_probe", options)
        del child

        assert result is False

    def test_required_when_guard_matches_when_the_conditionally_required_key_is_present(self) -> None:
        child = _make_required_when_child()
        options = Options({PMFG_RW_TRIGGER_KEY: "on", PMFG_RW_TARGET_KEY: "value"})
        result = child.match_feature_group_criteria("pmfg_rw_probe", options)
        del child

        assert result is True

    def test_name_path_presence_guard_flags_the_missing_inherited_no_default_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A merge child's name-carried match still owes the inherited, uncaptured 'b' key."""
        feature_name = "pmfg_source__demo_pmfg_merge_op"
        options = Options({PMFG_C_KEY: "explicit_c"})
        child = _make_merge_child()
        try:
            with caplog.at_level(logging.WARNING):
                result = child.match_feature_group_criteria(feature_name, options)
            messages = [record.getMessage() for record in caplog.records]
        finally:
            del child

        assert result is False
        assert any(PMFG_B_KEY in message for message in messages)


# --- Items 6-7: author-time rejections on the FeatureGroup surface. ---


class TestExplicitNoneOrNonDictPropertyMappingRaisesAtClassDefinition:
    """Item 6: PROPERTY_MAPPING = None or a non-dict raises ValueError at class-definition time."""

    def test_property_mapping_none_raises_naming_the_class_with_none_and_merge(self) -> None:
        with pytest.raises(ValueError) as exc_info:

            class PmfgNoneDecl(FeatureGroup):
                PROPERTY_MAPPING = None  # type: ignore[assignment]

        message = str(exc_info.value)
        del exc_info
        assert "PmfgNoneDecl" in message
        assert "None" in message
        assert "merge" in message

    def test_property_mapping_non_dict_raises_naming_the_class(self) -> None:
        """Widened to (ValueError, AttributeError): today's ``.items()`` call on a list raises
        AttributeError, not the target ValueError; the type assertion below fails for that reason."""
        with pytest.raises((ValueError, AttributeError)) as exc_info:

            class PmfgListDecl(FeatureGroup):
                PROPERTY_MAPPING = ["x"]  # type: ignore[assignment]

        error_type = type(exc_info.value)
        message = str(exc_info.value)
        del exc_info
        assert error_type is ValueError, f"expected ValueError, got {error_type.__name__}: {message}"
        assert "PmfgListDecl" in message


class TestMergeCacheAssignmentRaises:
    """Item 7: the merge cache is framework-written; declaring it in a class body is rejected."""

    def test_property_mapping_cache_assignment_raises(self) -> None:
        with pytest.raises(ValueError):

            class PmfgCacheAssign(FeatureGroup):
                _property_mapping_cache: ClassVar[dict[str, Any]] = {}


# --- Item 8: plain-mixin declarations. ---

PMFG_MIX_KEY = "pmfg_mix_key"


class PmfgMix:
    """A plain (non-FeatureGroup) mixin declaring one key."""

    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        PMFG_MIX_KEY: PropertySpec("m", default=1),
    }


class PmfgMixed(PmfgMix, FeatureGroup):
    """Mixes in PmfgMix; declares nothing of its own.

    Safe at module level: no FeatureChainParserMixin, so matching falls back to FeatureGroup's
    plain class-name check, never the config path.
    """


class TestPlainMixinDeclarations:
    """Item 8: a plain mixin's declaration merges in; a reader-only field on it still raises."""

    def test_declared_option_keys_includes_a_plain_mixin_key(self) -> None:
        assert PMFG_MIX_KEY in PmfgMixed.declared_option_keys()

    def test_mixin_framework_set_spec_raises_reaching_the_feature_group_subclass(self) -> None:
        class PmfgFrameworkSetMixin:
            PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
                "pmfg_fw_key": PropertySpec("fw", framework_set=True, default=None),
            }

        with pytest.raises(ValueError) as exc_info:

            class PmfgFrameworkSetMixed(PmfgFrameworkSetMixin, FeatureGroup):
                pass

        message = str(exc_info.value)
        del exc_info
        assert "PmfgFrameworkSetMixin" in message
        assert "framework_set" in message
        assert "reached defining PmfgFrameworkSetMixed" in message


# --- Item 9: mixin-only classes (no FeatureGroup base) keep working. ---

PMFG_ONLY_MIXIN_KEY = "pmfg_only_mixin_key"


class PmfgOnlyMixin(FeatureChainParserMixin):
    """A FeatureChainParserMixin subclass with no FeatureGroup base.

    Safe at module level: it is not a FeatureGroup subclass at all, so it never appears in a
    FeatureGroup resolution scan elsewhere.
    """

    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        PMFG_ONLY_MIXIN_KEY: PropertySpec("m", default="only_default"),
    }


class PmfgNoDeclMixin(FeatureChainParserMixin):
    """A FeatureChainParserMixin subclass declaring nothing at all; same safety as PmfgOnlyMixin."""


class TestMixinOnlyClassesKeepWorking:
    """Item 9: a bare FeatureChainParserMixin subclass, with or without a declaration."""

    def test_mixin_only_class_returns_its_own_mapping(self) -> None:
        mapping = PmfgOnlyMixin._get_property_mapping()

        assert mapping is not None
        assert PMFG_ONLY_MIXIN_KEY in mapping

    def test_mixin_only_class_matches_on_the_config_path(self) -> None:
        result = PmfgOnlyMixin.match_feature_group_criteria("pmfg_only_mixin_probe", Options())

        assert result is True

    def test_mixin_subclass_without_a_declaration_returns_none(self) -> None:
        assert PmfgNoDeclMixin._get_property_mapping() is None
