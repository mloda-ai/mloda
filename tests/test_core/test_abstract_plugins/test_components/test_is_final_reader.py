"""Structural is_final_reader contract on BaseInputData.

- BaseInputData._final_reader_requires() -> tuple[str, ...], default ().
- BaseInputData.final_reader_anchor() -> type; public anchor resolution.
- BaseInputData.is_final_reader() -> bool. Replaces the removed execution-probe
  supports_scoped_data_access.
- Anchor = most-derived class in cls.__mro__ that declares _final_reader_requires
  in its own __dict__ (BaseInputData declares the default, so an anchor always exists).
- If load_data is overridden relative to the anchor: True. Otherwise True iff the
  required tuple is non-empty and every named hook is overridden relative to the anchor.
- A required name missing on the anchor raises ValueError naming the hook and the anchor.
- Classification is purely structural: it never executes load_data or any hook.

The synthetic families below are built directly on BaseInputData and expose no
matching surface (match_subclass_data_access returns None, and their
data_access_name is just the class name, which no sibling test uses as an
options key), so they cannot pollute discovery in other tests.
"""

import gc
import weakref
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.provider import FeatureSet


def _invoke(cls: type[BaseInputData], method_name: str) -> Any:
    """Invoke a classmethod on cls by name."""
    return getattr(cls, method_name)()


def _is_final_reader(cls: type[BaseInputData]) -> bool:
    result = _invoke(cls, "is_final_reader")
    assert isinstance(result, bool)
    return result


def _final_reader_requires(cls: type[BaseInputData]) -> tuple[str, ...]:
    result = _invoke(cls, "_final_reader_requires")
    assert isinstance(result, tuple)
    return result


def _final_reader_anchor(cls: type[BaseInputData]) -> type:
    result = _invoke(cls, "final_reader_anchor")
    assert isinstance(result, type)
    return result


class _DefaultPlain(BaseInputData):
    """Default family (no _final_reader_requires declaration), no load_data override."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
        return None


class _DefaultWholesale(BaseInputData):
    """Default family, wholesale load_data override."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"unused_synthetic_column": [1]}

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
        return None


class _DefaultBareRedeclaration(BaseInputData):
    """Default family, load_data re-declared with a bare NotImplementedError body.

    Structural classification means declared-is-overridden: this pins the documented
    convention that intermediate bases must not re-declare bare hooks.
    """

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        raise NotImplementedError

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
        return None


class _FamilyBase(BaseInputData):
    """Synthetic seam family requiring hook_a and hook_b."""

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        return ("hook_a", "hook_b")

    @classmethod
    def hook_a(cls) -> Any:
        raise NotImplementedError

    @classmethod
    def hook_b(cls) -> Any:
        raise NotImplementedError

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        cls.hook_a()
        return cls.hook_b()

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
        return None


class _OnlyHookA(_FamilyBase):
    @classmethod
    def hook_a(cls) -> Any:
        return "a"


class _BothHooks(_FamilyBase):
    @classmethod
    def hook_a(cls) -> Any:
        return "a"

    @classmethod
    def hook_b(cls) -> Any:
        return "b"


class _WholesaleInFamily(_FamilyBase):
    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"unused_synthetic_column": [2]}


_RECORDED_CALLS: list[str] = []


class _RecordingHooksOnly(_FamilyBase):
    """Final via hooks; records every execution so structural classification is provable."""

    @classmethod
    def hook_a(cls) -> Any:
        _RECORDED_CALLS.append("hook_a")
        return None

    @classmethod
    def hook_b(cls) -> Any:
        _RECORDED_CALLS.append("hook_b")
        return None


class _RecordingWholesale(_FamilyBase):
    """Final via wholesale load_data; records every execution."""

    @classmethod
    def hook_a(cls) -> Any:
        _RECORDED_CALLS.append("hook_a_wholesale")
        return None

    @classmethod
    def hook_b(cls) -> Any:
        _RECORDED_CALLS.append("hook_b_wholesale")
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        _RECORDED_CALLS.append("load_data_wholesale")
        return None


class _SubFamilyBase(_FamilyBase):
    """Sub-family redeclaring the anchor: requires hook_c instead of hook_a/hook_b."""

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        return ("hook_c",)

    @classmethod
    def hook_c(cls) -> Any:
        raise NotImplementedError


class _SubFamilyChild(_SubFamilyBase):
    @classmethod
    def hook_c(cls) -> Any:
        return "c"


class _SubFamilyWholesale(_SubFamilyBase):
    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"unused_synthetic_column": [3]}


class _AnchorWithOwnLoadData(_FamilyBase):
    """Redeclares the anchor and defines its own load_data body in the same class."""

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        return ("hook_c",)

    @classmethod
    def hook_c(cls) -> Any:
        raise NotImplementedError

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        cls.hook_a()
        return cls.hook_c()


class TestOldNameRemoved:
    def test_supports_scoped_data_access_is_gone_from_base(self) -> None:
        assert not hasattr(BaseInputData, "supports_scoped_data_access")


class TestDefaultFamilyClassification:
    def test_base_default_requires_is_empty(self) -> None:
        assert _final_reader_requires(BaseInputData) == ()

    def test_plain_subclass_is_not_final(self) -> None:
        assert _is_final_reader(_DefaultPlain) is False

    def test_wholesale_load_data_is_final(self) -> None:
        assert _is_final_reader(_DefaultWholesale) is True

    def test_bare_redeclaration_counts_as_overridden(self) -> None:
        assert _is_final_reader(_DefaultBareRedeclaration) is True


class TestSyntheticSeamFamily:
    def test_family_base_is_not_final(self) -> None:
        assert _is_final_reader(_FamilyBase) is False

    def test_partial_hooks_are_not_final(self) -> None:
        assert _is_final_reader(_OnlyHookA) is False

    def test_all_hooks_are_final(self) -> None:
        assert _is_final_reader(_BothHooks) is True

    def test_wholesale_load_data_in_family_is_final(self) -> None:
        assert _is_final_reader(_WholesaleInFamily) is True

    def test_classification_never_executes_hooks_or_load_data(self) -> None:
        _RECORDED_CALLS.clear()
        assert _is_final_reader(_RecordingHooksOnly) is True
        assert _is_final_reader(_RecordingWholesale) is True
        assert _RECORDED_CALLS == []


class TestAnchorRedeclaration:
    def test_sub_family_base_is_not_final(self) -> None:
        assert _is_final_reader(_SubFamilyBase) is False

    def test_child_overriding_new_hook_is_final(self) -> None:
        assert _is_final_reader(_SubFamilyChild) is True

    def test_child_overriding_load_data_relative_to_new_anchor_is_final(self) -> None:
        assert _is_final_reader(_SubFamilyWholesale) is True

    def test_anchor_redeclaring_requires_with_own_load_data_is_not_final(self) -> None:
        """Pins the semantics: declaring a requires tuple makes a class a family base,
        and a family base is never final even with its own load_data; a concrete
        reader must not redeclare the tuple."""
        assert _is_final_reader(_AnchorWithOwnLoadData) is False


class TestAnchorResolution:
    """final_reader_anchor() is the public seam a family builds its hook seam on."""

    def test_base_is_its_own_anchor(self) -> None:
        assert _final_reader_anchor(BaseInputData) is BaseInputData

    def test_default_family_anchors_to_base(self) -> None:
        assert _final_reader_anchor(_DefaultPlain) is BaseInputData
        assert _final_reader_anchor(_DefaultWholesale) is BaseInputData

    def test_family_base_is_its_own_anchor(self) -> None:
        assert _final_reader_anchor(_FamilyBase) is _FamilyBase

    def test_family_members_anchor_to_family_base(self) -> None:
        assert _final_reader_anchor(_OnlyHookA) is _FamilyBase
        assert _final_reader_anchor(_BothHooks) is _FamilyBase
        assert _final_reader_anchor(_WholesaleInFamily) is _FamilyBase

    def test_sub_family_redeclaration_moves_the_anchor(self) -> None:
        assert _final_reader_anchor(_SubFamilyBase) is _SubFamilyBase
        assert _final_reader_anchor(_SubFamilyChild) is _SubFamilyBase
        assert _final_reader_anchor(_SubFamilyWholesale) is _SubFamilyBase


class TestLoudValidation:
    def test_missing_required_hook_raises_value_error(self) -> None:
        class _BadFamily(BaseInputData):
            @classmethod
            def _final_reader_requires(cls) -> tuple[str, ...]:
                return ("nonexistent_hook",)

            @classmethod
            def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
                return None

        bad_family = weakref.ref(_BadFamily)

        with pytest.raises(ValueError) as exc_info:
            _invoke(_BadFamily, "is_final_reader")
        assert "nonexistent_hook" in str(exc_info.value)
        assert "_BadFamily" in str(exc_info.value)
        # Drop the misconfigured local class from BaseInputData.__subclasses__ so it
        # cannot raise during discovery in sibling tests running in the same worker.
        # The collect must happen after dropping exc_info because its captured
        # traceback frames pin the class; the weakref assert fails loudly if the
        # class ever becomes uncollectable again.
        del exc_info
        del _BadFamily
        gc.collect()
        assert bad_family() is None


class TestFamilyDeclarations:
    """A family declares a hook seam and a template load_data without defining the classification itself."""

    def test_family_requires_tuple(self) -> None:
        assert _final_reader_requires(_FamilyBase) == ("hook_a", "hook_b")

    def test_family_declares_the_anchor_marker_locally(self) -> None:
        assert "_final_reader_requires" in _FamilyBase.__dict__

    def test_family_declares_template_load_data_yet_is_not_final(self) -> None:
        assert "load_data" in _FamilyBase.__dict__
        assert _is_final_reader(_FamilyBase) is False

    def test_family_does_not_define_classification_locally(self) -> None:
        assert "supports_scoped_data_access" not in _FamilyBase.__dict__
        assert "is_final_reader" not in _FamilyBase.__dict__
        assert "final_reader_anchor" not in _FamilyBase.__dict__
