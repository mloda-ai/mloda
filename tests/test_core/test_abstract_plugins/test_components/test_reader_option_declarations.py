"""Pins the collapsed ``READER_OPTIONS`` surface on ``BaseInputData`` (issue #949): one spec type,
``PropertySpec`` plus ``framework_set``, read via ``reader_option`` / ``reader_option_default``.
Leak policy: the throwaway readers here leak deliberately; none is final, so selection never collects them.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from functools import cache
from typing import Any, ClassVar

import pytest

import mloda.provider as mloda_provider
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, is_positive_int
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


def _spec(*args: Any, **kwargs: Any) -> PropertySpec:
    """Build a ``PropertySpec`` through an untyped seam so type-invalid calls stay runtime tests."""
    return PropertySpec(*args, **kwargs)


def _rod_never_required(options: Options) -> bool:
    """A required_when predicate; only its presence on a spec matters here."""
    return False


def _rod_match_guard(value: Any) -> bool:
    """A match_guard predicate; only its presence on a spec matters here."""
    return True


@cache
def _decl_family() -> tuple[type[BaseInputData], type[BaseInputData], type[BaseInputData]]:
    """(parent, child, override): the shared declaring family; lazy so a broken guard fails users, not collection."""

    class RodDeclParentReader(BaseInputData):
        """Parent declaring key A only; inherits the reserved framework key."""

        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            "rod_key_a": PropertySpec("Key A, declared on the parent.", default="parent_a"),
        }

        @classmethod
        def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
            return None

    class RodDeclChildReader(RodDeclParentReader):
        """Child declaring key B only; A and the reserved key must still be visible."""

        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            "rod_key_b": PropertySpec("Key B, declared on the child.", default="child_b"),
        }

    class RodDeclOverrideReader(RodDeclParentReader):
        """Child redeclaring key A with a different default; the most-derived declaration wins."""

        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            "rod_key_a": PropertySpec("Key A, redeclared on the child.", default="child_a"),
        }

    return RodDeclParentReader, RodDeclChildReader, RodDeclOverrideReader


class TestPropertySpecFrameworkSetField:
    """The one spec type carries the reader-surface ``framework_set`` flag."""

    def test_framework_set_defaults_to_false(self) -> None:
        """A plain spec is user-set: ``framework_set`` is False unless declared."""
        assert PropertySpec("x").framework_set is False

    def test_framework_set_true_constructs(self) -> None:
        """Marking a spec framework-written is a plain field, not a construction error."""
        assert PropertySpec("x", default=None, framework_set=True).framework_set is True

    def test_framework_set_is_frozen(self) -> None:
        """Assigning the field raises ``dataclasses.FrozenInstanceError`` like every other field."""
        spec = PropertySpec("x", default=None, framework_set=True)

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "framework_set", False)

    def test_framework_set_participates_in_value_equality(self) -> None:
        """Two specs differing only in ``framework_set`` are unequal; equal fields stay equal."""
        assert PropertySpec("x", default=None, framework_set=True) != PropertySpec("x", default=None)
        assert PropertySpec("x", default=None, framework_set=True) == PropertySpec(
            "x", default=None, framework_set=True
        )

    @pytest.mark.parametrize("non_bool", ["yes", 1, None])
    def test_non_bool_framework_set_rejected_at_construction(self, non_bool: Any) -> None:
        """A non-bool is a ``ValueError`` naming the field, like the other bool fields."""
        with pytest.raises(ValueError) as exc_info:
            _spec("x", framework_set=non_bool)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message
        assert "bool" in message


class TestTheOldSpecTypeIsGone:
    """The two-type world is over: the ``ReaderOptionSpec`` module and export no longer exist."""

    def test_the_reader_option_spec_module_is_removed(self) -> None:
        """Importing the old module path raises ``ModuleNotFoundError``."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mloda.core.abstract_plugins.components.input_data.reader_option_spec")

    def test_reader_option_spec_is_not_in_the_provider_all(self) -> None:
        """The provider surface no longer lists the old type."""
        assert "ReaderOptionSpec" not in mloda_provider.__all__

    def test_importing_reader_option_spec_from_the_provider_raises(self) -> None:
        """``from mloda.provider import ReaderOptionSpec`` is an ``ImportError``, not a survivor."""
        with pytest.raises(ImportError):
            from mloda.provider import ReaderOptionSpec  # type: ignore[attr-defined,unused-ignore]  # noqa: F401


class TestReservedFrameworkKey:
    """``BaseInputData`` itself declares the reserved ``"BaseInputData"`` options key."""

    def test_base_declares_reader_options_locally(self) -> None:
        """The declaration lives on ``BaseInputData``, not only on some subclass."""
        assert "READER_OPTIONS" in BaseInputData.__dict__
        assert isinstance(BaseInputData.READER_OPTIONS, dict)
        assert "BaseInputData" in BaseInputData.READER_OPTIONS

    def test_declared_keys_contain_the_reserved_key(self) -> None:
        """``declared_reader_option_keys()`` on the base contains the reserved key."""
        keys = BaseInputData.declared_reader_option_keys()

        assert isinstance(keys, frozenset)
        assert "BaseInputData" in keys

    def test_reserved_key_is_a_framework_set_property_spec(self) -> None:
        """The reserved key is a ``PropertySpec`` written by the framework, so users never set it."""
        spec = BaseInputData.reader_option_specs()["BaseInputData"]

        assert isinstance(spec, PropertySpec)
        assert spec.framework_set is True

    def test_reserved_key_declares_default_none(self) -> None:
        """A DECLARED ``default=None``: no reader code falls back for it; init_reader raises when absent."""
        assert BaseInputData.reader_option_specs()["BaseInputData"].default is None
        assert BaseInputData.reader_option_default("BaseInputData") is None

    def test_reserved_key_does_not_declare_allow_explicit_none(self) -> None:
        """The flag is inert on a framework-written key, so the base must not declare it."""
        assert BaseInputData.reader_option_specs()["BaseInputData"].allow_explicit_none is False

    def test_base_declaration_passes_its_own_guard(self) -> None:
        """``__init_subclass__`` never validates the base itself; this pin keeps the base honest."""
        BaseInputData._validate_reader_options()

    def test_reserved_key_is_the_key_the_framework_actually_writes(self) -> None:
        """``add_base_input_data_to_options`` writes exactly the keys declared ``framework_set``.

        This is what makes ``framework_set`` load-bearing: nothing in the framework reads the flag
        (deliberately, reader selection is a hot path), so the flag's meaning is pinned here instead.
        Flipping it to False, or marking a second key ``framework_set`` that nothing writes, fails.
        """
        parent, _, _ = _decl_family()
        options = Options()
        BaseInputData.add_base_input_data_to_options(parent, "rod_reserved_access", options)

        written = set(options.keys())
        specs = BaseInputData.reader_option_specs()
        assert written == {"BaseInputData"}
        assert written <= BaseInputData.declared_reader_option_keys()
        assert written == {key for key, spec in specs.items() if spec.framework_set}
        assert specs["BaseInputData"].framework_set is True

    def test_the_declaring_reader_family_marks_no_further_framework_keys(self) -> None:
        """A reader family inherits the one framework-written key and adds no second one.

        The flag marks keys USERS never set, so a reader declaring its own user-facing keys must
        leave them ``framework_set=False``; otherwise the invariant above stops describing writes.
        """
        _, child, _ = _decl_family()
        specs = child.reader_option_specs()

        assert {key for key, spec in specs.items() if spec.framework_set} == {"BaseInputData"}
        assert specs["rod_key_a"].framework_set is False
        assert specs["rod_key_b"].framework_set is False


class TestMroMerge:
    """Declarations merge across ``cls.__mro__``, most-derived class winning on a collision."""

    def test_parent_keys_include_own_and_inherited(self) -> None:
        """The parent sees its own key A plus the reserved key from the base."""
        parent, _, _ = _decl_family()

        assert parent.declared_reader_option_keys() == {"rod_key_a", "BaseInputData"}

    def test_child_merges_parent_and_own_keys(self) -> None:
        """A child declaring only B still sees A and the reserved key."""
        _, child, _ = _decl_family()

        assert child.declared_reader_option_keys() == {"rod_key_a", "rod_key_b", "BaseInputData"}

    def test_child_declaration_does_not_leak_into_the_parent(self) -> None:
        """The merge walks up the MRO only; the parent never gains the child's key."""
        parent, _, _ = _decl_family()

        assert "rod_key_b" not in parent.declared_reader_option_keys()

    def test_specs_are_returned_keyed_by_option_name(self) -> None:
        """``reader_option_specs()`` maps every merged key to its ``PropertySpec``."""
        _, child, _ = _decl_family()
        specs = child.reader_option_specs()

        assert set(specs) == child.declared_reader_option_keys()
        assert all(isinstance(spec, PropertySpec) for spec in specs.values())
        assert specs["rod_key_a"].default == "parent_a"
        assert specs["rod_key_b"].default == "child_b"

    def test_most_derived_declaration_wins_on_a_key_collision(self) -> None:
        """Redeclaring key A with another default overrides the parent's declaration."""
        parent, _, override = _decl_family()

        assert override.reader_option_default("rod_key_a") == "child_a"
        assert override.reader_option_specs()["rod_key_a"].default == "child_a"
        assert parent.reader_option_default("rod_key_a") == "parent_a"

    def test_returned_mapping_is_a_fresh_copy(self) -> None:
        """A caller mutating the merged mapping cannot corrupt the class declarations."""
        _, child, _ = _decl_family()
        specs = child.reader_option_specs()
        specs["rod_key_injected"] = PropertySpec("Injected by a caller.", default=None)

        assert "rod_key_injected" not in child.declared_reader_option_keys()
        assert "rod_key_injected" not in child.reader_option_specs()


class TestReaderOptionDefault:
    """``reader_option_default`` returns the declared fallback and is loud about typos."""

    def test_declared_default_is_returned(self) -> None:
        """The declared ``default`` is what reader code gets for an absent key."""
        parent, child, _ = _decl_family()

        assert parent.reader_option_default("rod_key_a") == "parent_a"
        assert child.reader_option_default("rod_key_b") == "child_b"

    def test_inherited_default_is_returned(self) -> None:
        """A child needs no re-declaration to reach the parent's default."""
        _, child, _ = _decl_family()

        assert child.reader_option_default("rod_key_a") == "parent_a"

    def test_undeclared_key_raises_value_error_naming_key_and_class(self) -> None:
        """A typo in reader code is loud, not a silent None."""
        _, child, _ = _decl_family()

        with pytest.raises(ValueError) as exc_info:
            child.reader_option_default("not_a_key")

        message = str(exc_info.value)
        del exc_info
        assert "not_a_key" in message
        assert child.__name__ in message

    def test_undeclared_key_on_base_raises_value_error(self) -> None:
        """The same guard holds on ``BaseInputData`` itself."""
        with pytest.raises(ValueError, match="not_a_key"):
            BaseInputData.reader_option_default("not_a_key")


class TestReaderOptionHonoursPresence:
    """``reader_option(key, options)`` on a flagless spec reads presence, not truthiness."""

    def test_signature_is_key_first_options_second(self) -> None:
        """The KEY comes first, mirroring ``reader_option_default(key)``; the Options is second."""
        parameters = list(inspect.signature(BaseInputData.reader_option).parameters)

        assert parameters == ["key", "options"]

    def test_absent_key_falls_back_to_the_declared_default(self) -> None:
        """Nothing supplied means the reader's own declared fallback applies."""
        parent, _, _ = _decl_family()

        assert parent.reader_option("rod_key_a", Options()) == "parent_a"

    def test_supplied_value_wins_over_the_declared_default(self) -> None:
        """A user-set value is returned unchanged."""
        parent, _, _ = _decl_family()
        options = Options({"rod_key_a": "supplied"})

        assert parent.reader_option("rod_key_a", options) == "supplied"

    @pytest.mark.parametrize("falsy_value", [frozenset(), (), [], "", 0, False, {}])
    def test_present_but_falsy_value_is_honoured_not_replaced(self, falsy_value: Any) -> None:
        """An explicit empty value means "hand nothing over" and must survive the read."""
        parent, _, _ = _decl_family()
        options = Options({"rod_key_a": falsy_value})

        result = parent.reader_option("rod_key_a", options)

        assert result == falsy_value
        assert result != "parent_a"

    def test_explicit_none_reads_as_absent_on_a_flagless_spec(self) -> None:
        """Without ``allow_explicit_none``, ``None`` is the framework's dominant absence marker."""
        parent, _, _ = _decl_family()
        options = Options({"rod_key_a": None})

        assert "rod_key_a" in options
        assert parent.reader_option("rod_key_a", options) == "parent_a"

    def test_a_context_option_is_read_like_a_group_option(self) -> None:
        """The accessor reads through ``Options.get``, so the category never changes the answer."""
        parent, _, _ = _decl_family()
        options = Options(context={"rod_key_a": frozenset()})

        assert parent.reader_option("rod_key_a", options) == frozenset()

    def test_inherited_declaration_supplies_the_default(self) -> None:
        """A child needs no re-declaration to reach the parent's fallback."""
        _, child, _ = _decl_family()

        assert child.reader_option("rod_key_a", Options()) == "parent_a"

    def test_most_derived_declaration_supplies_the_default(self) -> None:
        """A redeclared key resolves to the most-derived ``default``, like the sibling accessor."""
        parent, _, override = _decl_family()

        assert override.reader_option("rod_key_a", Options()) == "child_a"
        assert parent.reader_option("rod_key_a", Options()) == "parent_a"

    def test_undeclared_key_raises_value_error_naming_key_and_class(self) -> None:
        """A typo is loud here exactly as in ``reader_option_default``, not a silent None."""
        _, child, _ = _decl_family()

        with pytest.raises(ValueError) as exc_info:
            child.reader_option("not_a_key", Options())

        message = str(exc_info.value)
        del exc_info
        assert "not_a_key" in message
        assert child.__name__ in message

    def test_undeclared_key_raises_even_when_a_value_is_supplied(self) -> None:
        """The declaration gates the read: a supplied value cannot legitimize an undeclared key."""
        _, child, _ = _decl_family()
        options = Options({"not_a_key": "supplied"})

        with pytest.raises(ValueError, match="not_a_key"):
            child.reader_option("not_a_key", options)

    def test_undeclared_key_on_the_base_raises(self) -> None:
        """The same guard holds on ``BaseInputData`` itself."""
        with pytest.raises(ValueError, match="not_a_key"):
            BaseInputData.reader_option("not_a_key", Options())


class TestAllowExplicitNoneIsHonoured:
    """An ``allow_explicit_none=True`` spec reads presence as ``key in options`` (#768 matrix)."""

    def _reader(self) -> type[BaseInputData]:
        """A fresh reader whose one key opts into explicit None."""

        class RodOptInNoneReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_opt_in_none_key": PropertySpec(
                    "Opts into explicit None.", default="declared", allow_explicit_none=True
                ),
            }

        return RodOptInNoneReader

    def test_explicit_none_is_returned_not_replaced(self) -> None:
        """An opted-in explicit ``None`` is a VALUE: the declared default must not swallow it."""
        reader = self._reader()
        options = Options({"rod_opt_in_none_key": None})

        assert "rod_opt_in_none_key" in options
        assert reader.reader_option("rod_opt_in_none_key", options) is None

    def test_absent_key_falls_back_to_the_declared_default(self) -> None:
        """A truly absent key still reads the declared fallback."""
        reader = self._reader()

        assert reader.reader_option("rod_opt_in_none_key", Options()) == "declared"

    def test_supplied_value_still_wins(self) -> None:
        """The flag widens presence; it never changes what a present non-None value returns."""
        reader = self._reader()
        options = Options({"rod_opt_in_none_key": "supplied"})

        assert reader.reader_option("rod_opt_in_none_key", options) == "supplied"


class TestNoDefaultMakesTheKeyRequired:
    """A spec declaring no default (``NO_DEFAULT``) has nothing to fall back to."""

    def _reader(self) -> type[BaseInputData]:
        """A fresh reader whose one key declares no default."""

        class RodRequiredKeyReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_required_key": PropertySpec("Required at read time; no declared fallback."),
            }

        return RodRequiredKeyReader

    def test_reader_option_raises_for_the_absent_key_naming_key_and_class(self) -> None:
        """Absent plus no declared default is a loud ``ValueError``, not a silent sentinel leak."""
        reader = self._reader()

        with pytest.raises(ValueError) as exc_info:
            reader.reader_option("rod_required_key", Options())

        message = str(exc_info.value)
        del exc_info
        assert "rod_required_key" in message
        assert reader.__name__ in message

    def test_reader_option_returns_the_supplied_value(self) -> None:
        """With the key present, the supplied value is returned unchanged."""
        reader = self._reader()
        options = Options({"rod_required_key": "supplied"})

        assert reader.reader_option("rod_required_key", options) == "supplied"

    def test_reader_option_default_raises_naming_key_and_class(self) -> None:
        """Asking for the default of a key that declares none is the same loud ``ValueError``."""
        reader = self._reader()

        with pytest.raises(ValueError) as exc_info:
            reader.reader_option_default("rod_required_key")

        message = str(exc_info.value)
        del exc_info
        assert "rod_required_key" in message
        assert reader.__name__ in message


class TestReaderOptionToleratesNoneOptions:
    """``reader_option(key, None)`` mirrors the selection seam: no Options reads as every key absent."""

    def test_none_options_return_the_declared_default(self) -> None:
        """With no Options at all, the declared fallback applies exactly as for an absent key."""
        parent, _, _ = _decl_family()
        none_options: Any = None  # the accessor's contract widens to Optional[Options]

        assert parent.reader_option("rod_key_a", none_options) == "parent_a"

    def test_none_options_on_an_opted_in_spec_read_absent(self) -> None:
        """The ``allow_explicit_none`` presence read (``key in options``) must tolerate None too."""

        class RodNoneOptionsOptInReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_none_opt_in_key": PropertySpec(
                    "Opts into explicit None.", default="declared", allow_explicit_none=True
                ),
            }

        none_options: Any = None

        assert RodNoneOptionsOptInReader.reader_option("rod_none_opt_in_key", none_options) == "declared"

    def test_none_options_on_a_no_default_spec_raise_the_absent_key_error(self) -> None:
        """A NO_DEFAULT key with no Options raises EXACTLY the absent-key ValueError."""

        class RodNoneOptionsRequiredReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_none_required_key": PropertySpec("Required at read time; no declared fallback."),
            }

        none_options: Any = None
        with pytest.raises(ValueError) as none_exc:
            RodNoneOptionsRequiredReader.reader_option("rod_none_required_key", none_options)
        none_message = str(none_exc.value)
        del none_exc

        with pytest.raises(ValueError) as absent_exc:
            RodNoneOptionsRequiredReader.reader_option("rod_none_required_key", Options())
        absent_message = str(absent_exc.value)
        del absent_exc

        assert none_message == absent_message
        assert "rod_none_required_key" in none_message
        assert "RodNoneOptionsRequiredReader" in none_message


class TestMergedSpecCacheStaysFresh:
    """The merged-spec cache is per class, so it can never answer for the wrong class.

    Every family below is built INSIDE its test so the cache starts cold: module-level classes are
    warmed by whichever test the xdist worker happened to run first, which would make these vacuous.
    """

    def test_a_subclass_defined_after_a_warm_parent_cache_sees_its_own_declaration(self) -> None:
        """Warming the parent must not answer for a child that redeclares the key."""

        class RodColdParentReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Declared on the parent.", default="parent"),
            }

        assert RodColdParentReader.reader_option("rod_cache_key", Options()) == "parent"

        class RodLateChildReader(RodColdParentReader):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Redeclared on the child.", default="child"),
            }

        assert RodLateChildReader.reader_option("rod_cache_key", Options()) == "child"
        assert RodColdParentReader.reader_option("rod_cache_key", Options()) == "parent"

    def test_a_warm_child_cache_does_not_change_the_parent(self) -> None:
        """The reverse order: reading the child first leaves the parent's answer alone."""

        class RodParentReadSecond(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Declared on the parent.", default="parent"),
            }

        class RodChildReadFirst(RodParentReadSecond):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Redeclared on the child.", default="child"),
            }

        assert RodChildReadFirst.reader_option("rod_cache_key", Options()) == "child"
        assert RodParentReadSecond.reader_option("rod_cache_key", Options()) == "parent"

    def test_a_key_added_by_a_late_subclass_is_visible_after_a_warm_parent_cache(self) -> None:
        """``declared_reader_option_keys`` shares the cache, so it must not go stale either."""

        class RodKeysParentReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_parent_key": PropertySpec("Parent only.", default=None),
            }

        assert RodKeysParentReader.declared_reader_option_keys() == {"rod_cache_parent_key", "BaseInputData"}

        class RodKeysChildReader(RodKeysParentReader):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_child_key": PropertySpec("Child only.", default=None),
            }

        assert RodKeysChildReader.declared_reader_option_keys() == {
            "rod_cache_parent_key",
            "rod_cache_child_key",
            "BaseInputData",
        }
        assert "rod_cache_child_key" not in RodKeysParentReader.declared_reader_option_keys()

    def test_an_undeclared_key_still_raises_on_a_warm_cache(self) -> None:
        """The guard reads the cached mapping, so warming it cannot make a typo silent."""

        class RodWarmGuardReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Declared.", default="declared"),
            }

        assert RodWarmGuardReader.reader_option("rod_cache_key", Options()) == "declared"

        with pytest.raises(ValueError, match="not_a_key"):
            RodWarmGuardReader.reader_option("not_a_key", Options())

    def test_mutating_the_returned_specs_cannot_poison_the_cache(self) -> None:
        """The caller-facing mapping stays a fresh copy: a cache must not be handed out by reference."""

        class RodCopyProbeReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Declared.", default="declared"),
            }

        specs = RodCopyProbeReader.reader_option_specs()
        specs["rod_cache_key"] = PropertySpec("Injected.", default="injected")
        specs["rod_cache_injected_key"] = PropertySpec("Injected.", default=None)

        assert RodCopyProbeReader.reader_option("rod_cache_key", Options()) == "declared"
        assert "rod_cache_injected_key" not in RodCopyProbeReader.declared_reader_option_keys()

    def test_repeated_reads_stay_equal(self) -> None:
        """Caching is invisible: two reads of the same key on the same class agree."""

        class RodRepeatReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_cache_key": PropertySpec("Declared.", default=frozenset({".json"})),
            }

        first = RodRepeatReader.reader_option("rod_cache_key", Options())
        second = RodRepeatReader.reader_option("rod_cache_key", Options())

        assert first == second == frozenset({".json"})
        assert RodRepeatReader.reader_option_specs() == RodRepeatReader.reader_option_specs()


class TestReaderOptionsAreValidatedAtClassDefinition:
    """``READER_OPTIONS`` accepts reader-shaped ``PropertySpec`` instances and NOTHING else, rejected where written."""

    def test_string_value_rejected_at_class_definition(self) -> None:
        """``{"k": "just a string"}`` names the class, the key and the ``PropertySpec`` remedy."""
        with pytest.raises(ValueError) as exc_info:

            class RodBadStringSpecReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"rod_bad_key": "just a string"}  # type: ignore[dict-item]  # wrong type is the point

        message = str(exc_info.value)
        del exc_info
        assert "RodBadStringSpecReader" in message
        assert "rod_bad_key" in message
        assert "PropertySpec" in message

    def test_dict_value_rejected_at_class_definition(self) -> None:
        """A hand-rolled spec dict (the plausible authoring mistake) is rejected the same way."""
        with pytest.raises(ValueError) as exc_info:

            class RodBadDictSpecReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_bad_key": {"explanation": "x", "default": None}  # type: ignore[dict-item]  # wrong type is the point
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodBadDictSpecReader" in message
        assert "rod_bad_key" in message
        assert "PropertySpec" in message

    def test_int_value_rejected_at_class_definition(self) -> None:
        """A bare non-spec scalar is rejected with the same naming."""
        with pytest.raises(ValueError) as exc_info:

            class RodBadIntSpecReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"rod_bad_key": 42}  # type: ignore[dict-item]  # wrong type is the point

        message = str(exc_info.value)
        del exc_info
        assert "RodBadIntSpecReader" in message
        assert "rod_bad_key" in message
        assert "PropertySpec" in message

    def test_match_guard_spec_rejected_at_class_definition(self) -> None:
        """``match_guard`` gates FeatureGroup matching; a reader key never passes through it."""
        with pytest.raises(ValueError) as exc_info:

            class RodMatchGuardReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_guarded_key": PropertySpec("Guarded.", match_guard=_rod_match_guard, default=None),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodMatchGuardReader" in message
        assert "rod_guarded_key" in message
        assert "match_guard" in message

    def test_deferred_binding_spec_rejected_at_class_definition(self) -> None:
        """``deferred_binding`` exempts name-path capture; reader keys have no name path."""
        with pytest.raises(ValueError) as exc_info:

            class RodDeferredBindingReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_deferred_key": PropertySpec("Deferred.", deferred_binding=True),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodDeferredBindingReader" in message
        assert "rod_deferred_key" in message
        assert "deferred_binding" in message

    def test_context_false_spec_rejected_at_class_definition(self) -> None:
        """Reader keys are not categorized into group/context; ``context=False`` declares nothing."""
        with pytest.raises(ValueError) as exc_info:

            class RodContextFalseReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_group_key": PropertySpec("Group-categorized.", context=False, default=None),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodContextFalseReader" in message
        assert "rod_group_key" in message
        assert "context" in message

    def test_context_zero_spec_rejected_at_class_definition(self) -> None:
        """``context=0`` cannot slip past the ``is False`` check above: the construction bool guard rejects it first."""
        with pytest.raises(ValueError) as exc_info:

            class RodContextZeroReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_zero_key": _spec("Group-categorized by stealth.", context=0, default=None),
                }

        message = str(exc_info.value)
        del exc_info
        assert "context" in message
        assert "bool" in message

    def test_framework_set_with_strict_validation_rejected(self) -> None:
        """A framework-written value is never user-validated, so strictness on it is a lie."""
        with pytest.raises(ValueError) as exc_info:

            class RodFrameworkStrictReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_fw_strict_key": PropertySpec(
                        "Framework-written.",
                        allowed_values=("a", "b"),
                        strict_validation=True,
                        default="a",
                        framework_set=True,
                    ),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodFrameworkStrictReader" in message
        assert "rod_fw_strict_key" in message
        assert "framework_set" in message
        assert "strict_validation" in message

    def test_framework_set_with_required_when_rejected(self) -> None:
        """No user supplies a framework-written key, so conditional requiredness is meaningless."""
        with pytest.raises(ValueError) as exc_info:

            class RodFrameworkRequiredWhenReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_fw_required_key": PropertySpec(
                        "Framework-written.",
                        required_when=_rod_never_required,
                        default=None,
                        framework_set=True,
                    ),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodFrameworkRequiredWhenReader" in message
        assert "rod_fw_required_key" in message
        assert "framework_set" in message
        assert "required_when" in message

    def test_framework_set_without_a_declared_default_rejected(self) -> None:
        """A framework key must declare its absent-state default explicitly, ``None`` included."""
        with pytest.raises(ValueError) as exc_info:

            class RodFrameworkNoDefaultReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_fw_bare_key": PropertySpec("Framework-written.", framework_set=True),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodFrameworkNoDefaultReader" in message
        assert "rod_fw_bare_key" in message
        assert "framework_set" in message
        assert "default" in message

    def test_framework_set_with_allow_explicit_none_rejected(self) -> None:
        """The admit path skips framework keys before reading the flag, so declaring it is inert."""
        with pytest.raises(ValueError) as exc_info:

            class RodFrameworkAllowNoneReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                    "rod_fw_allow_none_key": PropertySpec(
                        "Framework-written.",
                        allow_explicit_none=True,
                        default=None,
                        framework_set=True,
                    ),
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodFrameworkAllowNoneReader" in message
        assert "rod_fw_allow_none_key" in message
        assert "framework_set" in message
        assert "allow_explicit_none" in message

    def test_a_strict_allowed_values_declaration_defines_fine(self) -> None:
        """Strict specs are the point of the collapse; a valued one with an in-space default defines."""

        class RodStrictValuesReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_strict_values_key": PropertySpec(
                    "Strictly validated (value enforcement lands in cycle 2).",
                    allowed_values=("a", "b"),
                    strict_validation=True,
                    default="a",
                ),
            }

        assert RodStrictValuesReader.reader_option_default("rod_strict_values_key") == "a"

    def test_a_strict_element_validator_declaration_defines_fine(self) -> None:
        """A validator-backed strict spec with a passing default defines fine too."""

        class RodStrictValidatorReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_strict_validator_key": PropertySpec(
                    "Validator-backed strict key.",
                    element_validator=is_positive_int,
                    strict_validation=True,
                    default=3,
                ),
            }

        assert RodStrictValidatorReader.reader_option_default("rod_strict_validator_key") == 3

    def test_a_valid_declaration_defines_fine(self) -> None:
        """Control: the check rejects only reader-invalid specs, never a real declaration."""

        class RodValidSpecReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_valid_key": PropertySpec("Valid.", default=frozenset()),
            }

        assert RodValidSpecReader.reader_option_default("rod_valid_key") == frozenset()

    def test_an_absent_declaration_defines_fine(self) -> None:
        """A reader declaring nothing is the common case and must stay definable."""

        class RodNoDeclarationReader(BaseInputData):
            pass

        assert RodNoDeclarationReader.declared_reader_option_keys() == {"BaseInputData"}

    def test_an_empty_declaration_defines_fine(self) -> None:
        """An explicitly empty mapping declares nothing new and is not an error."""

        class RodEmptyDeclarationReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {}

        assert RodEmptyDeclarationReader.declared_reader_option_keys() == {"BaseInputData"}

    def test_a_bad_declaration_on_a_subclass_of_a_good_one_still_raises(self) -> None:
        """The check runs per class, so inheriting a valid declaration does not buy a free pass."""

        class RodGoodBaseReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
                "rod_good_key": PropertySpec("Valid.", default=None),
            }

        with pytest.raises(ValueError) as exc_info:

            class RodBadChildReader(RodGoodBaseReader):
                READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {"rod_bad_key": 42}  # type: ignore[dict-item]  # wrong type is the point

        message = str(exc_info.value)
        del exc_info
        assert "RodBadChildReader" in message
        assert "rod_bad_key" in message


class TestDeclarationsDoNotAffectDiscovery:
    """The synthetic declaring classes stay invisible to reader selection."""

    def test_synthetic_declaring_classes_are_not_final_readers(self) -> None:
        """No ``load_data`` override means ``get_all_filtered_subclasses`` never collects them."""
        parent, child, override = _decl_family()

        assert parent.is_final_reader() is False
        assert child.is_final_reader() is False
        assert override.is_final_reader() is False

    def test_no_reader_this_module_leaks_is_a_final_reader(self) -> None:
        """The module's leak policy, machine-checked over every reader of this module still reachable."""
        _decl_family()
        local = [cls for cls in get_all_subclasses(BaseInputData) if cls.__module__ == __name__]

        assert local, "expected this module's throwaway readers to be reachable through __subclasses__()"
        assert [cls.__name__ for cls in local if cls.is_final_reader()] == []
