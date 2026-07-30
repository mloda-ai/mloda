"""Pins the ``READER_OPTIONS`` declaration surface on ``BaseInputData``.

Reader option keys are consumed at MATCH time, inside reader selection, which runs before the
framework materializes any ``PROPERTY_MAPPING`` default. A ``PropertySpec`` on a wrapper
FeatureGroup could therefore never be enforced for them, so readers get their own, deliberately
smaller spec type: ``ReaderOptionSpec`` declares an explanation, the ``runtime_default`` the
reader's OWN code applies when the key is absent, and a ``framework_set`` flag for keys the
framework writes and users do not.

What is pinned here:

* ``ReaderOptionSpec`` lives in ``mloda/core/abstract_plugins/components/input_data/
  reader_option_spec.py``, is a frozen dataclass with value equality, and is exported from
  ``mloda.provider``.
* ``BaseInputData.READER_OPTIONS`` declares the reserved ``"BaseInputData"`` key (the matched
  ``(ReaderClass, data_access)`` tuple written by ``add_base_input_data_to_options`` and read by
  ``init_reader``) with ``framework_set=True``.
* ``reader_option_specs()`` merges declarations across ``cls.__mro__`` with the most-derived class
  winning, ``declared_reader_option_keys()`` exposes the merged keys, and
  ``reader_option_default()`` raises a loud ``ValueError`` for an undeclared key so a typo in
  reader code cannot silently read as ``None``.
* ``reader_option(key, options)`` is the accessor reader code calls: it returns the SUPPLIED value
  whenever the key is present and the declared ``runtime_default`` otherwise, so an explicit
  falsy value ("hand nothing over") is never silently replaced by a non-empty declared default.
* ``READER_OPTIONS`` values must BE ``ReaderOptionSpec`` instances, enforced at class definition
  the way ``FeatureGroup.__init_subclass__`` enforces ``PROPERTY_MAPPING``.

Subclass-leak policy: this module DELIBERATELY leaks its throwaway ``BaseInputData`` subclasses
(module-level ones permanently, function-local ones until a gc cycle). That is benign and pinned:
none of them overrides ``load_data`` or declares ``_final_reader_requires``, so ``is_final_reader()``
is False and ``get_all_filtered_subclasses`` never collects them into reader selection.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, ClassVar

import pytest

import mloda.provider as mloda_provider
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.reader_option_spec import ReaderOptionSpec
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


def _spec(*args: Any, **kwargs: Any) -> ReaderOptionSpec:
    """Build a ``ReaderOptionSpec`` through an untyped seam so type-invalid calls stay runtime tests."""
    return ReaderOptionSpec(*args, **kwargs)


class _ReaderOptDeclParent(BaseInputData):
    """Parent declaring key A only; inherits the reserved framework key."""

    READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
        "rod_key_a": ReaderOptionSpec("Key A, declared on the parent.", runtime_default="parent_a"),
    }

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
        return None


class _ReaderOptDeclChild(_ReaderOptDeclParent):
    """Child declaring key B only; A and the reserved key must still be visible."""

    READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
        "rod_key_b": ReaderOptionSpec("Key B, declared on the child.", runtime_default="child_b"),
    }


class _ReaderOptDeclOverride(_ReaderOptDeclParent):
    """Child redeclaring key A with a different runtime_default; the most-derived declaration wins."""

    READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
        "rod_key_a": ReaderOptionSpec("Key A, redeclared on the child.", runtime_default="child_a"),
    }


class TestReaderOptionSpecType:
    """Construction, immutability, and value equality of the spec type itself."""

    def test_minimal_construction_uses_documented_defaults(self) -> None:
        """``ReaderOptionSpec("why")`` constructs with runtime_default None and framework_set False."""
        spec = ReaderOptionSpec("Why this key exists.")

        assert spec.explanation == "Why this key exists."
        assert spec.runtime_default is None
        assert spec.framework_set is False

    def test_explanation_is_positional_and_required(self) -> None:
        """The explanation is the one positional field; omitting it is Python's own TypeError."""
        with pytest.raises(TypeError):
            _spec()
        assert _spec("positional explanation").explanation == "positional explanation"

    def test_declared_fields_are_named_runtime_default_and_framework_set(self) -> None:
        """A typo'd keyword raises TypeError instead of being silently absorbed."""
        with pytest.raises(TypeError):
            _spec("x", runtime_defualt=1)
        with pytest.raises(TypeError):
            _spec("x", framework_setting=True)

    def test_instances_are_frozen(self) -> None:
        """Assigning any field raises ``dataclasses.FrozenInstanceError``."""
        spec = ReaderOptionSpec("x", runtime_default=frozenset(), framework_set=False)

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "explanation", "changed")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "runtime_default", "changed")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "framework_set", True)

    def test_equality_is_by_value(self) -> None:
        """Two specs with equal fields are equal; a differing field makes them unequal."""
        assert ReaderOptionSpec("x", runtime_default=frozenset()) == ReaderOptionSpec("x", runtime_default=frozenset())
        assert ReaderOptionSpec("x") != ReaderOptionSpec("y")
        assert ReaderOptionSpec("x", runtime_default=1) != ReaderOptionSpec("x", runtime_default=2)
        assert ReaderOptionSpec("x", framework_set=True) != ReaderOptionSpec("x", framework_set=False)

    def test_exported_from_mloda_provider(self) -> None:
        """``from mloda.provider import ReaderOptionSpec`` resolves to this exact class."""
        from mloda.provider import ReaderOptionSpec as ProviderReaderOptionSpec

        assert ProviderReaderOptionSpec is ReaderOptionSpec
        assert "ReaderOptionSpec" in mloda_provider.__all__


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

    def test_reserved_key_is_framework_set(self) -> None:
        """The reserved key is written by the framework, so users never set it."""
        spec = BaseInputData.reader_option_specs()["BaseInputData"]

        assert isinstance(spec, ReaderOptionSpec)
        assert spec.framework_set is True

    def test_reserved_key_has_no_runtime_default(self) -> None:
        """No reader code falls back for the reserved key; init_reader raises when it is absent."""
        assert BaseInputData.reader_option_default("BaseInputData") is None

    def test_reserved_key_is_the_key_the_framework_actually_writes(self) -> None:
        """``add_base_input_data_to_options`` writes exactly the keys declared ``framework_set``.

        This is what makes ``framework_set`` load-bearing: nothing in the framework reads the flag
        (deliberately, reader selection is a hot path), so the flag's meaning is pinned here instead.
        Flipping it to False, or marking a second key ``framework_set`` that nothing writes, fails.
        """
        options = Options()
        BaseInputData.add_base_input_data_to_options(_ReaderOptDeclParent, "rod_reserved_access", options)

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
        specs = _ReaderOptDeclChild.reader_option_specs()

        assert {key for key, spec in specs.items() if spec.framework_set} == {"BaseInputData"}
        assert specs["rod_key_a"].framework_set is False
        assert specs["rod_key_b"].framework_set is False


class TestMroMerge:
    """Declarations merge across ``cls.__mro__``, most-derived class winning on a collision."""

    def test_parent_keys_include_own_and_inherited(self) -> None:
        """The parent sees its own key A plus the reserved key from the base."""
        assert _ReaderOptDeclParent.declared_reader_option_keys() == {"rod_key_a", "BaseInputData"}

    def test_child_merges_parent_and_own_keys(self) -> None:
        """A child declaring only B still sees A and the reserved key."""
        assert _ReaderOptDeclChild.declared_reader_option_keys() == {"rod_key_a", "rod_key_b", "BaseInputData"}

    def test_child_declaration_does_not_leak_into_the_parent(self) -> None:
        """The merge walks up the MRO only; the parent never gains the child's key."""
        assert "rod_key_b" not in _ReaderOptDeclParent.declared_reader_option_keys()

    def test_specs_are_returned_keyed_by_option_name(self) -> None:
        """``reader_option_specs()`` maps every merged key to its ``ReaderOptionSpec``."""
        specs = _ReaderOptDeclChild.reader_option_specs()

        assert set(specs) == _ReaderOptDeclChild.declared_reader_option_keys()
        assert all(isinstance(spec, ReaderOptionSpec) for spec in specs.values())
        assert specs["rod_key_a"].runtime_default == "parent_a"
        assert specs["rod_key_b"].runtime_default == "child_b"

    def test_most_derived_declaration_wins_on_a_key_collision(self) -> None:
        """Redeclaring key A with another runtime_default overrides the parent's declaration."""
        assert _ReaderOptDeclOverride.reader_option_default("rod_key_a") == "child_a"
        assert _ReaderOptDeclOverride.reader_option_specs()["rod_key_a"].runtime_default == "child_a"
        assert _ReaderOptDeclParent.reader_option_default("rod_key_a") == "parent_a"

    def test_returned_mapping_is_a_fresh_copy(self) -> None:
        """A caller mutating the merged mapping cannot corrupt the class declarations."""
        specs = _ReaderOptDeclChild.reader_option_specs()
        specs["rod_key_injected"] = ReaderOptionSpec("Injected by a caller.")

        assert "rod_key_injected" not in _ReaderOptDeclChild.declared_reader_option_keys()
        assert "rod_key_injected" not in _ReaderOptDeclChild.reader_option_specs()


class TestReaderOptionDefault:
    """``reader_option_default`` returns the declared fallback and is loud about typos."""

    def test_declared_default_is_returned(self) -> None:
        """The declared ``runtime_default`` is what reader code gets for an absent key."""
        assert _ReaderOptDeclParent.reader_option_default("rod_key_a") == "parent_a"
        assert _ReaderOptDeclChild.reader_option_default("rod_key_b") == "child_b"

    def test_inherited_default_is_returned(self) -> None:
        """A child needs no re-declaration to reach the parent's default."""
        assert _ReaderOptDeclChild.reader_option_default("rod_key_a") == "parent_a"

    def test_undeclared_key_raises_value_error_naming_key_and_class(self) -> None:
        """A typo in reader code is loud, not a silent None."""
        with pytest.raises(ValueError) as exc_info:
            _ReaderOptDeclChild.reader_option_default("not_a_key")

        message = str(exc_info.value)
        assert "not_a_key" in message
        assert "_ReaderOptDeclChild" in message

    def test_undeclared_key_on_base_raises_value_error(self) -> None:
        """The same guard holds on ``BaseInputData`` itself."""
        with pytest.raises(ValueError, match="not_a_key"):
            BaseInputData.reader_option_default("not_a_key")


class TestReaderOptionHonoursPresence:
    """``reader_option(key, options)`` reads presence, not truthiness."""

    def test_signature_is_key_first_options_second(self) -> None:
        """The KEY comes first, mirroring ``reader_option_default(key)``; the Options is second."""
        parameters = list(inspect.signature(BaseInputData.reader_option).parameters)

        assert parameters == ["key", "options"]

    def test_absent_key_falls_back_to_the_declared_default(self) -> None:
        """Nothing supplied means the reader's own declared fallback applies."""
        assert _ReaderOptDeclParent.reader_option("rod_key_a", Options()) == "parent_a"

    def test_supplied_value_wins_over_the_declared_default(self) -> None:
        """A user-set value is returned unchanged."""
        options = Options({"rod_key_a": "supplied"})

        assert _ReaderOptDeclParent.reader_option("rod_key_a", options) == "supplied"

    @pytest.mark.parametrize("falsy_value", [frozenset(), (), [], "", 0, False, {}])
    def test_present_but_falsy_value_is_honoured_not_replaced(self, falsy_value: Any) -> None:
        """An explicit empty value means "hand nothing over" and must survive the read.

        This is the whole defect: ``options.get(key) or cls.reader_option_default(key)`` cannot tell
        an absent key from an explicit empty one, so a non-empty declared default silently wins and
        the option can never be turned off.
        """
        options = Options({"rod_key_a": falsy_value})

        result = _ReaderOptDeclParent.reader_option("rod_key_a", options)

        assert result == falsy_value
        assert result != "parent_a"

    def test_explicit_none_reads_as_absent(self) -> None:
        """``None`` is the framework's dominant absence marker, so the declared default applies."""
        options = Options({"rod_key_a": None})

        assert "rod_key_a" in options
        assert _ReaderOptDeclParent.reader_option("rod_key_a", options) == "parent_a"

    def test_a_context_option_is_read_like_a_group_option(self) -> None:
        """The accessor reads through ``Options.get``, so the category never changes the answer."""
        options = Options(context={"rod_key_a": frozenset()})

        assert _ReaderOptDeclParent.reader_option("rod_key_a", options) == frozenset()

    def test_inherited_declaration_supplies_the_default(self) -> None:
        """A child needs no re-declaration to reach the parent's fallback."""
        assert _ReaderOptDeclChild.reader_option("rod_key_a", Options()) == "parent_a"

    def test_most_derived_declaration_supplies_the_default(self) -> None:
        """A redeclared key resolves to the most-derived ``runtime_default``, like the sibling accessor."""
        assert _ReaderOptDeclOverride.reader_option("rod_key_a", Options()) == "child_a"
        assert _ReaderOptDeclParent.reader_option("rod_key_a", Options()) == "parent_a"

    def test_undeclared_key_raises_value_error_naming_key_and_class(self) -> None:
        """A typo is loud here exactly as in ``reader_option_default``, not a silent None."""
        with pytest.raises(ValueError) as exc_info:
            _ReaderOptDeclChild.reader_option("not_a_key", Options())

        message = str(exc_info.value)
        assert "not_a_key" in message
        assert "_ReaderOptDeclChild" in message

    def test_undeclared_key_raises_even_when_a_value_is_supplied(self) -> None:
        """The declaration gates the read: a supplied value cannot legitimize an undeclared key."""
        options = Options({"not_a_key": "supplied"})

        with pytest.raises(ValueError, match="not_a_key"):
            _ReaderOptDeclChild.reader_option("not_a_key", options)

    def test_undeclared_key_on_the_base_raises(self) -> None:
        """The same guard holds on ``BaseInputData`` itself."""
        with pytest.raises(ValueError, match="not_a_key"):
            BaseInputData.reader_option("not_a_key", Options())


class TestReaderOptionSpecCacheStaysFresh:
    """The merged-spec cache is per class, so it can never answer for the wrong class.

    Every family below is built INSIDE its test so the cache starts cold: module-level classes are
    warmed by whichever test the xdist worker happened to run first, which would make these vacuous.
    """

    def test_a_subclass_defined_after_a_warm_parent_cache_sees_its_own_declaration(self) -> None:
        """Warming the parent must not answer for a child that redeclares the key."""

        class RodColdParentReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Declared on the parent.", runtime_default="parent"),
            }

        assert RodColdParentReader.reader_option("rod_cache_key", Options()) == "parent"

        class RodLateChildReader(RodColdParentReader):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Redeclared on the child.", runtime_default="child"),
            }

        assert RodLateChildReader.reader_option("rod_cache_key", Options()) == "child"
        assert RodColdParentReader.reader_option("rod_cache_key", Options()) == "parent"

    def test_a_warm_child_cache_does_not_change_the_parent(self) -> None:
        """The reverse order: reading the child first leaves the parent's answer alone."""

        class RodParentReadSecond(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Declared on the parent.", runtime_default="parent"),
            }

        class RodChildReadFirst(RodParentReadSecond):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Redeclared on the child.", runtime_default="child"),
            }

        assert RodChildReadFirst.reader_option("rod_cache_key", Options()) == "child"
        assert RodParentReadSecond.reader_option("rod_cache_key", Options()) == "parent"

    def test_a_key_added_by_a_late_subclass_is_visible_after_a_warm_parent_cache(self) -> None:
        """``declared_reader_option_keys`` shares the cache, so it must not go stale either."""

        class RodKeysParentReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_parent_key": ReaderOptionSpec("Parent only."),
            }

        assert RodKeysParentReader.declared_reader_option_keys() == {"rod_cache_parent_key", "BaseInputData"}

        class RodKeysChildReader(RodKeysParentReader):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_child_key": ReaderOptionSpec("Child only."),
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
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Declared.", runtime_default="declared"),
            }

        assert RodWarmGuardReader.reader_option("rod_cache_key", Options()) == "declared"

        with pytest.raises(ValueError, match="not_a_key"):
            RodWarmGuardReader.reader_option("not_a_key", Options())

    def test_mutating_the_returned_specs_cannot_poison_the_cache(self) -> None:
        """The caller-facing mapping stays a fresh copy: a cache must not be handed out by reference."""

        class RodCopyProbeReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Declared.", runtime_default="declared"),
            }

        specs = RodCopyProbeReader.reader_option_specs()
        specs["rod_cache_key"] = ReaderOptionSpec("Injected.", runtime_default="injected")
        specs["rod_cache_injected_key"] = ReaderOptionSpec("Injected.")

        assert RodCopyProbeReader.reader_option("rod_cache_key", Options()) == "declared"
        assert "rod_cache_injected_key" not in RodCopyProbeReader.declared_reader_option_keys()

    def test_repeated_reads_stay_equal(self) -> None:
        """Caching is invisible: two reads of the same key on the same class agree."""

        class RodRepeatReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_cache_key": ReaderOptionSpec("Declared.", runtime_default=frozenset({".json"})),
            }

        first = RodRepeatReader.reader_option("rod_cache_key", Options())
        second = RodRepeatReader.reader_option("rod_cache_key", Options())

        assert first == second == frozenset({".json"})
        assert RodRepeatReader.reader_option_specs() == RodRepeatReader.reader_option_specs()


class TestReaderOptionsAreValidatedAtClassDefinition:
    """``READER_OPTIONS`` accepts ``ReaderOptionSpec`` instances and NOTHING else.

    Mirrors ``FeatureGroup.__init_subclass__``'s ``PROPERTY_MAPPING`` type rule (see
    ``tests/test_core/test_abstract_plugins/test_components/feature_chainer/test_property_spec_hard_break.py``):
    the mistake must surface where it is written, not later as an ``AttributeError`` on
    ``runtime_default`` deep inside reader matching.
    """

    def test_string_value_rejected_at_class_definition(self) -> None:
        """The reviewer's exact case: ``{"k": "just a string"}`` names the class, the key and the type."""
        with pytest.raises(ValueError) as exc_info:

            class RodBadStringSpecReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {"rod_bad_key": "just a string"}  # type: ignore[dict-item]  # wrong type is the point

        message = str(exc_info.value)
        del exc_info
        assert "RodBadStringSpecReader" in message
        assert "rod_bad_key" in message
        assert "ReaderOptionSpec" in message

    def test_dict_value_rejected_at_class_definition(self) -> None:
        """A hand-rolled spec dict (the plausible authoring mistake) is rejected the same way."""
        with pytest.raises(ValueError) as exc_info:

            class RodBadDictSpecReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                    "rod_bad_key": {"explanation": "x", "runtime_default": None}  # type: ignore[dict-item]  # wrong type is the point
                }

        message = str(exc_info.value)
        del exc_info
        assert "RodBadDictSpecReader" in message
        assert "rod_bad_key" in message
        assert "ReaderOptionSpec" in message

    def test_a_property_spec_is_not_a_reader_option_spec(self) -> None:
        """The two declaration surfaces are separate types; crossing them is the same error."""
        with pytest.raises(ValueError) as exc_info:

            class RodPropertySpecReader(BaseInputData):
                READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                    "rod_bad_key": PropertySpec("wrong surface")  # type: ignore[dict-item]  # wrong type is the point
                }

        message = str(exc_info.value)
        del exc_info
        assert "rod_bad_key" in message
        assert "ReaderOptionSpec" in message

    def test_a_valid_declaration_defines_fine(self) -> None:
        """Control: the check rejects only the wrong type, never a real declaration."""

        class RodValidSpecReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_valid_key": ReaderOptionSpec("Valid.", runtime_default=frozenset()),
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
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {}

        assert RodEmptyDeclarationReader.declared_reader_option_keys() == {"BaseInputData"}

    def test_a_bad_declaration_on_a_subclass_of_a_good_one_still_raises(self) -> None:
        """The check runs per class, so inheriting a valid declaration does not buy a free pass."""

        class RodGoodBaseReader(BaseInputData):
            READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {
                "rod_good_key": ReaderOptionSpec("Valid."),
            }

        with pytest.raises(ValueError) as exc_info:

            class RodBadChildReader(RodGoodBaseReader):
                READER_OPTIONS: ClassVar[dict[str, ReaderOptionSpec]] = {"rod_bad_key": 42}  # type: ignore[dict-item]  # wrong type is the point

        message = str(exc_info.value)
        del exc_info
        assert "RodBadChildReader" in message
        assert "rod_bad_key" in message


class TestDeclarationsDoNotAffectDiscovery:
    """The synthetic declaring classes stay invisible to reader selection."""

    def test_synthetic_declaring_classes_are_not_final_readers(self) -> None:
        """No ``load_data`` override means ``get_all_filtered_subclasses`` never collects them."""
        assert _ReaderOptDeclParent.is_final_reader() is False
        assert _ReaderOptDeclChild.is_final_reader() is False
        assert _ReaderOptDeclOverride.is_final_reader() is False

    def test_no_reader_this_module_leaks_is_a_final_reader(self) -> None:
        """The module's leak policy, machine-checked over every reader of this module still reachable."""
        local = [cls for cls in get_all_subclasses(BaseInputData) if cls.__module__ == __name__]

        assert local, "expected this module's throwaway readers to be reachable through __subclasses__()"
        assert [cls.__name__ for cls in local if cls.is_final_reader()] == []
