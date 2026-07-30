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

The synthetic classes below are built directly on ``BaseInputData``, never override ``load_data``
(so ``is_final_reader()`` is False and reader discovery never collects them), and their matcher
returns None; they cannot pollute reader selection in sibling tests.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import pytest

import mloda.provider as mloda_provider
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.reader_option_spec import ReaderOptionSpec
from mloda.core.abstract_plugins.components.options import Options


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
        """``add_base_input_data_to_options`` writes exactly the declared reserved key."""
        options = Options()
        BaseInputData.add_base_input_data_to_options(_ReaderOptDeclParent, "rod_reserved_access", options)

        written = set(options.keys())
        assert written == {"BaseInputData"}
        assert written <= BaseInputData.declared_reader_option_keys()


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


class TestDeclarationsDoNotAffectDiscovery:
    """The synthetic declaring classes stay invisible to reader selection."""

    def test_synthetic_declaring_classes_are_not_final_readers(self) -> None:
        """No ``load_data`` override means ``get_all_filtered_subclasses`` never collects them."""
        assert _ReaderOptDeclParent.is_final_reader() is False
        assert _ReaderOptDeclChild.is_final_reader() is False
        assert _ReaderOptDeclOverride.is_final_reader() is False
