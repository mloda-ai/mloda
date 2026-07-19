"""A PROPERTY_MAPPING spec's schema IS the ``PropertySpec`` constructor (issue #694).

The dict-spec era validated every spec against a curated key set (``PROPERTY_SPEC_KEYS``),
suggested the nearest known key for a typo and named replacements for removed keys, because a
dict absorbs any key silently: one typo'd flag used to become an accepted VALUE while strict
validation switched itself off. With ``PropertySpec`` there is no key set left to curate. The
constructor signature is the schema: a misspelled or retired field name is Python's own
``TypeError``, raised at the exact line where the spec literal is written, and nothing can
ever be absorbed as a value.

This module pins that schema surface: the field set, and the unknown-field ``TypeError``
naming the offending keyword. The machinery deletion itself (``PROPERTY_SPEC_KEYS``,
``REMOVED_PROPERTY_KEYS``) is pinned in ``test_property_spec_hard_break.py``; the value-shape
construction rules live in ``test_property_spec_type.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec

TYPO_STRICT_FIELD = "strict_validaton"  # the headline typo: 'strict_validation' minus one 'i'
STALE_ELEMENT_VALIDATOR_FIELD = "validation_function"
STALE_MATCH_GUARD_FIELD = "type_validator"


def _spec(*args: Any, **kwargs: Any) -> PropertySpec:
    """Build a ``PropertySpec`` through an untyped seam.

    The type-invalid calls below (unknown keywords) must stay runtime tests; routing them
    through ``Any`` keeps the module mypy --strict clean.
    """
    return PropertySpec(*args, **kwargs)


def _accept_anything(value: Any) -> bool:
    return True


class TestTheConstructorIsTheSchema:
    """The dataclass field set is the whole schema; an unknown field cannot exist."""

    def test_field_set_is_exactly_the_schema(self) -> None:
        """The schema is these ten fields, nothing more: there is no key set to drift from."""
        assert {field.name for field in dataclasses.fields(PropertySpec)} == {
            "explanation",
            "allowed_values",
            "default",
            "context",
            "strict_validation",
            "element_validator",
            "match_guard",
            "required_when",
            "allow_explicit_none",
            "deferred_binding",
        }

    def test_typo_field_is_a_constructor_type_error_naming_the_offender(self) -> None:
        """The headline typo is a ``TypeError`` at the spec literal, naming the bad keyword.

        The old model absorbed the typo as an accepted VALUE with strict validation silently
        off. (Python 3.12+ additionally suggests the nearest field name; that part of the
        message is the interpreter's and is deliberately not pinned.)
        """
        with pytest.raises(TypeError) as exc_info:
            _spec(
                "The arithmetic operation to apply",
                allowed_values={"add": "Addition", "sub": "Subtraction"},
                **{TYPO_STRICT_FIELD: True},
            )

        message = str(exc_info.value)
        assert "unexpected keyword argument" in message
        assert TYPO_STRICT_FIELD in message

    def test_arbitrary_unknown_field_is_a_constructor_type_error(self) -> None:
        """An unknown field with no near miss is the same ``TypeError``, naming the offender."""
        with pytest.raises(TypeError) as exc_info:
            _spec("Size of the time window", documentation_url="https://example.invalid/window_size")

        message = str(exc_info.value)
        assert "unexpected keyword argument" in message
        assert "documentation_url" in message


class TestRetiredFieldNames:
    """The pre-rename callable field names are plain unknown fields now, not curated renames.

    ``element_validator`` and ``match_guard`` are the accepted spellings (they are in the
    field set above); their predecessors get no rename hint, just the constructor's
    ``TypeError``.
    """

    def test_validation_function_is_an_unknown_field(self) -> None:
        """``validation_function`` is simply not a field."""
        with pytest.raises(TypeError) as exc_info:
            _spec("The arithmetic operation to apply", **{STALE_ELEMENT_VALIDATOR_FIELD: _accept_anything})

        message = str(exc_info.value)
        assert "unexpected keyword argument" in message
        assert STALE_ELEMENT_VALIDATOR_FIELD in message

    def test_type_validator_is_an_unknown_field(self) -> None:
        """``type_validator`` is simply not a field."""
        with pytest.raises(TypeError) as exc_info:
            _spec("List of columns to partition by", **{STALE_MATCH_GUARD_FIELD: _accept_anything})

        message = str(exc_info.value)
        assert "unexpected keyword argument" in message
        assert STALE_MATCH_GUARD_FIELD in message
