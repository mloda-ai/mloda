"""A PROPERTY_MAPPING spec has a SHAPE, not just a key set (issue #600 follow-up).

``test_property_mapping_spec_schema.py`` closed the unknown-KEY hole: every key of a spec must
be a known spec key. That rule is airtight, but it says nothing about the SHAPE OF THE VALUE
under each key, which reopens the same class of silent widening from the other side:

    {DefaultOptionKeys.allowed_values: ("add"), DefaultOptionKeys.strict_validation: True}

A forgotten comma makes that a ``str``, not a one-tuple. It defines cleanly, and at match time
``found_val in "add"`` is a SUBSTRING test: ``"add"``, ``"a"``, ``"ad"``, ``"dd"`` and ``""``
are all accepted. The retired flattened form could never express this (its value space was
always a dict), so this is net-new surface opened by making ``allowed_values`` the one channel.

The same truthiness-shaped hole runs through the rest of the layer:

* ``strict_validation`` needing a "non-empty" value space is a TRUTHINESS test, so an EMPTY
  generator passes (a generator object is always truthy), a NON-EMPTY one passes and is then
  CONSUMED (matching becomes stateful and order-dependent), and a scalar passes and then makes
  ``value in 5`` raise a ``TypeError`` that is swallowed into a silent reject-everything.
* the callable-valued keys are checked for PRESENCE, not CALLABILITY, so a list under
  ``element_validator`` escapes as ``TypeError: 'list' object is not callable`` out of matching,
  and with a declared default the broad ``except Exception`` in ``check_declared_default``
  rewrites it into a confidently WRONG message about the default.
* ``_is_strict_validation`` is truthiness-based, so ``strict_validation: "false"`` is strict and
  the error message asserts ``strict_validation=True`` about a spec that never said so.
* ``property_spec`` refuses to build a NON-STRICT spec with ``allowed_values``, on the premise
  that "allowed_values is never enforced without strict=True". That premise is false: the value
  space of a non-strict spec is consumed by ``_find_property_key_for_value`` and
  ``_build_effective_options`` to map a name-parsed value back to its PROPERTY_MAPPING key.

This module pins the shape rules that close all of it. The ordering constraint from the schema
module is preserved: the unknown-key rule still runs FIRST, so a spec carrying a removed key is
still reported as a rename, not as a shape error.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup

TYPO_STRICT_KEY = "strict_validaton"  # 'strict_validation' minus one 'i'
TYPO_CONTEXT_KEY = "contxt"  # 'context' minus one 'e'
STALE_ELEMENT_VALIDATOR_KEY = "validation_function"


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Mirrors ``test_property_mapping_spec_schema.py``: the class-definition tests below define
    FeatureGroup subclasses, which linger in ``FeatureGroup.__subclasses__()`` until a GC cycle
    runs and would otherwise be seen by tests that enumerate feature groups.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _one_shot_values() -> Iterator[str]:
    """A NON-EMPTY generator: truthy, but a one-shot value space that empties itself."""
    yield "add"
    yield "sub"


def _no_values() -> Iterator[str]:
    """An EMPTY generator: still truthy, so it passes a truthiness-based non-empty test."""
    yield from ()


class TestAllowedValuesMustBeACollection:
    """``allowed_values`` declares a VALUE SPACE, so it must be a Collection, never a str.

    A ``str`` is the dangerous case: it is a perfectly good Collection of characters, so
    membership silently degrades into a substring test. ``bytes`` behaves the same way. Both
    are rejected outright, because no author ever means "any substring of this word".
    """

    def test_str_allowed_values_rejected_at_class_definition(self) -> None:
        """A ``str`` value space (the forgotten-comma bug) raises, naming the real fault."""
        with pytest.raises(ValueError) as exc_info:

            class StrAllowedValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        # A forgotten comma: ("add") is the str "add", not the tuple ("add",).
                        DefaultOptionKeys.allowed_values: "add",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "StrAllowedValuesFeatureGroup" in message
        assert "operation_type" in message
        assert DefaultOptionKeys.allowed_values.value in message
        assert "str" in message, "the message must name the offending shape, not just say 'invalid'"

    def test_bytes_allowed_values_rejected_at_class_definition(self) -> None:
        """``bytes`` substring-matches exactly like ``str``, so it is rejected the same way."""
        with pytest.raises(ValueError) as exc_info:

            class BytesAllowedValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: b"add",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "BytesAllowedValuesFeatureGroup" in message
        assert DefaultOptionKeys.allowed_values.value in message

    @pytest.mark.parametrize("candidate", ["add", "a", "ad", "dd", ""])
    def test_str_allowed_values_never_substring_matches(self, candidate: str) -> None:
        """The regression itself, checked through MATCHING.

        Every one of these is accepted today: the full word, each of its substrings, and the
        empty string. A spec that cannot be defined can never accept any of them.
        """
        matched: bool

        try:

            class SubstringMatchFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: "add",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

            matched = SubstringMatchFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": candidate})
            )
        except ValueError:
            matched = False

        assert matched is False, "a str allowed_values must never turn membership into a substring test"

    def test_parser_entry_point_rejects_str_allowed_values(self) -> None:
        """The rule lives in core, so validating a mapping directly raises the same way."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.allowed_values: "add",
                DefaultOptionKeys.strict_validation: True,
            }
        }

        with pytest.raises(ValueError, match="operation_type"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    def test_str_allowed_values_rejected_without_strict_validation(self) -> None:
        """The shape rule is about the value space itself, so it does not depend on strict.

        A non-strict value space is still consumed (name-parsed value -> property key), so a
        str there would substring-map arbitrary operation configs onto the wrong key.
        """
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.allowed_values: "add",
                DefaultOptionKeys.strict_validation: False,
            }
        }

        with pytest.raises(ValueError, match="operation_type"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    @pytest.mark.parametrize(
        "value_space",
        [
            {"add": "Addition", "sub": "Subtraction"},
            ("add", "sub"),
            ["add", "sub"],
            {"add", "sub"},
            frozenset({"add", "sub"}),
        ],
        ids=["dict", "tuple", "list", "set", "frozenset"],
    )
    def test_every_real_collection_form_still_defines_and_matches(self, value_space: Any) -> None:
        """The guard rejects only non-Collections: every legitimate container keeps working."""

        class CollectionValueSpaceFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "operation_type": {
                    DefaultOptionKeys.allowed_values: value_space,
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            CollectionValueSpaceFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": "add"})
            )
            is True
        )
        assert (
            CollectionValueSpaceFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": "mul"})
            )
            is False
        )


class TestNonEmptyIsNotATruthinessTest:
    """ "Strict needs a non-empty value space" must mean non-empty, not merely truthy.

    A generator object and a scalar are both truthy and both pass the rule today, each in a
    different, worse-than-empty way. The Collection guard is what makes "non-empty" checkable.
    """

    def test_empty_generator_allowed_values_rejected(self) -> None:
        """An EMPTY generator is truthy, so it slips past the strict-needs-a-value-space rule."""
        with pytest.raises(ValueError) as exc_info:

            class EmptyGeneratorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: _no_values(),
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "EmptyGeneratorFeatureGroup" in message
        assert DefaultOptionKeys.allowed_values.value in message

    def test_non_empty_generator_allowed_values_rejected(self) -> None:
        """A NON-EMPTY generator is a one-shot value space, so it is not a value space at all."""
        with pytest.raises(ValueError) as exc_info:

            class GeneratorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: _one_shot_values(),
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "GeneratorFeatureGroup" in message
        assert DefaultOptionKeys.allowed_values.value in message

    def test_generator_allowed_values_never_makes_matching_stateful(self) -> None:
        """Matching the SAME value twice must give the SAME answer.

        Today a generator value space is consumed by matching: the first match succeeds, the
        second raises "not found in mapping". That makes matching order-dependent, and so
        nondeterministic under pytest-xdist. A spec that cannot be defined cannot be stateful.
        """
        matches: list[bool]

        try:

            class StatefulGeneratorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: _one_shot_values(),
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

            options = Options(context={"operation_type": "add"})
            matches = [StatefulGeneratorFeatureGroup.match_feature_group_criteria("any_feature", options)] * 1
            matches.append(StatefulGeneratorFeatureGroup.match_feature_group_criteria("any_feature", options))
        except ValueError:
            matches = [False, False]

        assert matches[0] == matches[1], "matching must not depend on how often the value space was consulted"
        assert matches == [False, False], "a generator is not a value space, so the spec must not be definable"

    def test_generator_allowed_values_with_a_default_is_rejected_not_burnt(self) -> None:
        """With a declared default, ``check_declared_default`` burns the generator at definition.

        Today this class defines cleanly and then rejects EVERY runtime value, including the
        very values it declares, because the default check consumed the value space.
        """
        with pytest.raises(ValueError):

            class BurntGeneratorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: _one_shot_values(),
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: "add",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

    def test_scalar_allowed_values_rejected(self) -> None:
        """A scalar is truthy, then ``value in 5`` raises a TypeError that is swallowed.

        The spec silently rejects every value, including the scalar itself.
        """
        with pytest.raises(ValueError) as exc_info:

            class ScalarAllowedValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "window_size": {
                        DefaultOptionKeys.allowed_values: 5,
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "ScalarAllowedValuesFeatureGroup" in message
        assert "window_size" in message
        assert DefaultOptionKeys.allowed_values.value in message

    def test_strict_with_empty_collection_allowed_values_still_raises(self) -> None:
        """The existing empty-value-space rule survives the shape guard, for every container."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.allowed_values: (),
                DefaultOptionKeys.strict_validation: True,
            }
        }

        with pytest.raises(ValueError, match="operation_type"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)


class TestValidatorKeysMustBeCallable:
    """``element_validator``, ``required_when`` and ``match_guard`` are checked for CALLABILITY.

    Today they are checked only for PRESENCE. ``property_spec`` DOES check callability, so core
    and the builder disagree: a hand-written spec can carry a list where a callable belongs.
    """

    def test_non_callable_element_validator_rejected_at_class_definition(self) -> None:
        """A list under ``element_validator`` raises at definition, naming the real fault."""
        with pytest.raises(ValueError) as exc_info:

            class NonCallableElementValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.element_validator: ["add", "sub"],
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "NonCallableElementValidatorFeatureGroup" in message
        assert "operation_type" in message
        assert DefaultOptionKeys.element_validator.value in message
        assert "callable" in message

    def test_non_callable_element_validator_message_does_not_blame_the_default(self) -> None:
        """The headline mis-diagnosis: with a default, the broad ``except Exception`` lies.

        ``check_declared_default`` catches the ``TypeError: 'list' object is not callable`` and
        reports "the key's element_validator raised an error when called with that default. Add
        the default to the accepted values, or remove the default", none of which is the fault.
        """
        with pytest.raises(ValueError) as exc_info:

            class BlamedDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.element_validator: ["add", "sub"],
                        DefaultOptionKeys.default: "add",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert DefaultOptionKeys.element_validator.value in message
        assert "callable" in message
        assert "Add the default to the accepted values" not in message, (
            "a non-callable validator is not a default problem, and must not be reported as one"
        )

    def test_non_callable_element_validator_never_reaches_match_time(self) -> None:
        """Today matching a spec with a list validator escapes ``'list' object is not callable``.

        The class definition must reject it, so no ``TypeError`` can ever leave matching. An
        escaping ``TypeError`` fails this test rather than being caught.
        """
        matched: bool

        try:

            class EscapingTypeErrorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.element_validator: ["add", "sub"],
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

            matched = EscapingTypeErrorFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": "add"})
            )
        except ValueError:
            matched = False

        assert matched is False

    def test_non_callable_required_when_rejected_at_class_definition(self) -> None:
        """A non-callable ``required_when`` is silently skipped with a warning today."""
        with pytest.raises(ValueError) as exc_info:

            class NonCallableRequiredWhenFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "region": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.required_when: ["prod"],
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "NonCallableRequiredWhenFeatureGroup" in message
        assert "region" in message
        assert DefaultOptionKeys.required_when.value in message
        assert "callable" in message

    def test_non_callable_match_guard_rejected_at_class_definition(self) -> None:
        """A non-callable ``match_guard`` silently turns into a reject-everything guard today."""
        with pytest.raises(ValueError) as exc_info:

            class NonCallableMatchGuardFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "partition_by": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.match_guard: "not_a_callable",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "NonCallableMatchGuardFeatureGroup" in message
        assert "partition_by" in message
        assert DefaultOptionKeys.match_guard.value in message
        assert "callable" in message

    def test_parser_entry_point_rejects_non_callable_validator(self) -> None:
        """The rule lives in core, so validating a mapping directly raises the same way."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.element_validator: ["add", "sub"],
            }
        }

        with pytest.raises(ValueError, match="callable"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    def test_callable_validators_still_define(self) -> None:
        """The guard rejects only non-callables: real callables keep working, unchanged."""

        class CallableValidatorsFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "window_size": {
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: _positive_int,
                    DefaultOptionKeys.match_guard: _positive_int,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            CallableValidatorsFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"window_size": 7})
            )
            is True
        )


class TestSchemaRuleStillRunsFirst:
    """The unknown-key rule keeps precedence over the new shape rules.

    A malformed spec's remaining keys cannot be trusted, and a REMOVED key is the one offender
    with an exact remedy. The shape rules must slot in BEHIND the schema rule, so a stale spec
    is still reported as a rename.
    """

    def test_removed_key_is_reported_even_when_a_shape_rule_also_fires(self) -> None:
        """A stale key plus a bad ``allowed_values`` shape: the rename still leads."""
        with pytest.raises(ValueError) as exc_info:

            class StaleAndMisshapenFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: "add",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        STALE_ELEMENT_VALIDATOR_KEY: _positive_int,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert STALE_ELEMENT_VALIDATOR_KEY in message
        assert DefaultOptionKeys.element_validator.value in message

    def test_unknown_key_is_reported_even_when_a_validator_is_not_callable(self) -> None:
        """An unknown key plus a non-callable validator: the unknown key still leads."""
        with pytest.raises(ValueError) as exc_info:

            class UnknownAndNonCallableFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.element_validator: ["add"],
                        "documentation_url": "https://example.invalid/op",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "documentation_url" in message


class TestEveryUnknownKeyIsReported:
    """The unknown-key message lists EVERY offender, not just the first one.

    Fixing one typo and re-running to discover the next is a needless round trip, and with a
    removed key present the typo is never mentioned at all.
    """

    def test_all_unknown_keys_are_listed(self) -> None:
        """Two typo'd keys, one message: both are named."""
        with pytest.raises(ValueError) as exc_info:

            class TwoTyposFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: {"add": "Addition"},
                        TYPO_STRICT_KEY: True,
                        TYPO_CONTEXT_KEY: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert TYPO_STRICT_KEY in message
        assert TYPO_CONTEXT_KEY in message, "every unknown key must be reported, not just the first"

    def test_removed_key_leads_but_other_unknown_keys_are_still_listed(self) -> None:
        """A removed key still leads with its exact remedy, and the typo is named too."""
        with pytest.raises(ValueError) as exc_info:

            class StalePlusTypoFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: {"add": "Addition"},
                        DefaultOptionKeys.strict_validation: True,
                        STALE_ELEMENT_VALIDATOR_KEY: _positive_int,
                        TYPO_CONTEXT_KEY: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert STALE_ELEMENT_VALIDATOR_KEY in message
        assert DefaultOptionKeys.element_validator.value in message, "the removed key keeps its exact remedy"
        assert TYPO_CONTEXT_KEY in message, "a removed key must not hide the other unknown keys"


class TestStrictValidationFlagMustBeBool:
    """``strict_validation`` must be a real ``bool``, not merely truthy.

    ``_is_strict_validation`` is truthiness-based, so ``strict_validation: "false"`` reads as
    strict. The error message then asserts ``strict_validation=True`` about a spec that literally
    says ``"false"``. Requiring a real bool is the fix rather than softening the message: a truthy
    non-bool flag is itself a latent bug of exactly the kind this layer exists to eliminate, and
    no spec in the repository passes a non-bool today.
    """

    @pytest.mark.parametrize("flag", ["false", "true", 1, 0, "", None], ids=repr)
    def test_non_bool_strict_validation_flag_rejected(self, flag: Any) -> None:
        """Any non-bool flag raises at class definition, truthy or falsy."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.allowed_values: {"add": "Addition"},
                DefaultOptionKeys.strict_validation: flag,
            }
        }

        with pytest.raises(ValueError, match="strict_validation"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    def test_non_bool_flag_message_does_not_assert_strict_validation_is_true(self) -> None:
        """The message must not claim ``strict_validation=True`` about a spec saying ``"false"``."""
        with pytest.raises(ValueError) as exc_info:

            class TruthyStringStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: "false",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "TruthyStringStrictFeatureGroup" in message
        assert DefaultOptionKeys.strict_validation.value in message
        assert "bool" in message
        assert "strict_validation=True" not in message, (
            "the spec says 'false': the message must not assert a value the author never wrote"
        )

    def test_truthy_non_bool_flag_never_silently_enables_strict_matching(self) -> None:
        """A ``"false"`` flag reads as strict today, so the spec does the OPPOSITE of what it says.

        The observable tell is the ValueError the strict path raises and
        ``match_feature_group_criteria`` swallows: ``_strict_validation_rejection_reason``
        surfaces it. A flag saying ``"false"`` must never produce a strict rejection, and after
        the guard it cannot: the spec does not define at all.
        """
        reason: str | None

        try:

            class SilentlyStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.allowed_values: {"add": "Addition"},
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: "false",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

            reason = SilentlyStrictFeatureGroup._strict_validation_rejection_reason(
                "any_feature", Options(context={"operation_type": "mul"})
            )
        except ValueError:
            reason = None

        assert reason is None, "a spec whose flag literally says 'false' must never enforce strict validation"

    @pytest.mark.parametrize("flag", [True, False], ids=["True", "False"])
    def test_real_bool_flags_still_define(self, flag: bool) -> None:
        """Both real bools keep working: the guard rejects only non-bools."""

        class RealBoolStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "operation_type": {
                    DefaultOptionKeys.allowed_values: {"add": "Addition"},
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: flag,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            RealBoolStrictFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": "add"})
            )
            is True
        )


def _region_required_in_prod(options: Options) -> bool:
    """``region`` is only required when deploying to prod."""
    return bool(options.get("env") == "prod")


class TestPropertySpecBuildsNonStrictValueSpace:
    """``property_spec`` must be able to build a NON-STRICT spec WITH ``allowed_values``.

    It refuses today, on the premise "allowed_values is never enforced without strict=True".
    That premise is false. A non-strict value space IS consumed, by two seams that call
    ``_extract_property_values`` unconditionally:

    * ``FeatureChainParserMixin._find_property_key_for_value`` (the forwarded-name-mismatch guard)
    * ``FeatureChainParserMixin._build_effective_options`` (required_when)

    Both use it to map a name-parsed ``operation_config`` back to its PROPERTY_MAPPING key.
    Before the flattened form was retired, such a spec carried its values flattened; now
    ``allowed_values`` is the ONLY channel, and the builder refuses the shape the repository's
    own fixtures need (``propagate_context_feature.py`` 'env', ``chainer_context_feature.py``
    'property3'), which is why those fixtures are hand-written dicts.
    """

    def test_property_spec_builds_a_non_strict_spec_with_allowed_values(self) -> None:
        """The shape the fixtures need: a declared value space, no strict enforcement."""
        spec = property_spec(
            "Deployment environment",
            strict=False,
            allowed_values={"prod": "Production", "staging": "Staging"},
        )

        assert spec[DefaultOptionKeys.allowed_values] == {"prod": "Production", "staging": "Staging"}
        assert spec[DefaultOptionKeys.strict_validation] is False

    def test_built_non_strict_value_space_maps_a_name_parsed_value_to_its_property_key(self) -> None:
        """``_find_property_key_for_value`` recovers the key from a NON-STRICT value space."""
        mapping: dict[str, Any] = {
            "env": property_spec(
                "Deployment environment",
                strict=False,
                allowed_values={"prod": "Production", "staging": "Staging"},
            )
        }

        assert FeatureChainParserMixin._find_property_key_for_value(mapping, "prod") == "env"
        assert FeatureChainParserMixin._find_property_key_for_value(mapping, "nonsense") is None

    def test_built_non_strict_value_space_merges_the_name_parsed_value_into_effective_options(self) -> None:
        """``_build_effective_options`` maps the name-parsed value onto the key via that space."""
        mapping: dict[str, Any] = {
            "env": property_spec(
                "Deployment environment",
                strict=False,
                allowed_values={"prod": "Production", "staging": "Staging"},
            )
        }

        effective = FeatureChainParserMixin._build_effective_options(
            "service__prod", [r".*__([\w]+)$"], mapping, Options()
        )

        assert effective.get("env") == "prod"

    def test_non_strict_value_space_drives_required_when_end_to_end(self) -> None:
        """The whole point: without the value space, ``required_when`` never sees the env.

        ``region`` is required only for prod. The env arrives from the feature NAME, so it can
        only reach the predicate if the non-strict value space maps 'prod' back onto 'env'.
        """

        class DeploymentFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_deploy$"
            PROPERTY_MAPPING = {
                "env": property_spec(
                    "Deployment environment",
                    strict=False,
                    allowed_values={"prod": "Production", "staging": "Staging"},
                ),
                "region": property_spec("Target region", required_when=_region_required_in_prod),
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert DeploymentFeatureGroup.match_feature_group_criteria("service__prod_deploy", Options()) is False, (
            "prod deployments require a region, which required_when can only know via the value space"
        )
        assert (
            DeploymentFeatureGroup.match_feature_group_criteria(
                "service__prod_deploy", Options(context={"region": "eu"})
            )
            is True
        )
        assert DeploymentFeatureGroup.match_feature_group_criteria("service__staging_deploy", Options()) is True, (
            "staging needs no region"
        )

    def test_non_strict_spec_still_accepts_values_outside_its_value_space(self) -> None:
        """A non-strict value space is a mapping aid, not an enforcement: it never rejects."""

        class UnenforcedValueSpaceFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "env": property_spec(
                    "Deployment environment",
                    strict=False,
                    allowed_values={"prod": "Production", "staging": "Staging"},
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            UnenforcedValueSpaceFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"env": "dev"})
            )
            is True
        )

    def test_property_spec_still_rejects_element_validator_without_strict(self) -> None:
        """The OTHER never-enforced-without-strict rule is true, and stays.

        ``element_validator`` really is dead without strict: ``_validate_property_value`` returns
        early when the spec is not strict. Only the ``allowed_values`` premise was false.
        """
        with pytest.raises(ValueError, match="element_validator"):
            property_spec("Window size", strict=False, element_validator=_positive_int)

    def test_property_spec_rejects_a_str_allowed_values(self) -> None:
        """``tuple("add")`` is ``('a', 'd', 'd')``: the builder must not silently do that."""
        with pytest.raises(ValueError, match="allowed_values"):
            property_spec("Operation", strict=True, allowed_values="add")

    def test_property_spec_rejects_a_str_allowed_values_without_strict(self) -> None:
        """Now that a non-strict value space is legal, it must be shape-checked for its own sake.

        This must be rejected for its SHAPE. It is rejected today, but on the false premise that
        a non-strict ``allowed_values`` is never enforced, so the message is asserted to have
        dropped that premise.
        """
        with pytest.raises(ValueError) as exc_info:
            property_spec("Operation", strict=False, allowed_values="add")

        message = str(exc_info.value)
        del exc_info
        assert DefaultOptionKeys.allowed_values.value in message
        assert "never enforced without strict" not in message, (
            "a non-strict value space IS consumed (name-parsed value -> property key), "
            "so the only fault here is the str shape"
        )

    def test_property_spec_rejects_bytes_allowed_values(self) -> None:
        """``bytes`` substring-matches like ``str`` and is rejected the same way."""
        with pytest.raises(ValueError, match="allowed_values"):
            property_spec("Operation", strict=True, allowed_values=b"add")
