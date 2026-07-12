"""A PROPERTY_MAPPING spec has a SCHEMA: unknown keys are errors, never allowed values (issue #600).

The legacy flattened authoring form is retired here. It recovered a spec's allowed-value
space by SUBTRACTING the reserved metadata keys, which means every key the parser did not
recognize was silently absorbed as an accepted VALUE. One typo, two silent failures::

    {"add": "Addition", "sub": "Subtraction", "strict_validaton": True}   # missing an 'i'

    _extract_property_values(...) -> {"add": ..., "sub": ..., "strict_validaton": True}
                                     the typo'd FLAG is now an accepted VALUE
    _is_strict_validation(...)    -> False
                                     strict validation is silently OFF

Nothing raised, at any lifecycle moment. This module pins the hard break that makes that
impossible to express:

1. A spec's value space is DECLARED under ``DefaultOptionKeys.allowed_values``. It is never
   inferred by subtraction, so a spec without ``allowed_values`` declares an EMPTY value
   space (and, being non-strict, accepts anything).
2. Because the value space is explicit, every remaining key in a spec must be a known spec
   key. ONE general rule at class-definition time rejects any unknown key, naming the owning
   class, the property key and the offender. For a REMOVED key the message names its
   replacement; otherwise it suggests the nearest known key.
3. The known-key set is the spec SCHEMA now, not a subtraction set: ``PROPERTY_SPEC_KEYS``
   replaces ``RESERVED_PROPERTY_KEYS`` (hard rename, no alias).
4. Now checkable: ``strict_validation: True`` requires a non-empty ``allowed_values`` OR an
   ``element_validator``. Strict with neither accepts nothing and rejects every value. This
   mirrors the invariant ``property_spec`` already enforces, so core and the builder agree.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components import default_options_key as default_options_key_module
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup

TYPO_STRICT_KEY = "strict_validaton"  # the headline typo: 'strict_validation' minus one 'i'
STALE_ELEMENT_VALIDATOR_KEY = "validation_function"
STALE_MATCH_GUARD_KEY = "type_validator"


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Mirrors ``test_property_mapping_unified_model.py``: the class-definition tests below
    define FeatureGroup subclasses, which linger in ``FeatureGroup.__subclasses__()`` until
    a GC cycle runs and would otherwise be seen by tests that enumerate feature groups.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


class TestSpecSchemaKeySet:
    """The known-key set is the spec SCHEMA: ``PROPERTY_SPEC_KEYS``, and only that name."""

    def test_property_spec_keys_exists_and_is_complete(self) -> None:
        """``PROPERTY_SPEC_KEYS`` is an importable frozenset holding every known spec key."""
        spec_keys = getattr(default_options_key_module, "PROPERTY_SPEC_KEYS", None)

        assert spec_keys is not None, "PROPERTY_SPEC_KEYS must exist in default_options_key"
        assert isinstance(spec_keys, frozenset)
        for member in (
            DefaultOptionKeys.allowed_values,
            DefaultOptionKeys.default,
            DefaultOptionKeys.context,
            DefaultOptionKeys.group,
            DefaultOptionKeys.strict_validation,
            DefaultOptionKeys.element_validator,
            DefaultOptionKeys.required_when,
            DefaultOptionKeys.match_guard,
        ):
            assert member in spec_keys
        assert "explanation" in spec_keys

    def test_reserved_property_keys_is_gone(self) -> None:
        """The subtraction-set name is retired: no alias, no re-export."""
        assert not hasattr(default_options_key_module, "RESERVED_PROPERTY_KEYS"), (
            "RESERVED_PROPERTY_KEYS is renamed to PROPERTY_SPEC_KEYS, with no alias"
        )


class TestUnknownSpecKeyFailsAtClassDefinition:
    """The headline: a typo'd flag can no longer be expressed silently.

    Any key that is not a known spec key raises at class definition, naming the owning
    class, the property key and the offending key.
    """

    def test_typo_flag_key_rejected_at_class_definition(self) -> None:
        """``strict_validaton`` (one character off) raises, and the message suggests the real key."""
        with pytest.raises(ValueError) as exc_info:

            class TypoFlagFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "explanation": "The arithmetic operation to apply",
                        DefaultOptionKeys.allowed_values: {"add": "Addition", "sub": "Subtraction"},
                        DefaultOptionKeys.context: True,
                        TYPO_STRICT_KEY: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "TypoFlagFeatureGroup" in message
        assert "operation_type" in message
        assert TYPO_STRICT_KEY in message
        assert DefaultOptionKeys.strict_validation.value in message, (
            "the message must name the key the author meant (nearest-known-key suggestion)"
        )

    def test_arbitrary_unknown_key_rejected_at_class_definition(self) -> None:
        """An unknown key with no near miss still raises, naming class, property and offender."""
        with pytest.raises(ValueError) as exc_info:

            class UnknownKeyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "window_size": {
                        "explanation": "Size of the time window",
                        DefaultOptionKeys.context: True,
                        "documentation_url": "https://example.invalid/window_size",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "UnknownKeyFeatureGroup" in message
        assert "window_size" in message
        assert "documentation_url" in message

    def test_legacy_flattened_value_entries_are_unknown_keys(self) -> None:
        """The legacy flattened form is retired: inline value entries are now unknown keys."""
        with pytest.raises(ValueError) as exc_info:

            class LegacyFlattenedFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "LegacyFlattenedFeatureGroup" in message
        assert "operation_type" in message
        assert "add" in message
        assert DefaultOptionKeys.allowed_values.value in message, (
            "the remedy is to declare the value space under allowed_values"
        )

    def test_parser_entry_point_rejects_unknown_key(self) -> None:
        """The rule lives in core, so validating a mapping directly raises the same way."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.allowed_values: {"add": "Addition"},
                DefaultOptionKeys.context: True,
                TYPO_STRICT_KEY: True,
            }
        }

        with pytest.raises(ValueError) as exc_info:
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

        message = str(exc_info.value)
        del exc_info
        assert "SomeOwner" in message
        assert TYPO_STRICT_KEY in message

    def test_typo_flag_is_never_absorbed_into_the_accepted_value_set(self) -> None:
        """The regression this whole change exists to prevent, checked through MATCHING.

        Under the old subtraction rule the typo'd flag became a member of the accepted set,
        so a feature whose option value is literally ``"strict_validaton"`` matched, while
        strict validation was silently off. No feature group may ever accept that.
        """
        matched: bool

        try:

            class AbsorbedTypoFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        TYPO_STRICT_KEY: True,
                        DefaultOptionKeys.context: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

            matched = AbsorbedTypoFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": TYPO_STRICT_KEY})
            )
        except ValueError:
            matched = False

        assert matched is False, "an unknown spec key must never become an accepted option value"


class TestExtractPropertyValuesReadsAllowedValuesOnly:
    """``_extract_property_values`` returns exactly what ``allowed_values`` declares."""

    def test_flattened_spec_no_longer_widens_the_value_space(self) -> None:
        """No ``allowed_values`` means an EMPTY value space, never the leftover entries.

        Unreachable through class definition now, so the parser seam is tested directly.
        """
        spec: dict[Any, Any] = {
            "add": "Addition",
            "sub": "Subtraction",
            DefaultOptionKeys.context: True,
        }

        extracted = FeatureChainParser._extract_property_values(spec)

        assert not extracted, "a spec with no allowed_values declares no value space"
        assert "add" not in extracted
        assert "sub" not in extracted

    def test_stray_unknown_key_does_not_widen_a_declared_value_space(self) -> None:
        """With ``allowed_values`` declared, a stray key is never appended to the value space."""
        spec: dict[Any, Any] = {
            "explanation": "The arithmetic operation to apply",
            DefaultOptionKeys.allowed_values: {"add": "Addition"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            TYPO_STRICT_KEY: True,
        }

        extracted = FeatureChainParser._extract_property_values(spec)

        assert extracted == {"add": "Addition"}
        assert TYPO_STRICT_KEY not in extracted

    def test_empty_value_space_is_returned_for_a_spec_without_allowed_values(self) -> None:
        """A metadata-only spec (the ``in_features`` shape) declares no value space."""
        spec: dict[Any, Any] = {
            "explanation": "The input features",
            DefaultOptionKeys.context: True,
        }

        assert not FeatureChainParser._extract_property_values(spec)


class TestKnownKeySpecsAreAccepted:
    """The valid authoring forms keep working: this rule rejects only UNKNOWN keys."""

    def test_allowed_values_plus_known_flags_defines_and_matches(self) -> None:
        """``allowed_values`` plus known flags is the canonical strict spec."""

        class ExplicitFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "operation_type": {
                    "explanation": "The arithmetic operation to apply",
                    DefaultOptionKeys.allowed_values: {"add": "Addition", "sub": "Subtraction"},
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.default: "add",
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            ExplicitFeatureGroup.match_feature_group_criteria("any_feature", Options(context={"operation_type": "sub"}))
            is True
        )
        assert (
            ExplicitFeatureGroup.match_feature_group_criteria("any_feature", Options(context={"operation_type": "mul"}))
            is False
        )

    def test_metadata_only_spec_declares_no_value_space_and_accepts_anything(self) -> None:
        """No ``allowed_values`` and no strict (the ``in_features`` shape) stays valid."""

        class MetadataOnlyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                DefaultOptionKeys.in_features: {
                    "explanation": "The input features",
                    DefaultOptionKeys.context: True,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            MetadataOnlyFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={DefaultOptionKeys.in_features.value: "anything_at_all"})
            )
            is True
        )

    def test_match_guard_only_spec_defines(self) -> None:
        """A non-strict guarded spec carries only known keys and defines cleanly."""

        class GuardedFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "partition_by": {
                    "explanation": "List of columns to partition by",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.match_guard: _positive_int,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert DefaultOptionKeys.match_guard in GuardedFeatureGroup.PROPERTY_MAPPING["partition_by"]


class TestRemovedKeysFoldIntoTheGeneralRule:
    """The removed-key guard becomes a MESSAGE VARIANT of the unknown-key rule, not a second check."""

    def test_removed_element_validator_key_names_its_replacement(self) -> None:
        """``validation_function`` is an unknown key whose message names ``element_validator``."""
        with pytest.raises(ValueError) as exc_info:

            class StaleElementValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "explanation": "The arithmetic operation to apply",
                        DefaultOptionKeys.allowed_values: {"add": "Addition"},
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
        assert "StaleElementValidatorFeatureGroup" in message
        assert "operation_type" in message
        assert STALE_ELEMENT_VALIDATOR_KEY in message
        assert DefaultOptionKeys.element_validator.value in message

    def test_removed_match_guard_key_names_its_replacement(self) -> None:
        """``type_validator`` is an unknown key whose message names ``match_guard``."""
        with pytest.raises(ValueError) as exc_info:

            class StaleMatchGuardFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "partition_by": {
                        "explanation": "List of columns to partition by",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: False,
                        STALE_MATCH_GUARD_KEY: _positive_int,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "StaleMatchGuardFeatureGroup" in message
        assert "partition_by" in message
        assert STALE_MATCH_GUARD_KEY in message
        assert DefaultOptionKeys.match_guard.value in message


class TestStrictValidationNeedsAValueSpace:
    """``strict_validation: True`` needs something to validate AGAINST.

    Strict with neither a non-empty ``allowed_values`` nor an ``element_validator`` accepts
    nothing and rejects every value: a spec that can never match. ``property_spec`` already
    rejects this at import time; core now agrees, so the two entry points cannot drift.
    """

    def test_strict_without_allowed_values_or_element_validator_raises(self) -> None:
        """Strict, but nothing to validate against: raises at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class EmptyStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "explanation": "The arithmetic operation to apply",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "EmptyStrictFeatureGroup" in message
        assert "operation_type" in message
        assert DefaultOptionKeys.allowed_values.value in message
        assert DefaultOptionKeys.element_validator.value in message

    def test_strict_with_empty_allowed_values_raises(self) -> None:
        """An empty declared value space would reject every value: raises at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class EmptyAllowedValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "explanation": "The arithmetic operation to apply",
                        DefaultOptionKeys.allowed_values: {},
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "EmptyAllowedValuesFeatureGroup" in message
        assert "operation_type" in message

    def test_parser_entry_point_rejects_strict_without_a_value_space(self) -> None:
        """The invariant lives in core, so validating a mapping directly raises too."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
            }
        }

        with pytest.raises(ValueError, match="operation_type"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    def test_strict_with_element_validator_only_is_fine(self) -> None:
        """An ``element_validator`` IS a value space: no ``allowed_values`` needed."""

        class StrictValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Size of the time window",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: _positive_int,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            StrictValidatorFeatureGroup.match_feature_group_criteria("any_feature", Options(context={"window_size": 7}))
            is True
        )

    def test_strict_with_non_empty_allowed_values_is_fine(self) -> None:
        """A non-empty declared value space is the other way to satisfy the invariant."""

        class StrictAllowedValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "operation_type": {
                    "explanation": "The arithmetic operation to apply",
                    DefaultOptionKeys.allowed_values: {"add": "Addition"},
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            StrictAllowedValuesFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": "add"})
            )
            is True
        )


class TestNonDictSpecValuesAreRejected:
    """A spec must BE a spec dict: a bare container bypasses every class-definition rule.

    ``PROPERTY_MAPPING = {"op": ["add", "sub", "strict_validaton"]}`` defines cleanly today and
    the list is handed back verbatim as the value space, because all three rules skip non-dicts.
    It is not a widening hole in the strict sense (a bare container has nowhere to put
    ``strict_validation``, so it can never be strict), but it contradicts the one-authoring-form
    contract the schema rule establishes, and it is the one shape the repo-wide guard cannot see.
    No in-repo spec is a non-dict, so this is a clean hard break, consistent with the rest.
    """

    def test_non_dict_spec_value_rejected_at_class_definition(self) -> None:
        """A bare list where a spec dict belongs raises, naming the class and the property key."""
        with pytest.raises(ValueError) as exc_info:

            class BareContainerFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"operation_type": ["add", "sub"]}

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "BareContainerFeatureGroup" in message
        assert "operation_type" in message

    def test_parser_entry_point_rejects_a_non_dict_spec_value(self) -> None:
        """The rule lives in core, so validating a mapping directly raises the same way."""
        mapping: dict[str, Any] = {"operation_type": ["add", "sub"]}

        with pytest.raises(ValueError, match="operation_type"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    def test_non_dict_spec_value_cannot_smuggle_a_typo_flag_as_an_accepted_value(self) -> None:
        """The schema rule's whole point, checked on the shape that used to skip it."""
        matched: bool

        try:

            class SmuggledTypoFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"operation_type": ["add", "sub", TYPO_STRICT_KEY]}

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

            matched = SmuggledTypoFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"operation_type": TYPO_STRICT_KEY})
            )
        except ValueError:
            matched = False

        assert matched is False, "a spec must be a spec dict, so a bare container can carry no flags at all"
