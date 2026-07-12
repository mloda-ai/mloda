"""One coherent mental model for the PROPERTY_MAPPING validation layer (issue #600).

Two decisions are pinned here.

1. Hard rename, no back-compat. The two callable-valued PROPERTY_MAPPING keys get
   names that state what they do:

   * ``element_validator`` (was ``validation_function``): runs on EACH parsed element
     after list unpacking, requires ``strict_validation=True``, and on a falsy return
     RAISES ``ValueError`` (recovered by the mixin as a non-match, surfaced to the user
     via ``_strict_validation_rejection_reason``).
   * ``match_guard`` (was ``type_validator``): runs on the RAW option value with no list
     unpacking, does NOT require ``strict_validation``, and on a falsy return simply
     means "this feature group does not match" (debug log, ``False``, no error).

   The old names are GONE: no enum members, no aliases. A spec dict still carrying the
   stale string key must FAIL LOUDLY at class-definition time rather than be silently
   ignored. The stale keys are not part of the spec schema (``PROPERTY_SPEC_KEYS``), so
   the general unknown-key rule catches them; being REMOVED keys, they get the message
   variant that names their replacement.

2. Shared default-check semantics. ``property_spec`` and
   ``FeatureChainParser.validate_property_mapping_defaults`` encoded the same rules for a
   declared default twice. There is now ONE implementation,
   ``FeatureChainParser.check_declared_default(owner, key, spec)``, that both entry points
   route through, so the "validator RAISED" vs "validator REJECTED" distinction cannot
   drift between them.
"""

from __future__ import annotations

import gc
from typing import Any
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.default_options_key import (
    PROPERTY_SPEC_KEYS,
    DefaultOptionKeys,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import property_spec

STALE_ELEMENT_VALIDATOR_KEY = "validation_function"
STALE_MATCH_GUARD_KEY = "type_validator"


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Mirrors ``test_property_mapping_default_invariant.py``: the class-definition tests
    below define FeatureGroup subclasses, which linger in ``FeatureGroup.__subclasses__()``
    until a GC cycle runs and would otherwise be seen by tests that enumerate feature groups.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


class _Recorder:
    """Callable validator/guard that records exactly what it was called with."""

    def __init__(self, verdict: bool = True) -> None:
        self.calls: list[Any] = []
        self._verdict = verdict

    def __call__(self, value: Any) -> bool:
        self.calls.append(value)
        return self._verdict


class _PositiveIntRecorder:
    """Element validator: accepts positive ints, records every element it saw."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, value: Any) -> bool:
        self.calls.append(value)
        return isinstance(value, int) and value > 0


class _ListOfStringsRecorder:
    """Match guard: accepts a list of strings, records every raw value it saw."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, value: Any) -> bool:
        self.calls.append(value)
        return isinstance(value, list) and all(isinstance(item, str) for item in value)


class _RaisingRecorder:
    """Validator/guard that raises instead of returning a verdict."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, value: Any) -> bool:
        self.calls.append(value)
        raise TypeError(f"boom for {value!r}")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _boom(value: Any) -> bool:
    raise RuntimeError("boom")


class TestRenamedKeys:
    """The two keys carry their new names, and the old names are gone (no aliases)."""

    def test_element_validator_member_exists(self) -> None:
        """``DefaultOptionKeys.element_validator`` exists with value ``"element_validator"``."""
        assert DefaultOptionKeys.element_validator.value == "element_validator"
        assert DefaultOptionKeys("element_validator") is DefaultOptionKeys.element_validator

    def test_match_guard_member_exists(self) -> None:
        """``DefaultOptionKeys.match_guard`` exists with value ``"match_guard"``."""
        assert DefaultOptionKeys.match_guard.value == "match_guard"
        assert DefaultOptionKeys("match_guard") is DefaultOptionKeys.match_guard

    def test_old_members_are_gone(self) -> None:
        """The old enum members no longer exist: attribute access raises ``AttributeError``."""
        with pytest.raises(AttributeError):
            getattr(DefaultOptionKeys, STALE_ELEMENT_VALIDATOR_KEY)
        with pytest.raises(AttributeError):
            getattr(DefaultOptionKeys, STALE_MATCH_GUARD_KEY)

    def test_old_values_are_not_members(self) -> None:
        """The old string values do not resolve to any member (no value alias)."""
        with pytest.raises(ValueError):
            DefaultOptionKeys(STALE_ELEMENT_VALIDATOR_KEY)
        with pytest.raises(ValueError):
            DefaultOptionKeys(STALE_MATCH_GUARD_KEY)

    def test_spec_schema_carries_the_new_members(self) -> None:
        """``PROPERTY_SPEC_KEYS`` admits the new keys, so a spec may carry them."""
        assert DefaultOptionKeys.element_validator in PROPERTY_SPEC_KEYS
        assert DefaultOptionKeys.match_guard in PROPERTY_SPEC_KEYS

    def test_spec_schema_drops_the_old_strings(self) -> None:
        """The stale string keys are not part of the spec schema, so they are unknown keys."""
        assert STALE_ELEMENT_VALIDATOR_KEY not in PROPERTY_SPEC_KEYS
        assert STALE_MATCH_GUARD_KEY not in PROPERTY_SPEC_KEYS


class TestRenamedInternalHelpers:
    """The internal helpers follow the rename, so the vocabulary is consistent end to end."""

    def test_parser_exposes_get_element_validator(self) -> None:
        """``FeatureChainParser._get_element_validator`` replaces ``_get_validation_function``."""
        assert not hasattr(FeatureChainParser, "_get_validation_function")

        validator = _positive_int
        spec = {DefaultOptionKeys.element_validator: validator}
        assert FeatureChainParser._get_element_validator(spec) is validator
        assert FeatureChainParser._get_element_validator({}) is None

    def test_mixin_exposes_validate_match_guards(self) -> None:
        """``FeatureChainParserMixin._validate_match_guards`` replaces ``_validate_type_validators``."""
        assert not hasattr(FeatureChainParserMixin, "_validate_type_validators")
        assert hasattr(FeatureChainParserMixin, "_validate_match_guards")


class TestStaleKeysFailLoudly:
    """A spec dict still carrying a stale key is rejected at class definition, never honored.

    This is NOT backwards compatibility. A stale key is simply not in the spec schema, so the
    general unknown-key rule rejects it; because it is a REMOVED key, the error names the
    replacement instead of guessing at one. Silently ignoring it would leave the spec claiming a
    validation it does not perform, so the rule converts that into an actionable migration error.
    """

    def test_stale_validation_function_key_rejected_at_class_definition(self) -> None:
        """A PROPERTY_MAPPING carrying ``"validation_function"`` raises, naming ``element_validator``."""
        with pytest.raises(ValueError) as exc_info:

            class StaleValidationFunctionFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
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
        assert "StaleValidationFunctionFeatureGroup" in message
        assert "operation_type" in message
        assert STALE_ELEMENT_VALIDATOR_KEY in message
        assert "element_validator" in message

    def test_stale_type_validator_key_rejected_at_class_definition(self) -> None:
        """A PROPERTY_MAPPING carrying ``"type_validator"`` raises, naming ``match_guard``."""
        with pytest.raises(ValueError) as exc_info:

            class StaleTypeValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
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
        assert "StaleTypeValidatorFeatureGroup" in message
        assert "partition_by" in message
        assert STALE_MATCH_GUARD_KEY in message
        assert "match_guard" in message

    def test_stale_key_is_reported_ahead_of_other_unknown_keys(self) -> None:
        """A spec with several unknown keys still names the stale one: it has the exact remedy."""
        with pytest.raises(ValueError) as exc_info:

            class MultiOffenderStaleFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
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
        assert "element_validator" in message

    def test_parser_entry_point_rejects_stale_spec(self) -> None:
        """The guard lives in core, so validating a stale mapping directly also raises."""
        mapping: dict[str, Any] = {
            "operation_type": {
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                STALE_ELEMENT_VALIDATOR_KEY: _positive_int,
            }
        }

        with pytest.raises(ValueError, match="element_validator"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", mapping)

    def test_stale_callable_is_never_absorbed_into_the_accepted_value_set(self) -> None:
        """The regression this guard exists to prevent: the stale callable as an allowed VALUE.

        A spec is not always reached through class definition, so matching is checked too. The
        stale callable must never end up a member of the accepted set, which is what recovering
        the value space by subtraction used to do. Matching such a spec must never succeed.
        """
        validator = _positive_int
        mapping: dict[str, Any] = {
            "operation_type": {
                "add": "Addition",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                STALE_ELEMENT_VALIDATOR_KEY: validator,
            }
        }

        try:
            matched = FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": validator}), mapping
            )
        except ValueError:
            matched = False

        assert matched is False


class TestElementValidatorSemantics:
    """``element_validator``: per element, strict-only, falsy -> ``ValueError``."""

    def test_runs_on_each_element_after_list_unpacking(self) -> None:
        """A list option is unpacked: the validator sees each element, never the list."""
        validator = _PositiveIntRecorder()

        class ElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Sizes of time windows",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: validator,
                }
            }

        options = Options(context={"window_size": [1, 2, 3]})

        assert ElementFeatureGroup.match_feature_group_criteria("any_feature", options) is True
        assert set(validator.calls) == {1, 2, 3}

    def test_rejected_element_is_a_non_match(self) -> None:
        """A falsy verdict on any element makes the whole feature group not match."""
        validator = _PositiveIntRecorder()

        class RejectingElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Sizes of time windows",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: validator,
                }
            }

        options = Options(context={"window_size": [1, -3]})

        assert RejectingElementFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert validator.calls, "the element_validator must have been called"
        assert all(not isinstance(call, list) for call in validator.calls), "elements, not the raw list"

    def test_rejection_raises_value_error_in_the_parser(self) -> None:
        """The rejection is a ``ValueError`` at the parser seam (the mixin recovers it)."""
        mapping: dict[str, Any] = {
            "window_size": {
                "explanation": "Size of time window",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.element_validator: _positive_int,
            }
        }

        with pytest.raises(ValueError, match="window_size"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"window_size": -3}), mapping
            )

    def test_rejection_is_surfaced_via_strict_validation_rejection_reason(self) -> None:
        """The discarded ``ValueError`` message reaches the user diagnostic hook."""

        class SurfacedElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Size of time window",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: _positive_int,
                }
            }

        options = Options(context={"window_size": -3})

        assert SurfacedElementFeatureGroup.match_feature_group_criteria("any_feature", options) is False

        reason = SurfacedElementFeatureGroup._strict_validation_rejection_reason("any_feature", options)

        assert reason is not None
        assert "window_size" in reason
        assert "-3" in reason

    def test_not_enforced_without_strict_validation(self) -> None:
        """``element_validator`` is strict-only: without ``strict_validation`` it never runs."""
        validator = _PositiveIntRecorder()

        class NonStrictElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Size of time window",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.element_validator: validator,
                }
            }

        options = Options(context={"window_size": -3})

        assert NonStrictElementFeatureGroup.match_feature_group_criteria("any_feature", options) is True
        assert validator.calls == []


class TestMatchGuardSemantics:
    """``match_guard``: raw whole value, no strict requirement, falsy -> non-match, no error."""

    def test_runs_on_the_raw_value_without_strict_validation(self) -> None:
        """The guard sees the raw list, un-unpacked, with ``strict_validation`` off."""
        guard = _ListOfStringsRecorder()

        class GuardedFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "partition_by": {
                    "explanation": "List of columns to partition by",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.match_guard: guard,
                }
            }

        options = Options(context={"partition_by": ["region", "category"]})

        assert GuardedFeatureGroup.match_feature_group_criteria("any_feature", options) is True
        assert guard.calls == [["region", "category"]]

    def test_falsy_guard_is_a_silent_non_match(self) -> None:
        """A falsy verdict returns ``False`` from ``match_feature_group_criteria``, it does not raise."""
        guard = _ListOfStringsRecorder()

        class RejectingGuardFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "partition_by": {
                    "explanation": "List of columns to partition by",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.match_guard: guard,
                }
            }

        options = Options(context={"partition_by": "region"})

        assert RejectingGuardFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert guard.calls == ["region"]

    def test_falsy_guard_is_not_a_strict_validation_rejection(self) -> None:
        """A guard non-match is not an option-value rejection, so it has no rejection reason."""

        class QuietGuardFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "partition_by": {
                    "explanation": "List of columns to partition by",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.match_guard: _ListOfStringsRecorder(),
                }
            }

        options = Options(context={"partition_by": "region"})

        assert QuietGuardFeatureGroup._strict_validation_rejection_reason("any_feature", options) is None

    def test_raising_guard_is_caught_and_treated_as_non_match(self) -> None:
        """A guard that raises is caught: non-match, no exception escapes."""
        guard = _RaisingRecorder()

        class RaisingGuardFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "items": {
                    "explanation": "A list of items",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.match_guard: guard,
                }
            }

        options = Options(context={"items": "not_a_list"})

        assert RaisingGuardFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert guard.calls == ["not_a_list"]


class TestElementValidatorAndMatchGuardPrecedence:
    """With both on one entry, the element_validator runs first and its rejection wins."""

    def test_element_validator_rejection_short_circuits_the_match_guard(self) -> None:
        """A rejected element fails the match before the guard is ever consulted."""
        validator = _PositiveIntRecorder()
        guard = _Recorder(verdict=True)

        class BothFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Size of time window",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: validator,
                    DefaultOptionKeys.match_guard: guard,
                }
            }

        options = Options(context={"window_size": -3})

        assert BothFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert validator.calls == [-3]
        assert guard.calls == [], "the match_guard must not be reached once an element was rejected"

    def test_match_guard_runs_when_every_element_passes(self) -> None:
        """When the element_validator accepts, the guard still gets the raw value."""
        validator = _PositiveIntRecorder()
        guard = _Recorder(verdict=False)

        class BothPassFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": {
                    "explanation": "Size of time window",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: validator,
                    DefaultOptionKeys.match_guard: guard,
                }
            }

        options = Options(context={"window_size": [1, 2]})

        assert BothPassFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert set(validator.calls) == {1, 2}
        assert guard.calls == [[1, 2]], "the match_guard sees the raw, un-unpacked value"


class TestCheckDeclaredDefaultSeam:
    """``FeatureChainParser.check_declared_default`` is the ONE default-check implementation."""

    def test_is_a_no_op_for_a_valid_declared_default(self) -> None:
        """A default the spec honors passes silently."""
        spec: dict[str, Any] = {
            DefaultOptionKeys.allowed_values: {"add": "Addition"},
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "add",
        }

        assert FeatureChainParser.check_declared_default("Owner", "operation_type", spec) is None

    def test_raises_for_a_default_outside_the_accepted_set(self) -> None:
        """A default outside the accepted set raises, naming the owner and the key."""
        spec: dict[str, Any] = {
            DefaultOptionKeys.allowed_values: {"add": "Addition"},
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "mul",
        }

        with pytest.raises(ValueError) as exc_info:
            FeatureChainParser.check_declared_default("Owner", "operation_type", spec)

        message = str(exc_info.value)
        del exc_info
        assert "Owner" in message
        assert "operation_type" in message
        assert "mul" in message

    def test_feature_group_definition_routes_through_the_seam(self) -> None:
        """``FeatureGroup.__init_subclass__`` delegates its default check to the shared seam."""
        spec = {
            DefaultOptionKeys.allowed_values: {"add": "Addition"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "add",
        }

        with patch.object(FeatureChainParser, "check_declared_default") as spy:

            class RoutedFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"operation_type": spec}

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        assert spy.call_count == 1
        passed = list(spy.call_args.args) + list(spy.call_args.kwargs.values())
        assert "operation_type" in passed
        assert spec in passed

    def test_property_spec_routes_through_the_seam(self) -> None:
        """``property_spec`` delegates its default check to the same shared seam."""
        with patch.object(FeatureChainParser, "check_declared_default") as spy:
            property_spec("op", strict=True, allowed_values={"add": "Addition"}, default="add")

        assert spy.call_count == 1
        passed = list(spy.call_args.args) + list(spy.call_args.kwargs.values())
        assert any(isinstance(arg, str) and "property_spec" in arg for arg in passed), (
            "property_spec must pass an owner label identifying itself"
        )

    def test_property_spec_owner_label_carries_the_explanation(self) -> None:
        """The shared error message identifies the offending ``property_spec`` call."""
        with pytest.raises(ValueError) as exc_info:
            property_spec("op", strict=True, allowed_values={"add": "Addition"}, default="mul")

        message = str(exc_info.value)
        del exc_info
        assert "property_spec" in message
        assert "op" in message
        assert "mul" in message


class TestRaisedVersusRejectedThroughBothEntryPoints:
    """The raised/rejected distinction is identical through both entry points, by construction."""

    def test_class_definition_reports_a_raising_element_validator_as_raised(self) -> None:
        """A validator that errors on the default reports "raised", chaining the original."""
        with pytest.raises(ValueError) as exc_info:

            class RaisingFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.element_validator: _boom,
                        DefaultOptionKeys.default: "x",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        cause_is_runtime_error = isinstance(exc_info.value.__cause__, RuntimeError)
        del exc_info
        assert "RaisingFnFeatureGroup" in message
        assert "raised" in message
        assert "rejected by" not in message
        assert cause_is_runtime_error, "expected the original RuntimeError chained as __cause__"

    def test_class_definition_reports_a_rejecting_element_validator_as_rejected(self) -> None:
        """A validator that runs and returns falsy reports "rejected by"."""
        with pytest.raises(ValueError) as exc_info:

            class RejectingFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.element_validator: _positive_int,
                        DefaultOptionKeys.default: -1,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "RejectingFnFeatureGroup" in message
        assert "rejected by" in message
        assert "raised" not in message

    def test_property_spec_reports_a_raising_element_validator_as_raised(self) -> None:
        """The same wording and the same chaining, through the ``property_spec`` entry point."""
        with pytest.raises(ValueError) as exc_info:
            property_spec("op", strict=True, element_validator=_boom, default=5)

        message = str(exc_info.value)
        cause_is_runtime_error = isinstance(exc_info.value.__cause__, RuntimeError)
        del exc_info
        assert "raised" in message
        assert "rejected by" not in message
        assert cause_is_runtime_error, "expected the original RuntimeError chained as __cause__"

    def test_property_spec_reports_a_rejecting_element_validator_as_rejected(self) -> None:
        """The same wording for a genuine rejection, through the ``property_spec`` entry point."""
        with pytest.raises(ValueError) as exc_info:
            property_spec("op", strict=True, element_validator=_positive_int, default=-1)

        message = str(exc_info.value)
        del exc_info
        assert "rejected by" in message
        assert "raised" not in message


class TestPropertySpecEmitsTheNewKeys:
    """``property_spec`` speaks the new vocabulary: new kwargs in, new keys out."""

    def test_emits_element_validator_key(self) -> None:
        """``element_validator=`` lands under ``DefaultOptionKeys.element_validator``."""
        spec = property_spec("op", strict=True, element_validator=_positive_int)

        assert spec[DefaultOptionKeys.element_validator] is _positive_int
        assert spec[DefaultOptionKeys.strict_validation] is True

    def test_emits_match_guard_key(self) -> None:
        """``match_guard=`` lands under ``DefaultOptionKeys.match_guard``, no strict needed."""
        guard = _ListOfStringsRecorder()
        spec = property_spec("cols", match_guard=guard)

        assert spec[DefaultOptionKeys.match_guard] is guard
        assert spec[DefaultOptionKeys.strict_validation] is False

    def test_omitted_new_keys_are_absent(self) -> None:
        """An omitted validator/guard is absent, never present-and-``None``."""
        spec = property_spec("op")

        assert DefaultOptionKeys.element_validator not in spec
        assert DefaultOptionKeys.match_guard not in spec

    def test_stale_kwargs_are_gone(self) -> None:
        """The old keyword arguments no longer exist: passing them fails loudly."""
        with pytest.raises((TypeError, ValueError)):
            property_spec("op", strict=True, validation_function=_positive_int)  # type: ignore[call-arg]
        with pytest.raises((TypeError, ValueError)):
            property_spec("op", type_validator=_positive_int)  # type: ignore[call-arg]

    def test_authoring_invariants_still_live_in_property_spec(self) -> None:
        """The authoring-time-only invariants stay put; only the default check moved to core."""
        with pytest.raises(ValueError):
            property_spec("op", strict=True)
        with pytest.raises(ValueError):
            property_spec("op", allowed_values={"add": "Addition"})
        with pytest.raises(ValueError):
            property_spec("op", strict=True, element_validator=_positive_int, allowed_values=[])
        not_callable: Any = "not callable"
        with pytest.raises(ValueError):
            property_spec("op", strict=True, element_validator=not_callable)
        with pytest.raises(ValueError):
            property_spec("op", match_guard=not_callable)
