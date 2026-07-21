"""The ``PropertySpec`` dataclass is the ONE model for PROPERTY_MAPPING validation (issue #694).

The unified-model decisions from issue #600 survive the dict-spec deletion, re-anchored on the
type:

1. The two callable validators are FIELDS of ``PropertySpec``, under the names that state what
   they do:

   * ``element_validator``: runs on EACH parsed element after list unpacking, requires
     ``strict_validation=True`` (enforced at construction, so the old tolerance for a
     non-strict spec carrying an unenforced validator is unexpressible now), and on a falsy
     return raises ``ValueError`` (recovered by the mixin as a non-match, surfaced to the user
     via ``_strict_validation_rejection_reason``).
   * ``match_guard``: runs on the RAW option value with no list unpacking, needs no strict
     flag, and on a falsy return simply means "this feature group does not match" (debug log,
     ``False``, no error).

   The pre-rename names are GONE: ``validation_function`` and ``type_validator`` are unknown
   constructor keywords (``TypeError``), through ``PropertySpec`` and the ``property_spec``
   builder alike, and the internal helpers follow the field names.

2. The declared-default check has exactly ONE implementation: the ``PropertySpec``
   constructor. Direct construction and the ``property_spec`` builder raise the identical
   message for the same bad default, prefixed ``PropertySpec('<explanation>')`` so the
   offending spec is identifiable without an owning class.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import PropertySpec, property_spec

STALE_ELEMENT_VALIDATOR_NAME = "validation_function"
STALE_MATCH_GUARD_NAME = "type_validator"


def _construct(*args: Any, **kwargs: Any) -> PropertySpec:
    """Call ``PropertySpec`` through an untyped seam, keeping the stale-keyword tests mypy-clean."""
    return PropertySpec(*args, **kwargs)


def _build(*args: Any, **kwargs: Any) -> PropertySpec:
    """Call ``property_spec`` through an untyped seam, keeping the stale-keyword tests mypy-clean."""
    return property_spec(*args, **kwargs)


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


class TestValidatorFieldNames:
    """``element_validator`` and ``match_guard`` are the fields; the stale names are gone."""

    def test_element_validator_and_match_guard_are_property_spec_fields(self) -> None:
        """Both validators live as first-class fields on the spec."""
        spec = PropertySpec(
            "Size of the time window",
            strict_validation=True,
            element_validator=_positive_int,
            match_guard=_positive_int,
        )

        assert spec.element_validator is _positive_int
        assert spec.match_guard is _positive_int

    def test_stale_validation_function_name_is_a_type_error(self) -> None:
        """``validation_function`` is an unknown keyword on the constructor and the builder."""
        with pytest.raises(TypeError):
            _construct("op", strict_validation=True, validation_function=_positive_int)
        with pytest.raises(TypeError):
            _build("op", strict=True, validation_function=_positive_int)

    def test_stale_type_validator_name_is_a_type_error(self) -> None:
        """``type_validator`` is an unknown keyword on the constructor and the builder."""
        with pytest.raises(TypeError):
            _construct("cols", type_validator=_positive_int)
        with pytest.raises(TypeError):
            _build("cols", type_validator=_positive_int)


class TestRenamedInternalHelpers:
    """The internal helpers follow the field names, so the vocabulary is consistent end to end."""

    def test_element_validator_is_read_from_the_field(self) -> None:
        """``PropertySpec.element_validator`` is the field callers read; the old getter name is gone."""
        assert not hasattr(FeatureChainParser, f"_get_{STALE_ELEMENT_VALIDATOR_NAME}")

        spec = PropertySpec("Size of the time window", strict_validation=True, element_validator=_positive_int)
        assert spec.element_validator is _positive_int
        assert PropertySpec("x").element_validator is None

    def test_mixin_exposes_validate_match_guards(self) -> None:
        """``FeatureChainParserMixin._validate_match_guards`` replaces ``_validate_type_validators``."""
        assert not hasattr(FeatureChainParserMixin, "_validate_type_validators")
        assert hasattr(FeatureChainParserMixin, "_validate_match_guards")


class TestElementValidatorSemantics:
    """``element_validator``: per element, strict-only by construction, falsy -> ``ValueError``."""

    def test_runs_on_each_element_after_list_unpacking(self) -> None:
        """A list option is unpacked: the validator sees each element, never the list."""
        validator = _PositiveIntRecorder()

        class Unified694ElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": PropertySpec(
                    "Sizes of time windows",
                    context=True,
                    strict_validation=True,
                    element_validator=validator,
                )
            }

        options = Options(context={"window_size": [1, 2, 3]})

        assert Unified694ElementFeatureGroup.match_feature_group_criteria("any_feature", options) is True
        assert set(validator.calls) == {1, 2, 3}

    def test_rejected_element_is_a_non_match(self) -> None:
        """A falsy verdict on any element makes the whole feature group not match."""
        validator = _PositiveIntRecorder()

        class Unified694RejectingElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": PropertySpec(
                    "Sizes of time windows",
                    context=True,
                    strict_validation=True,
                    element_validator=validator,
                )
            }

        options = Options(context={"window_size": [1, -3]})

        assert Unified694RejectingElementFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert validator.calls, "the element_validator must have been called"
        assert all(not isinstance(call, list) for call in validator.calls), "elements, not the raw list"

    def test_rejection_raises_value_error_in_the_parser(self) -> None:
        """The rejection is a ``ValueError`` at the parser seam (the mixin recovers it)."""
        mapping: dict[str, PropertySpec] = {
            "window_size": PropertySpec(
                "Size of time window",
                context=True,
                strict_validation=True,
                element_validator=_positive_int,
            )
        }

        with pytest.raises(ValueError, match="window_size"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"window_size": -3}), mapping
            )

    def test_rejection_is_surfaced_via_strict_validation_rejection_reason(self) -> None:
        """The discarded ``ValueError`` message reaches the user diagnostic hook."""

        class Unified694SurfacedElementFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": PropertySpec(
                    "Size of time window",
                    context=True,
                    strict_validation=True,
                    element_validator=_positive_int,
                )
            }

        options = Options(context={"window_size": -3})

        assert Unified694SurfacedElementFeatureGroup.match_feature_group_criteria("any_feature", options) is False

        reason = Unified694SurfacedElementFeatureGroup._strict_validation_rejection_reason("any_feature", options)

        assert reason is not None
        assert "window_size" in reason
        assert "-3" in reason


class TestMatchGuardSemantics:
    """``match_guard``: raw whole value, no strict requirement, falsy -> non-match, no error."""

    def test_runs_on_the_raw_value_without_strict_validation(self) -> None:
        """The guard sees the raw list, un-unpacked, with ``strict_validation`` off."""
        guard = _ListOfStringsRecorder()

        class Unified694GuardedFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "partition_by": PropertySpec(
                    "List of columns to partition by",
                    context=True,
                    strict_validation=False,
                    match_guard=guard,
                )
            }

        options = Options(context={"partition_by": ["region", "category"]})

        assert Unified694GuardedFeatureGroup.match_feature_group_criteria("any_feature", options) is True
        assert guard.calls == [["region", "category"]]

    def test_falsy_guard_is_a_silent_non_match(self) -> None:
        """A falsy verdict returns ``False`` from ``match_feature_group_criteria``, it does not raise."""
        guard = _ListOfStringsRecorder()

        class Unified694RejectingGuardFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "partition_by": PropertySpec(
                    "List of columns to partition by",
                    context=True,
                    strict_validation=False,
                    match_guard=guard,
                )
            }

        options = Options(context={"partition_by": "region"})

        assert Unified694RejectingGuardFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert guard.calls == ["region"]

    def test_falsy_guard_is_not_a_strict_validation_rejection(self) -> None:
        """A guard non-match is not an option-value rejection, so it has no rejection reason."""

        class Unified694QuietGuardFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "partition_by": PropertySpec(
                    "List of columns to partition by",
                    context=True,
                    strict_validation=False,
                    match_guard=_ListOfStringsRecorder(),
                )
            }

        options = Options(context={"partition_by": "region"})

        assert Unified694QuietGuardFeatureGroup._strict_validation_rejection_reason("any_feature", options) is None

    def test_raising_guard_is_caught_and_treated_as_non_match(self) -> None:
        """A guard that raises is caught: non-match, no exception escapes."""
        guard = _RaisingRecorder()

        class Unified694RaisingGuardFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "items": PropertySpec(
                    "A list of items",
                    context=True,
                    strict_validation=False,
                    match_guard=guard,
                )
            }

        options = Options(context={"items": "not_a_list"})

        assert Unified694RaisingGuardFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert guard.calls == ["not_a_list"]


class TestElementValidatorAndMatchGuardPrecedence:
    """With both on one entry, the element_validator runs first and its rejection wins."""

    def test_element_validator_rejection_short_circuits_the_match_guard(self) -> None:
        """A rejected element fails the match before the guard is ever consulted."""
        validator = _PositiveIntRecorder()
        guard = _Recorder(verdict=True)

        class Unified694BothFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": PropertySpec(
                    "Size of time window",
                    context=True,
                    strict_validation=True,
                    element_validator=validator,
                    match_guard=guard,
                )
            }

        options = Options(context={"window_size": -3})

        assert Unified694BothFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert validator.calls == [-3]
        assert guard.calls == [], "the match_guard must not be reached once an element was rejected"

    def test_match_guard_runs_when_every_element_passes(self) -> None:
        """When the element_validator accepts, the guard still gets the raw value."""
        validator = _PositiveIntRecorder()
        guard = _Recorder(verdict=False)

        class Unified694BothPassFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "window_size": PropertySpec(
                    "Size of time window",
                    context=True,
                    strict_validation=True,
                    element_validator=validator,
                    match_guard=guard,
                )
            }

        options = Options(context={"window_size": [1, 2]})

        assert Unified694BothPassFeatureGroup.match_feature_group_criteria("any_feature", options) is False
        assert set(validator.calls) == {1, 2}
        assert guard.calls == [[1, 2]], "the match_guard sees the raw, un-unpacked value"


class TestOneDefaultCheckImplementation:
    """The declared-default check has one implementation: the ``PropertySpec`` constructor."""

    def test_builder_and_constructor_raise_the_identical_default_message(self) -> None:
        """Both entry points are the same code path, so the messages cannot drift."""
        with pytest.raises(ValueError) as constructor_exc:
            PropertySpec("op", allowed_values={"add": "Addition"}, strict_validation=True, default="mul")
        with pytest.raises(ValueError) as builder_exc:
            property_spec("op", strict=True, allowed_values={"add": "Addition"}, default="mul")

        assert str(constructor_exc.value) == str(builder_exc.value)

    def test_default_message_identifies_the_spec_by_its_explanation(self) -> None:
        """The ``PropertySpec('...')`` prefix identifies the offender; no owner label needed."""
        with pytest.raises(ValueError, match=r"PropertySpec\('op'\)") as exc_info:
            property_spec("op", strict=True, allowed_values={"add": "Addition"}, default="mul")

        message = str(exc_info.value)
        assert "default" in message
        assert "mul" in message
