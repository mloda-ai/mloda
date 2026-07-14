"""The PROPERTY_MAPPING declared-default invariant (issue #530), relocated to construction.

For every PROPERTY_MAPPING key that declares a non-``None`` ``default`` under strict
validation, the declared default must be honored by the key's own rules: be within
``allowed_values``, or pass the ``element_validator``. The check now runs when the
``PropertySpec`` is CONSTRUCTED (issue #694), so a FeatureGroup declaring a bad default still
fails at class definition: the spec literal in the class body raises and the class never
comes into existence. The error identifies the spec by its explanation
(``PropertySpec('...')``) rather than by the class, which does not exist yet.

``required_when`` is NOT an escape hatch: it expresses a conditional requirement, so when its
predicate is False and the key is omitted, the bad default would still apply silently. The
only sound way to mark a key required is to declare NO default; mloda treats a key with no
default and no ``required_when`` as unconditionally required.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import NO_DEFAULT, PropertySpec


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    The tests below define FeatureGroup subclasses to exercise the author experience.
    Those class objects sit in reference cycles, so they linger in
    ``FeatureGroup.__subclasses__()`` until a GC cycle runs. While they linger, other
    tests that enumerate feature groups via ``get_all_subclasses(FeatureGroup)`` trip
    over them. After each test we force a collection to reclaim the now-unreferenced
    classes and assert that none of this module's classes remain registered, pinning
    the no-pollution contract.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _spec(*args: Any, **kwargs: Any) -> PropertySpec:
    """Build a ``PropertySpec`` through an untyped seam.

    The buggy zero-argument element_validator below is intentionally type-invalid; routing
    it through ``Any`` keeps the module mypy --strict clean.
    """
    return PropertySpec(*args, **kwargs)


def _never_required(options: Any) -> bool:
    """Conditional-requirement predicate that is never satisfied.

    With such a predicate the key is not required, so a bad default would still
    apply silently when the key is omitted: ``required_when`` must therefore not
    exempt a strict, non-None default from the invariant.
    """
    return False


class TestStrictEnumeratedDefaultInvariant:
    """Strict validation with an enumerated accepted set."""

    def test_rejects_strict_default_outside_mapping_keys(self) -> None:
        """A strict default outside a Mapping value space's KEYS raises at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class BadDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The arithmetic operation to apply",
                        allowed_values={"add": "Addition", "sub": "Subtraction"},
                        context=True,
                        strict_validation=True,
                        default="mul",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "default" in message
        assert "mul" in message
        assert all(c.__name__ != "BadDefaultFeatureGroup" for c in get_all_subclasses(FeatureGroup)), (
            "the spec raised inside the class body, so the class must never come into existence"
        )

    def test_rejects_strict_default_outside_tuple_allowed_values(self) -> None:
        """The invariant covers a tuple value space exactly like a Mapping one."""
        with pytest.raises(ValueError) as exc_info:

            class BadTupleDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The arithmetic operation to apply",
                        allowed_values=("add", "sub"),
                        context=True,
                        strict_validation=True,
                        default="mul",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "default" in message
        assert "mul" in message

    def test_accepts_strict_default_in_accepted_set(self) -> None:
        """A strict key whose default is in the accepted set defines without error."""

        class GoodDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "The arithmetic operation to apply",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    context=True,
                    strict_validation=True,
                    default="add",
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert GoodDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"].default == "add"

    def test_required_when_does_not_rescue_strict_bad_default(self) -> None:
        """``required_when`` must not exempt a strict default outside the accepted set.

        ``required_when`` is a conditional requirement: when its predicate is False
        and the key is omitted, the bad default still applies silently, reopening the
        exact bug the invariant closes. So a strict, non-None default outside the
        accepted set must reject at construction even with ``required_when`` set.
        """
        with pytest.raises(ValueError) as exc_info:

            class RequiredWhenBadDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The arithmetic operation to apply",
                        allowed_values={"add": "Addition", "sub": "Subtraction"},
                        context=True,
                        strict_validation=True,
                        default="mul",
                        required_when=_never_required,
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "default" in message
        assert "mul" in message

    def test_accepts_none_default_under_strict_validation(self) -> None:
        """A ``None`` default is the unset/optional sentinel and is always legal.

        Even when the key is strictly validated against an enumerated accepted set
        that does not include ``None`` (mirroring real plugins such as
        ``SklearnPipelineFeatureGroup.pipeline_name``), defining the FeatureGroup
        subclass must not raise.
        """

        class NoneDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "The pipeline step to run",
                    allowed_values={
                        "feature_engineering": "Feature engineering step",
                        "scaling": "Scaling step",
                    },
                    context=True,
                    strict_validation=True,
                    default=None,
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert NoneDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"].default is None

    def test_no_op_for_non_strict_spec(self) -> None:
        """When strict_validation is False, the listed values are illustrative; any default is fine."""

        class NonStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "The arithmetic operation to apply",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    context=True,
                    strict_validation=False,
                    default="mul",
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert NonStrictFeatureGroup.PROPERTY_MAPPING["operation_type"].default == "mul"

    def test_unhashable_default_reports_clear_error(self) -> None:
        """An unhashable strict default surfaces as ``ValueError``, not a bare ``TypeError``.

        Membership in a Mapping value space tests KEYS, and an unhashable default
        (e.g. a list) can never be a key: ``["mul"] in {...}`` would raise
        ``TypeError``. The invariant must translate that into the regular
        out-of-value-space ``ValueError`` rather than leaking the ``TypeError``.
        """
        with pytest.raises(ValueError) as exc_info:

            class UnhashableDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The arithmetic operation to apply",
                        allowed_values={"add": "Addition", "sub": "Subtraction"},
                        context=True,
                        strict_validation=True,
                        default=["mul"],
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "default" in message
        assert "mul" in message


class TestStrictElementValidatorDefaultInvariant:
    """Strict validation driven by an element_validator callable."""

    def test_rejects_default_failing_element_validator(self) -> None:
        """A strict key whose default fails its element_validator must reject at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class BadFnDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The operation to apply",
                        context=True,
                        strict_validation=True,
                        element_validator=lambda v: v in {"x", "y"},
                        default="z",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "default" in message
        assert "z" in message

    def test_accepts_default_passing_element_validator(self) -> None:
        """A strict key whose default passes its element_validator defines without error."""

        class GoodFnDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "The operation to apply",
                    context=True,
                    strict_validation=True,
                    element_validator=lambda v: v in {"x", "y"},
                    default="x",
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert GoodFnDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"].default == "x"


class TestElementValidatorRaisedVersusRejected:
    """The error must distinguish a default REJECTED by a working validator from a validator
    that itself RAISED when called: only the first is a verdict on the default."""

    def test_buggy_element_validator_reported_as_raised_not_rejected(self) -> None:
        """A element_validator that errors when called is reported as having raised.

        ``lambda: True`` takes no arguments, so calling it with the default raises
        ``TypeError``. That is a bug in the validator, not a verdict on the default,
        so construction must surface a ``ValueError`` saying the element_validator
        raised (not that the default was "rejected by" it), chaining the original
        ``TypeError`` as the cause.
        """
        with pytest.raises(ValueError) as exc_info:

            class BuggyFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": _spec(
                        "The operation to apply",
                        context=True,
                        strict_validation=True,
                        element_validator=lambda: True,
                        default="x",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "raised" in message
        assert "rejected by" not in message
        assert isinstance(exc_info.value.__cause__, TypeError), "expected the original TypeError chained as __cause__"

    def test_element_validator_raising_attribute_error_reports_raised(self) -> None:
        """A element_validator raising ``AttributeError`` surfaces as the curated ``ValueError``.

        ``lambda v: v.startswith("a")`` crashes with ``AttributeError`` for the
        non-string default ``5``. The raw ``AttributeError`` must never escape class
        definition; construction must say the element_validator raised and chain the
        original error as the cause.
        """
        with pytest.raises(ValueError) as exc_info:

            class AttributeErrorFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The operation to apply",
                        context=True,
                        strict_validation=True,
                        element_validator=lambda v: v.startswith("a"),
                        default=5,
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "raised" in message
        assert "rejected by" not in message
        assert isinstance(exc_info.value.__cause__, AttributeError), (
            "expected the original AttributeError chained as __cause__"
        )

    def test_element_validator_raising_value_error_reports_raised(self) -> None:
        """A element_validator raising ``ValueError`` internally is reported as having raised.

        ``lambda v: int(v) > 0`` crashes with ``ValueError`` for the default
        ``"abc"``: the function never returned a verdict, it errored. Construction
        must not misreport this as the default being "rejected by" the validator.
        """
        with pytest.raises(ValueError) as exc_info:

            class ValueErrorFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The operation to apply",
                        context=True,
                        strict_validation=True,
                        element_validator=lambda v: int(v) > 0,
                        default="abc",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "raised" in message
        assert "rejected by" not in message
        assert isinstance(exc_info.value.__cause__, ValueError), "expected the original ValueError chained as __cause__"

    def test_genuine_rejection_still_labeled_as_rejection(self) -> None:
        """A working element_validator that returns False is still reported as a rejection.

        Distinguishing a buggy validator (it raised) must not erase the rejection
        wording for the genuine case: a callable that runs and returns False has
        rejected the default, and the message must say so.
        """
        with pytest.raises(ValueError) as exc_info:

            class RejectingFnDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "The operation to apply",
                        context=True,
                        strict_validation=True,
                        element_validator=lambda v: False,
                        default="x",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "rejected by the element_validator" in message
        assert "raised" not in message


class TestNoDefaultDeclared:
    """When no default is declared, the invariant performs no check."""

    def test_strict_key_without_default_defines_without_error(self) -> None:
        """A strict, narrowed key with no default declared (required by omission) is accepted."""

        class NoDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "The arithmetic operation to apply",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    context=True,
                    strict_validation=True,
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert NoDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"].default is NO_DEFAULT
