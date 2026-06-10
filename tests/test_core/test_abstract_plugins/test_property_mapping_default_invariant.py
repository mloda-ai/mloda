"""Tests for the class-definition-time PROPERTY_MAPPING default invariant (issue #520).

At class definition time, for every PROPERTY_MAPPING key whose spec is a dict and
declares a non-``None`` ``DefaultOptionKeys.default``, the declared default must be
honored by the key's own strict-validation rules (be in the accepted set or pass the
``validation_function``). Otherwise defining the FeatureGroup subclass must raise
``ValueError`` immediately.

``DefaultOptionKeys.required_when`` is NOT an escape hatch: it expresses a conditional
requirement, so when its predicate is False (or it is non-callable) and the key is
omitted, the bad default would still apply silently. The only sound way to mark a key
required is to declare NO default; mloda treats a key with no default and no
``required_when`` as unconditionally required.
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
from mloda.provider import DefaultOptionKeys


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    The tests below define FeatureGroup subclasses to exercise
    ``FeatureGroup.__init_subclass__``. Those class objects sit in reference
    cycles, so they linger in ``FeatureGroup.__subclasses__()`` until a GC cycle
    runs. While they linger, other tests that enumerate feature groups via
    ``get_all_subclasses(FeatureGroup)`` trip over them. After each test we force
    a collection to reclaim the now-unreferenced classes and assert that none of
    this module's classes remain registered, pinning the no-pollution contract.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _never_required(options: Any) -> bool:
    """Conditional-requirement predicate that is never satisfied.

    With such a predicate the key is not required, so a bad default would still
    apply silently when the key is omitted: ``required_when`` must therefore not
    exempt a strict, non-None default from the invariant.
    """
    return False


class TestStrictEnumeratedDefaultInvariant:
    """Strict validation with an enumerated accepted set."""

    def test_rejects_strict_default_outside_accepted_set(self) -> None:
        """A strict key whose default is not in the accepted set must reject at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class BadDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: "mul",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "BadDefaultFeatureGroup" in message
        assert "operation_type" in message
        assert "mul" in message
        assert "add" in message
        assert "sub" in message

    def test_accepts_strict_default_in_accepted_set(self) -> None:
        """A strict key whose default is in the accepted set defines without error."""

        class GoodDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": {
                    "add": "Addition",
                    "sub": "Subtraction",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.default: "add",
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert GoodDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"][DefaultOptionKeys.default] == "add"

    def test_required_when_does_not_rescue_strict_bad_default(self) -> None:
        """``required_when`` must not exempt a strict default outside the accepted set.

        ``required_when`` is a conditional requirement: when its predicate is False
        and the key is omitted, the bad default still applies silently, reopening the
        exact bug the invariant closes. So a strict, non-None default outside the
        accepted set must reject at class definition even with ``required_when`` set.
        """
        with pytest.raises(ValueError) as exc_info:

            class RequiredWhenBadDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: "mul",
                        DefaultOptionKeys.required_when: _never_required,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "RequiredWhenBadDefaultFeatureGroup" in message
        assert "operation_type" in message
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
                "operation_type": {
                    "feature_engineering": "Feature engineering step",
                    "scaling": "Scaling step",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.default: None,
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert NoneDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"][DefaultOptionKeys.default] is None

    def test_no_op_for_non_strict_spec(self) -> None:
        """When strict_validation is False, the listed values are illustrative; any default is fine."""

        class NonStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": {
                    "add": "Addition",
                    "sub": "Subtraction",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.default: "mul",
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert NonStrictFeatureGroup.PROPERTY_MAPPING["operation_type"][DefaultOptionKeys.default] == "mul"

    def test_rejects_empty_accepted_set_under_strict_validation(self) -> None:
        """A strict key with no enumerated values and no validation_function rejects a default.

        Without any accepted values and without a ``validation_function``, the
        declared default can never be honored, so defining the FeatureGroup subclass
        must raise ``ValueError``.
        """
        with pytest.raises(ValueError) as exc_info:

            class EmptyAcceptedSetFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: "x",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "EmptyAcceptedSetFeatureGroup" in message
        assert "operation_type" in message

    def test_unhashable_default_reports_clear_error(self) -> None:
        """An unhashable strict default surfaces as ``ValueError``, not a bare ``TypeError``.

        The membership check ``default not in accepted`` raises ``TypeError`` for an
        unhashable default (e.g. a list). The invariant must translate that into a
        clear ``ValueError`` naming the class and key rather than leaking the
        ``TypeError``.
        """
        with pytest.raises(ValueError) as exc_info:

            class UnhashableDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: ["mul"],
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "UnhashableDefaultFeatureGroup" in message
        assert "operation_type" in message


class TestStrictValidationFunctionDefaultInvariant:
    """Strict validation driven by a validation_function callable."""

    def test_rejects_default_failing_validation_function(self) -> None:
        """A strict key whose default fails its validation_function must reject at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class BadFnDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.validation_function: lambda v: v in {"x", "y"},
                        DefaultOptionKeys.default: "z",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        assert "BadFnDefaultFeatureGroup" in message
        assert "operation_type" in message
        assert "z" in message

    def test_accepts_default_passing_validation_function(self) -> None:
        """A strict key whose default passes its validation_function defines without error."""

        class GoodFnDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": {
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.validation_function: lambda v: v in {"x", "y"},
                    DefaultOptionKeys.default: "x",
                }
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert GoodFnDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"][DefaultOptionKeys.default] == "x"


class TestNoDefaultDeclared:
    """When no default is declared, the invariant performs no check."""

    def test_strict_key_without_default_defines_without_error(self) -> None:
        """A strict, narrowed key with no default declared (required by omission) is accepted."""

        class NoDefaultFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
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

        assert DefaultOptionKeys.default not in NoDefaultFeatureGroup.PROPERTY_MAPPING["operation_type"]


class TestDefaultInvariantErrorMessaging:
    """The ValueError raised by the invariant must give accurate, actionable advice.

    The two legal remedies for a strict default outside the accepted set are: add the
    default to the accepted values, or remove the default. ``required_when`` is NOT a
    remedy (its predicate being False still lets the bad default apply silently), so
    the message must never advise it. The message must also distinguish a default that
    was genuinely rejected by a working validation_function from a validation_function
    that itself errored when called.
    """

    def test_message_advises_only_the_two_legal_remedies(self) -> None:
        """The message for a strict out-of-set default advises exactly the legal remedies.

        It must advise adding the default to the accepted values and removing the
        default, and must NOT advise ``required_when``: a False predicate plus an
        omitted key still applies the bad default silently, so suggesting it would
        steer users back into the very bug the invariant closes.
        """
        with pytest.raises(ValueError) as exc_info:

            class RemedyAdviceFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: "mul",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        # Drop the ExceptionInfo before asserting: when an assert fails, pytest's
        # report retains this frame, and via the captured traceback the throwaway
        # class would stay alive and trip the no-pollution fixture.
        del exc_info
        assert "accepted values" in message
        assert "remove the default" in message
        assert "required_when" not in message

    def test_message_renders_accepted_values_without_nested_quoting(self) -> None:
        """Accepted values render as plain reprs, not repr-of-repr.

        The accepted set for the key below must appear as ``['add', 'sub']`` in the
        message, not as ``["'add'", "'sub'"]`` (stringifying each value before
        repr-ing the list double-quotes every entry and obscures the actual values).
        """
        with pytest.raises(ValueError) as exc_info:

            class ValueRenderingFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        "add": "Addition",
                        "sub": "Subtraction",
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.default: "mul",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        # Drop the ExceptionInfo before asserting (see remedy-advice test above).
        del exc_info
        assert "['add', 'sub']" in message

    def test_buggy_validation_function_reported_as_raised_not_rejected(self) -> None:
        """A validation_function that errors when called is reported as having raised.

        ``lambda: True`` takes no arguments, so calling it with the default raises
        ``TypeError``. That is a bug in the plugin's validation_function, not a
        verdict on the default, so the invariant must still surface a ``ValueError``
        at class definition, must say the validation_function raised (not that the
        default was "rejected by" it), and must chain the original ``TypeError`` as
        the cause.
        """
        with pytest.raises(ValueError) as exc_info:

            class BuggyFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.validation_function: lambda: True,
                        DefaultOptionKeys.default: "x",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        cause_is_type_error = isinstance(exc_info.value.__cause__, TypeError)
        # Drop the ExceptionInfo before asserting (see remedy-advice test above).
        del exc_info
        assert "BuggyFnFeatureGroup" in message
        assert "operation_type" in message
        assert "rejected by" not in message
        assert "raised" in message
        assert cause_is_type_error, "expected the original TypeError chained as __cause__"

    def test_genuine_rejection_still_labeled_as_rejection(self) -> None:
        """A working validation_function that returns False is still reported as a rejection.

        Distinguishing a buggy validation_function (it raised) must not erase the
        rejection wording for the genuine case: a callable that runs and returns
        False has rejected the default, and the message must say so.
        """
        with pytest.raises(ValueError) as exc_info:

            class RejectingFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.validation_function: lambda v: False,
                        DefaultOptionKeys.default: "x",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        # Drop the ExceptionInfo before asserting (see remedy-advice test above).
        del exc_info
        assert "RejectingFnFeatureGroup" in message
        assert "operation_type" in message
        assert "rejected by the key's validation_function" in message

    def test_validation_function_raising_attribute_error_reports_raised(self) -> None:
        """A validation_function raising ``AttributeError`` surfaces as the curated ``ValueError``.

        ``lambda v: v.startswith("a")`` crashes with ``AttributeError`` for the
        non-string default ``5``. That is a bug in the plugin's validation_function,
        not a verdict on the default, so the invariant must still surface a
        ``ValueError`` at class definition naming the class and key, must say the
        validation_function raised (not that the default was "rejected by" it), and
        must chain the original ``AttributeError`` as the cause. The raw
        ``AttributeError`` must never escape class definition.
        """
        with pytest.raises(ValueError) as exc_info:

            class AttributeErrorFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.validation_function: lambda v: v.startswith("a"),
                        DefaultOptionKeys.default: 5,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        cause_is_attribute_error = isinstance(exc_info.value.__cause__, AttributeError)
        # Drop the ExceptionInfo before asserting (see remedy-advice test above).
        del exc_info
        assert "AttributeErrorFnFeatureGroup" in message
        assert "operation_type" in message
        assert "raised" in message
        assert "rejected by" not in message
        assert cause_is_attribute_error, "expected the original AttributeError chained as __cause__"

    def test_validation_function_raising_value_error_reports_raised(self) -> None:
        """A validation_function raising ``ValueError`` internally is reported as having raised.

        ``lambda v: int(v) > 0`` crashes with ``ValueError`` for the default
        ``"abc"``: the function never returned a verdict, it errored. The invariant
        must not misreport this as the default being "rejected by" the
        validation_function; it must say the validation_function raised and chain
        the original ``ValueError`` as the cause.
        """
        with pytest.raises(ValueError) as exc_info:

            class ValueErrorFnFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": {
                        DefaultOptionKeys.context: True,
                        DefaultOptionKeys.strict_validation: True,
                        DefaultOptionKeys.validation_function: lambda v: int(v) > 0,
                        DefaultOptionKeys.default: "abc",
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        cause_is_value_error = isinstance(exc_info.value.__cause__, ValueError)
        # Drop the ExceptionInfo before asserting (see remedy-advice test above).
        del exc_info
        assert "ValueErrorFnFeatureGroup" in message
        assert "operation_type" in message
        assert "raised" in message
        assert "rejected by" not in message
        assert cause_is_value_error, "expected the original ValueError chained as __cause__"
