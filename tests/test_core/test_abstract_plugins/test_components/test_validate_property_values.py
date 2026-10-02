"""validate_property_values checks a plain dict against a PropertySpec mapping without echoing values."""

from __future__ import annotations

import logging
import types
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import contained_raise_log_level
from mloda.provider import NO_DEFAULT, PropertySpec, PropertyValidationError, validate_property_values
from tests.test_core.test_abstract_plugins.test_components.feature_chainer.test_property_mapping_sequence_unpacking import (
    CONTAINERS,
    EMPTY_CONTAINERS,
    INT_CONTAINERS,
)

SECRET = "s3cr3t-token"  # nosec B105
VALIDATOR_LOGGER = "mloda.core.abstract_plugins.components.property_values_validation"


class _RaisingEq:
    """An element whose equality check raises a non-TypeError carrying the secret."""

    def __eq__(self, other: object) -> bool:
        raise ValueError(f"leak {SECRET}")

    def __hash__(self) -> int:
        return 1


def _strict(**kwargs: Any) -> dict[str, PropertySpec]:
    return {"k": PropertySpec("k", strict_validation=True, default=None, **kwargs)}


def _reject_all(_: Any) -> bool:
    return False


def _raise_with_value(value: Any) -> bool:
    raise ValueError(f"bad {value}")


class TestWorld:
    def test_closed_world_rejects_undeclared_key(self) -> None:
        with pytest.raises(PropertyValidationError) as exc:
            validate_property_values({"a": 1, "zzz": 2}, {"a": PropertySpec("a")}, closed_world=True)
        assert exc.value.key == "zzz"

    def test_open_world_ignores_undeclared_key(self) -> None:
        validate_property_values({"a": 1, "zzz": 2}, {"a": PropertySpec("a")}, closed_world=False)


class TestAllowedValues:
    @pytest.mark.parametrize("allowed", [{"a": "A", "b": "B"}, ("a", "b"), ["a", "b"], {"a", "b"}])
    def test_member_accepted_and_non_member_rejected(self, allowed: Any) -> None:
        mapping = _strict(allowed_values=allowed)
        validate_property_values({"k": "a"}, mapping, closed_world=True)
        with pytest.raises(PropertyValidationError) as exc:
            validate_property_values({"k": "c"}, mapping, closed_world=True)
        assert exc.value.key == "k"

    @pytest.mark.parametrize("allowed", [("a",), {"a"}], ids=["tuple", "set"])
    @pytest.mark.parametrize("value", [[["a"]], {"a": 1}], ids=["nested_list", "dict_scalar"])
    def test_unhashable_element_rejected_not_type_error(self, allowed: Any, value: Any) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({"k": value}, _strict(allowed_values=allowed), closed_world=True)


class TestElementValidator:
    def test_accepts(self) -> None:
        validate_property_values({"k": 1}, _strict(element_validator=lambda v: True), closed_world=True)

    def test_rejects(self) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({"k": 1}, _strict(element_validator=_reject_all), closed_world=True)

    def test_raising_validator_is_rejected_not_escaped(self) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({"k": 1}, _strict(element_validator=_raise_with_value), closed_world=True)


class TestValidatorLogLevel:
    def test_value_error_validator_logs_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger=VALIDATOR_LOGGER):
            with pytest.raises(PropertyValidationError):
                validate_property_values({"k": 1}, _strict(element_validator=_raise_with_value), closed_world=True)
        records = [r for r in caplog.records if r.name == VALIDATOR_LOGGER]
        assert not [r for r in records if r.levelno >= logging.WARNING]

    def test_runtime_error_validator_logs_at_contained_raise_level(self, caplog: pytest.LogCaptureFixture) -> None:
        def broken(_: Any) -> bool:
            raise RuntimeError("boom")

        with caplog.at_level(logging.DEBUG, logger=VALIDATOR_LOGGER):
            with pytest.raises(PropertyValidationError):
                validate_property_values({"k": 1}, _strict(element_validator=broken), closed_world=True)
        expected = contained_raise_log_level(RuntimeError("boom"))
        records = [r for r in caplog.records if r.name == VALIDATOR_LOGGER]
        assert records
        assert max(r.levelno for r in records) == expected


class TestSequenceUnpacking:
    @pytest.mark.parametrize(("label", "value"), CONTAINERS, ids=[label for label, _ in CONTAINERS])
    def test_members_accepted(self, label: str, value: Any) -> None:
        validate_property_values({"k": value}, _strict(allowed_values=("a", "b")), closed_world=True)

    @pytest.mark.parametrize(("label", "value"), CONTAINERS, ids=[label for label, _ in CONTAINERS])
    def test_non_member_element_rejected(self, label: str, value: Any) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({"k": value}, _strict(allowed_values=("a",)), closed_world=True)

    @pytest.mark.parametrize(("label", "value"), INT_CONTAINERS, ids=[label for label, _ in INT_CONTAINERS])
    def test_elements_keep_their_real_type(self, label: str, value: Any) -> None:
        validate_property_values({"k": value}, _strict(element_validator=lambda v: type(v) is int), closed_world=True)

    @pytest.mark.parametrize(("label", "value"), EMPTY_CONTAINERS, ids=[label for label, _ in EMPTY_CONTAINERS])
    def test_empty_container_is_present_and_valid(self, label: str, value: Any) -> None:
        mapping = {"k": PropertySpec("k", strict_validation=True, allowed_values=("a",))}
        validate_property_values({"k": value}, mapping, closed_world=True)

    def test_str_is_one_scalar(self) -> None:
        validate_property_values({"k": "abc"}, _strict(allowed_values=("abc",)), closed_world=True)


class TestScalarOnly:
    def test_rejects_collection(self) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({"k": ["a"]}, _strict(scalar_only=True, allowed_values=("a",)), closed_world=True)

    def test_accepts_scalar(self) -> None:
        validate_property_values({"k": "a"}, _strict(scalar_only=True, allowed_values=("a",)), closed_world=True)


class TestNonStrict:
    def test_never_validates_value(self) -> None:
        mapping = {"k": PropertySpec("k", allowed_values=("a",), default=None)}
        validate_property_values({"k": "zzz"}, mapping, closed_world=True)


class TestPresence:
    def test_no_default_absent_is_error(self) -> None:
        with pytest.raises(PropertyValidationError) as exc:
            validate_property_values({}, {"k": PropertySpec("k", default=NO_DEFAULT)}, closed_world=True)
        assert exc.value.key == "k"

    @pytest.mark.parametrize("default", [1, None])
    def test_declared_default_absent_is_fine(self, default: Any) -> None:
        validate_property_values({}, {"k": PropertySpec("k", default=default)}, closed_world=True)

    def test_explicit_none_present_when_allowed(self) -> None:
        mapping = {"k": PropertySpec("k", allow_explicit_none=True)}
        validate_property_values({"k": None}, mapping, closed_world=True)

    def test_explicit_none_absent_when_not_allowed(self) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({"k": None}, {"k": PropertySpec("k")}, closed_world=True)


class TestRequiredWhen:
    def test_fires_and_absent_is_error(self) -> None:
        mapping = {"k": PropertySpec("k", default=None, required_when=lambda o: True)}
        with pytest.raises(PropertyValidationError) as exc:
            validate_property_values({}, mapping, closed_world=True)
        assert exc.value.key == "k"

    def test_not_firing_is_fine(self) -> None:
        mapping = {"k": PropertySpec("k", default=None, required_when=lambda o: False)}
        validate_property_values({}, mapping, closed_world=True)

    def test_predicate_receives_options_of_raw_values(self) -> None:
        seen: list[Any] = []

        def predicate(options: Any) -> bool:
            seen.append((isinstance(options, Options), options.get("mode")))
            return False

        mapping = {"k": PropertySpec("k", default=None, required_when=predicate), "mode": PropertySpec("m")}
        validate_property_values({"mode": "x"}, mapping, closed_world=True)
        assert seen == [(True, "x")]

    def test_raising_predicate_is_rejected(self) -> None:
        def predicate(_: Any) -> bool:
            raise RuntimeError("boom")

        mapping = {"k": PropertySpec("k", default=None, required_when=predicate)}
        with pytest.raises(PropertyValidationError):
            validate_property_values({}, mapping, closed_world=True)


class TestFrameworkSetKeys:
    def test_absent_framework_set_key_is_skipped(self) -> None:
        mapping = {"k": PropertySpec("k", framework_set=True, default=NO_DEFAULT)}
        validate_property_values({}, mapping, closed_world=True)


class TestRequiredWhenPresentKey:
    def test_predicate_not_called_when_key_present(self) -> None:
        calls: list[Any] = []

        def predicate(options: Any) -> bool:
            calls.append(options)
            return True

        mapping = {"k": PropertySpec("k", default=None, required_when=predicate)}
        validate_property_values({"k": 1}, mapping, closed_world=True)
        assert calls == []


class TestKeyTypes:
    def test_non_str_undeclared_key_rejected_with_str_key(self) -> None:
        with pytest.raises(PropertyValidationError) as exc:
            validate_property_values({1: "x"}, {}, closed_world=True)  # type: ignore[dict-item]
        assert exc.value.key == "1"

    def test_mixed_type_undeclared_keys_do_not_raise_type_error(self) -> None:
        with pytest.raises(PropertyValidationError):
            validate_property_values({1: "x", "a": "y"}, {}, closed_world=True)  # type: ignore[dict-item]


class TestMappingProxyValues:
    def test_valid_and_invalid(self) -> None:
        mapping = _strict(allowed_values=("a",))
        validate_property_values(types.MappingProxyType({"k": "a"}), mapping, closed_world=True)
        with pytest.raises(PropertyValidationError):
            validate_property_values(types.MappingProxyType({"k": "c"}), mapping, closed_world=True)


class TestMappingShape:
    def test_non_property_spec_entry_is_type_error(self) -> None:
        with pytest.raises(TypeError):
            validate_property_values({"k": 1}, {"k": {"explanation": "raw"}}, closed_world=True)  # type: ignore[dict-item]


class TestSecretHygiene:
    @pytest.mark.parametrize(
        "spec_kwargs",
        [
            {"allowed_values": ("public",)},
            {"element_validator": _reject_all},
            {"element_validator": _raise_with_value},
            {"scalar_only": True, "allowed_values": (SECRET,)},
        ],
        ids=["non_member", "validator_reject", "validator_raises", "scalar_only"],
    )
    def test_value_never_leaks(self, spec_kwargs: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        value: Any = [SECRET] if "scalar_only" in spec_kwargs else SECRET
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(PropertyValidationError) as exc:
                validate_property_values({"k": value}, _strict(**spec_kwargs), closed_world=True)
        assert SECRET not in str(exc.value)
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None
        assert SECRET not in caplog.text

    @pytest.mark.parametrize("allowed", [("a",), {"a"}], ids=["tuple", "set"])
    def test_raising_eq_membership_is_rejected_and_does_not_leak(
        self, allowed: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(PropertyValidationError) as exc:
                validate_property_values({"k": _RaisingEq()}, _strict(allowed_values=allowed), closed_world=True)
        assert SECRET not in str(exc.value)
        assert exc.value.__context__ is None
        assert SECRET not in caplog.text

    def test_raising_required_when_predicate_does_not_leak(self, caplog: pytest.LogCaptureFixture) -> None:
        def predicate(_: Any) -> bool:
            raise RuntimeError(f"leak {SECRET}")

        mapping = {"k": PropertySpec("k", default=None, required_when=predicate)}
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(PropertyValidationError) as exc:
                validate_property_values({"other": SECRET}, mapping, closed_world=False)
        assert SECRET not in caplog.text
        assert SECRET not in str(exc.value)


class TestErrorShape:
    def test_is_value_error(self) -> None:
        assert issubclass(PropertyValidationError, ValueError)

    @pytest.mark.parametrize(
        ("values", "mapping"),
        [
            ({"x": 1}, {}),
            ({}, {"k": PropertySpec("k")}),
            ({"k": "c"}, _strict(allowed_values=("a",))),
        ],
        ids=["undeclared", "absent_required", "non_member"],
    )
    def test_key_set_on_every_error(self, values: dict[str, Any], mapping: dict[str, PropertySpec]) -> None:
        with pytest.raises(PropertyValidationError) as exc:
            validate_property_values(values, mapping, closed_world=True)
        assert isinstance(exc.value.key, str) and exc.value.key
