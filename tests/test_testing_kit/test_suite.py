"""Tests for ConformanceSuite, switch-off declarations and assert_passed, using a toy suite."""

import ast
import importlib
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda._testing.declarations import Cause, DeclarationError, SwitchOff, unknown_key_failures
from mloda._testing.result import CheckResult
from mloda._testing.suite import ConformanceSuite, assert_passed, check_switched_off
from mloda.user.python_dict import PythonDictFramework


def check_has_name(plugin: type[Any]) -> CheckResult:
    failures = () if hasattr(plugin, "name") else ("plugin has no name",)
    return CheckResult("has_name", plugin.__qualname__, failures)


class _GoodPlugin:
    name = "good"


class _BadPlugin:
    pass


class ToySuite(ConformanceSuite, suite=True):
    plugin: ClassVar[type[Any]]

    def test_has_name(self) -> None:
        assert_passed(check_has_name(self.plugin))

    def test_other(self) -> None:
        assert_passed(CheckResult("other", self.plugin.__qualname__))


class TestToyProvider(ToySuite):
    """Provider subclass: collected and green."""

    plugin = _GoodPlugin


def test_provider_failure_surfaces_through_assert_passed() -> None:
    class BadProvider(ToySuite):
        plugin = _BadPlugin

    with pytest.raises(pytest.fail.Exception, match="plugin has no name"):
        BadProvider().test_has_name()


def test_undeclared_override_raises() -> None:
    with pytest.raises(DeclarationError, match="test_other") as info:

        class Provider(ToySuite):
            plugin = _GoodPlugin

            def test_other(self) -> None:
                return None

    assert "_testing" not in str(info.value)


def test_none_override_raises() -> None:
    with pytest.raises(DeclarationError, match="test_other"):

        class Provider(ToySuite):
            plugin = _GoodPlugin
            test_other = None  # type: ignore[assignment]


def test_override_through_non_suite_intermediate_raises() -> None:
    class Mid(ToySuite):
        plugin = _GoodPlugin

    with pytest.raises(DeclarationError, match="test_other"):

        class Leaf(Mid):
            def test_other(self) -> None:
                return None


def test_new_test_and_sub_suite_override_are_allowed() -> None:
    class Extra(ToySuite):
        plugin = _GoodPlugin

        def test_extra(self) -> None:
            return None

    class SubSuite(ToySuite, suite=True):
        def test_other(self) -> None:
            return None

    assert Extra.test_extra is not None
    assert SubSuite.test_other is not ToySuite.test_other


def test_check_switched_off_reports_unknown_key() -> None:
    class Late(ToySuite):
        plugin = _GoodPlugin

    Late.switched_off = {"test_othr": SwitchOff(Cause.NOT_IMPLEMENTED)}
    result = check_switched_off(Late)
    assert result.passed is False
    assert any("test_othr" in f and "test_other" in f for f in result.failures)


def test_check_switched_off_passes_on_valid_provider() -> None:
    assert check_switched_off(TestToyProvider).passed is True


def test_class_body_pytestmark_raises() -> None:
    with pytest.raises(DeclarationError, match="pytestmark"):

        class Provider(ToySuite):
            plugin = _GoodPlugin
            pytestmark = pytest.mark.skip(reason="hides tests")


def test_not_implemented_declaration_replaces_test_with_skip() -> None:
    class Provider(ToySuite):
        plugin = _BadPlugin
        switched_off = {"test_has_name": SwitchOff(Cause.NOT_IMPLEMENTED)}

    with pytest.raises(pytest.skip.Exception, match="NOT_IMPLEMENTED"):
        Provider().test_has_name()
    Provider().test_other()


def test_skip_message_carries_cause_and_reason() -> None:
    class Provider(ToySuite):
        plugin = _BadPlugin
        switched_off = {"test_has_name": SwitchOff(Cause.BACKEND, reason="engine lacks it")}

    with pytest.raises(pytest.skip.Exception, match="BACKEND: engine lacks it"):
        Provider().test_has_name()


def test_provider_setup_method_does_not_undo_stub() -> None:
    class Provider(ToySuite):
        plugin = _BadPlugin
        switched_off = {"test_has_name": SwitchOff(Cause.NOT_IMPLEMENTED)}

        def setup_method(self) -> None:
            return None

    instance = Provider()
    instance.setup_method()
    with pytest.raises(pytest.skip.Exception):
        instance.test_has_name()


@pytest.mark.parametrize("cause", [Cause.BACKEND, Cause.ENVIRONMENT])
def test_backend_and_environment_without_reason_raise(cause: Cause) -> None:
    with pytest.raises(DeclarationError):
        SwitchOff(cause)


def test_other_requires_docs() -> None:
    with pytest.raises(DeclarationError):
        SwitchOff(Cause.OTHER, reason="why")
    assert SwitchOff(Cause.OTHER, reason="why", docs="see docs").docs == "see docs"
    assert SwitchOff(Cause.NOT_IMPLEMENTED).reason == ""
    assert SwitchOff(Cause.BACKEND, reason="r").cause is Cause.BACKEND


def test_unknown_key_failures() -> None:
    failures = unknown_key_failures(["test_othr", "zzz_unrelated_key"], ["test_other", "test_has_name"], "ToySuite")
    assert len(failures) == 2
    assert "test_other" in failures[0]
    assert "test_othr" in failures[0]
    assert "ToySuite" in failures[0]
    assert unknown_key_failures(["test_other"], ["test_other"], "ToySuite") == ()


def test_assert_passed_fails_with_check_subject_and_failures() -> None:
    with pytest.raises(pytest.fail.Exception) as info:
        assert_passed(CheckResult("my_check", "SubjectCls", ("first problem", "second problem")))
    message = str(info.value)
    for part in ("my_check", "SubjectCls", "first problem", "second problem"):
        assert part in message
    assert "_testing" not in message


def test_assert_passed_accepts_passing_result() -> None:
    assert_passed(CheckResult("c", "S"))


def test_no_string_literal_names_private_module() -> None:
    package = importlib.import_module("mloda._testing")
    files = [p for root in package.__path__ for p in Path(root).glob("*.py")]
    assert len(files) >= 6
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in files
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "mloda._testing" in node.value
    ]
    assert offenders == []


def test_default_compute_framework_is_python_dict() -> None:
    assert ConformanceSuite.compute_framework is PythonDictFramework
