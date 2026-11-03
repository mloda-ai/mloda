"""ConformanceSuite hooks are validated at class creation: near-miss names on a test class are errors."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda._testing.result import CheckResult
from mloda._testing.suite import ConformanceSuite, HookError, assert_passed, check_hooks
from mloda.user.python_dict import PythonDictFramework


class _Plugin:
    pass


class _Base(ConformanceSuite, suite_base=True):
    match_names: ClassVar[tuple[str, ...]] = ()

    def input_data(self) -> list[str]:
        return []


def _hint(name: str, hook: str) -> str:
    return f"'{name}' is not a hook; did you mean '{hook}'? Rename it if it is not meant as one."


def test_misspelled_attribute_hook_raises_at_class_creation() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugin = _Plugin
            match_name = ("x",)

    message = str(excinfo.value)
    assert "_Bad" in message
    assert _hint("match_name", "match_names") in message
    assert "_testing" not in message


def test_misspelled_inherited_plugin_hook_raises() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugn = _Plugin

    assert _hint("plugn", "plugin") in str(excinfo.value)


def test_misspelled_method_hook_raises() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            def input_dat(self) -> list[str]:
                return []

    assert _hint("input_dat", "input_data") in str(excinfo.value)


def test_annotation_only_typo_raises() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugn: ClassVar[type[Any]]

    assert _hint("plugn", "plugin") in str(excinfo.value)


def test_multiple_failures_are_joined() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugn = _Plugin
            match_name = ("x",)

    assert str(excinfo.value) == "_Bad: " + _hint("match_name", "match_names") + "; " + _hint("plugn", "plugin")


def test_correct_subclass_with_unrelated_public_names_passes() -> None:
    class _Good(_Base):
        pytestmark = [pytest.mark.filterwarnings("ignore")]
        plugin = _Plugin
        match_names = ("x",)
        DATA = {"a": [1]}

        def setup_method(self) -> None:
            pass

        def input_data(self) -> list[str]:
            return ["x"]

        def build_rows(self) -> list[int]:
            return [1]

        match_values = ("x",)

        def _input_dat(self) -> None:
            pass

        def test_input_data(self) -> None:
            pass

    assert check_hooks(_Good) == CheckResult(check="hooks", subject="_Good", failures=())


def test_check_hooks_on_class_mutated_after_creation_returns_exact_failure() -> None:
    class _Good(_Base):
        plugin = _Plugin

    setattr(_Good, "match_name", ("x",))

    result = check_hooks(_Good)

    assert result == CheckResult(
        check="hooks",
        subject="_Good",
        failures=(_hint("match_name", "match_names"),),
    )


def test_check_hooks_on_root_suite_passes() -> None:
    assert check_hooks(ConformanceSuite) == CheckResult(check="hooks", subject="ConformanceSuite", failures=())


def test_second_level_suite_base_hook_becomes_known() -> None:
    class _Second(_Base, suite_base=True):
        extra_hook: ClassVar[int] = 0

    class _Good(_Second):
        extra_hook = 1

    assert check_hooks(_Good).failures == ()

    with pytest.raises(HookError) as excinfo:

        class _Bad(_Second):
            extra_hok = 1

    assert _hint("extra_hok", "extra_hook") in str(excinfo.value)


def test_suite_base_is_not_inherited() -> None:
    class _Plain(_Base):
        extra_setting = 1

    class _Child(_Plain):
        extra_settin = 2

    assert check_hooks(_Child).failures == ()


def test_plain_mixin_overriding_a_hook_is_allowed() -> None:
    class _Mixin:
        match_names: ClassVar[tuple[str, ...]] = ("x",)
        match_name: ClassVar[tuple[str, ...]] = ("y",)

        def input_data(self) -> list[str]:
            return ["x"]

    class _Good(_Mixin, _Base):
        plugin = _Plugin

    assert check_hooks(_Good).failures == ()


def _raises_name_error(parent: type) -> bool:
    # dont_inherit keeps this file's future import out of the probe source.
    code = compile("class Probe(Parent):\n    peer: Undefined\n", "<suite>", "exec", dont_inherit=True)
    try:
        exec(code, {"Parent": parent})  # nosec B102
    except NameError:
        return True
    return False


def test_unresolvable_annotation_behaves_like_a_plain_class() -> None:
    assert _raises_name_error(_Base) == _raises_name_error(object)


def test_default_compute_framework_is_python_dict() -> None:
    assert ConformanceSuite.compute_framework is PythonDictFramework


def test_assert_passed_accepts_passing_result() -> None:
    assert_passed(CheckResult(check="c", subject="s"))


def test_assert_passed_fails_with_the_failures() -> None:
    result = CheckResult(check="c", subject="s", failures=("first problem", "second problem"))

    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_passed(result)

    message = str(excinfo.value)
    assert "first problem" in message and "second problem" in message
