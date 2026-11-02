"""ConformanceSuite declarations are validated at class creation."""

from __future__ import annotations

from typing import ClassVar

import pytest

from mloda._testing.result import CheckResult
from mloda._testing.suite import ConformanceSuite, DeclarationError, assert_passed, check_declarations
from mloda.user.python_dict import PythonDictFramework


class _Plugin:
    pass


class _Base(ConformanceSuite):
    declaration_keys: ClassVar[frozenset[str]] = frozenset({"match_names"})


def test_misspelled_key_raises_at_class_creation_naming_closest_key() -> None:
    with pytest.raises(DeclarationError) as excinfo:

        class _Bad(_Base):
            plugin = _Plugin
            declarations = {"match_name": ["x"]}

    message = str(excinfo.value)
    assert "match_name" in message
    assert "match_names" in message
    assert "_Bad" in message
    assert "_testing" not in message


def test_correct_subclass_is_fine() -> None:
    class _Good(_Base):
        plugin = _Plugin
        declarations = {"match_names": ["x"]}

    result = check_declarations(_Good)

    assert isinstance(result, CheckResult)
    assert result.failures == ()


def test_check_declarations_reports_failures_without_init_subclass() -> None:
    class _Good(_Base):
        plugin = _Plugin
        declarations = {"match_names": ["x"]}

    bad: dict[str, list[str]] = {"match_name": ["x"]}
    _Good.declarations = bad

    result = check_declarations(_Good)

    assert result.check == "declarations"
    assert result.failures == ("unknown declaration key 'match_name', did you mean 'match_names'?",)


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
