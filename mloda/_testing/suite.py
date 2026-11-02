"""Subclass-form conformance suite with class-creation-time declaration checks."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from typing import Any, ClassVar

from mloda._testing.result import CheckResult
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user.python_dict import PythonDictFramework


class DeclarationError(Exception):
    """A suite declares a key its base does not know."""


class ConformanceSuite:
    """Base for suites; each suite base sets its own declaration_keys."""

    plugin: ClassVar[type[Any]]
    compute_framework: ClassVar[type[ComputeFramework]] = PythonDictFramework
    declaration_keys: ClassVar[frozenset[str]] = frozenset()
    declarations: ClassVar[Mapping[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        result = check_declarations(cls)
        if not result.passed:
            raise DeclarationError(f"{cls.__name__}: " + "; ".join(result.failures))


def check_declarations(suite_cls: type[ConformanceSuite]) -> CheckResult:
    """Fail each declared key outside declaration_keys, naming the closest known key."""
    failures: list[str] = []
    for key in suite_cls.declarations:
        if key in suite_cls.declaration_keys:
            continue
        message = f"unknown declaration key '{key}'"
        close = difflib.get_close_matches(key, sorted(suite_cls.declaration_keys), n=1)
        if close:
            message += f", did you mean '{close[0]}'?"
        failures.append(message)
    return CheckResult(check="declarations", subject=suite_cls.__name__, failures=tuple(failures))


def assert_passed(result: CheckResult) -> None:
    """Fail the running pytest test with the result's failures."""
    import pytest

    if not result.passed:
        pytest.fail(f"{result.check} failed for {result.subject}:\n" + "\n".join(result.failures), pytrace=False)
