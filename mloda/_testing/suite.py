"""Subclass-form conformance suite whose hook names are checked at class creation."""

from __future__ import annotations

import difflib
import inspect
import sys
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar

from mloda._testing.declarations import Cause
from mloda._testing.result import CheckResult
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user.python_dict import PythonDictFramework

if sys.version_info >= (3, 14):
    import annotationlib


class HookError(Exception):
    """A suite names something that looks like a misspelled hook."""


class SwitchedOffError(Exception):
    """A host switches off a contract test without declaring a cause."""


class ConformanceSuite:
    """Base for suites (`suite_base=True`); a host declares a cause in `switched_off` to redefine, blank or skip a test.

    A stricter override calling super() is declared OTHER; a module skipif for a missing dependency needs
    ENVIRONMENT for every test. Hosts must be named Test*. Not caught: pytest.skip in a fixture or test body,
    importorskip, item.add_marker in a collection hook, mutating the shared base function.
    """

    plugin: ClassVar[type[Any]]
    compute_framework: ClassVar[type[ComputeFramework]] = PythonDictFramework
    switched_off: ClassVar[Mapping[str, Cause]] = MappingProxyType({})
    _suite_base: ClassVar[bool] = True

    def __init_subclass__(cls, suite_base: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._suite_base = suite_base
        if suite_base:
            if _has_skip_mark(cls.__dict__.get("pytestmark")):
                raise SwitchedOffError(f"{cls.__name__}: a suite base must not carry skip, skipif or xfail marks")
            return
        result = check_hooks(cls)
        if not result.passed:
            raise HookError(f"{cls.__name__}: " + "; ".join(result.failures))
        result = check_switched_off(cls)
        if not result.passed:
            raise SwitchedOffError(f"{cls.__name__}: " + "; ".join(result.failures))


_SKIP_MARKS = frozenset({"skip", "skipif", "xfail"})


def _has_skip_mark(marks: Any) -> bool:
    if marks is None:
        return False
    items = marks if isinstance(marks, (list, tuple)) else [marks]
    return any(getattr(m, "name", None) in _SKIP_MARKS for m in items)


def has_skip_mark_names(names: Iterable[str]) -> bool:
    """True when any mark name is skip, skipif or xfail."""
    return any(name in _SKIP_MARKS for name in names)


def host_has_skip_mark(suite_cls: type) -> bool:
    """True when the host or a non-suite-base class in its MRO carries a skip mark in its own body."""
    return any(
        not base.__dict__.get("_suite_base", False) and _has_skip_mark(base.__dict__.get("pytestmark"))
        for base in suite_cls.__mro__
    )


def _contract_tests(suite_cls: type) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    for base in suite_cls.__mro__:
        if base.__dict__.get("_suite_base", False):
            for name, value in base.__dict__.items():
                if name.startswith("test_"):
                    tests.setdefault(name, value)
    return tests


def undeclared_contract_tests(suite_cls: type[ConformanceSuite], marked: bool) -> list[str]:
    """Contract tests lacking a declared cause; all of them when `marked`, else only the overridden ones."""
    tests = _contract_tests(suite_cls)
    declared = suite_cls.switched_off
    return sorted(
        name
        for name, reference in tests.items()
        if name not in declared and (marked or inspect.getattr_static(suite_cls, name, None) is not reference)
    )


def check_switched_off(suite_cls: type[ConformanceSuite]) -> CheckResult:
    """Fail unknown keys, non-Cause values and undeclared switched-off contract tests."""
    tests = _contract_tests(suite_cls)
    failures: list[str] = []
    for name, cause in suite_cls.switched_off.items():
        if name not in tests:
            close = difflib.get_close_matches(name, sorted(tests), n=1, cutoff=0.6)
            hint = f"; did you mean '{close[0]}'?" if close else ""
            failures.append(f"switched_off key '{name}' is not a contract test{hint}")
        elif not isinstance(cause, Cause):
            failures.append(f"switched_off['{name}'] must be a Cause, got {cause!r}")
    undeclared = undeclared_contract_tests(suite_cls, host_has_skip_mark(suite_cls))
    if undeclared:
        allowed = ", ".join(c.name for c in Cause)
        failures.append(f"switched off without a declared cause: {', '.join(undeclared)} (causes: {allowed})")
    return CheckResult(check="switched_off", subject=suite_cls.__name__, failures=tuple(failures))


def _is_public(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("test_")


def _annotation_names(cls: type) -> list[str]:
    if sys.version_info >= (3, 14):
        return list(annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF))
    return list(inspect.get_annotations(cls))


def check_hooks(suite_cls: type[ConformanceSuite]) -> CheckResult:
    """Fail each public name on the suite that nearly matches a hook but is not one."""
    hooks: set[str] = set()
    for base in suite_cls.__mro__:
        if base.__dict__.get("_suite_base", False):
            hooks.update(name for name in (*base.__dict__, *_annotation_names(base)) if _is_public(name))
    checked = {name for name in (*suite_cls.__dict__, *_annotation_names(suite_cls)) if _is_public(name)}
    failures: list[str] = []
    for name in sorted(checked - hooks):
        close = difflib.get_close_matches(name, sorted(hooks), n=1, cutoff=0.8)
        if close:
            failures.append(f"'{name}' is not a hook; did you mean '{close[0]}'? Rename it if it is not meant as one.")
    return CheckResult(check="hooks", subject=suite_cls.__name__, failures=tuple(failures))


def assert_passed(result: CheckResult) -> None:
    """Fail the running pytest test with the result's failures."""
    import pytest

    if not result.passed:
        pytest.fail(f"{result.check} failed for {result.subject}:\n" + "\n".join(result.failures), pytrace=False)
