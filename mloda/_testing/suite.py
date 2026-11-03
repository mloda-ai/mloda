"""Subclass-form conformance suite whose hook names are checked at class creation."""

from __future__ import annotations

import difflib
import inspect
import sys
from typing import Any, ClassVar

from mloda._testing.result import CheckResult
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user.python_dict import PythonDictFramework

if sys.version_info >= (3, 14):
    import annotationlib


class HookError(Exception):
    """A suite names something that looks like a misspelled hook."""


class ConformanceSuite:
    """Base for suites; hooks are declared on suite bases (`suite_base=True`)."""

    plugin: ClassVar[type[Any]]
    compute_framework: ClassVar[type[ComputeFramework]] = PythonDictFramework
    _suite_base: ClassVar[bool] = True

    def __init_subclass__(cls, suite_base: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._suite_base = suite_base
        if suite_base:
            return
        result = check_hooks(cls)
        if not result.passed:
            raise HookError(f"{cls.__name__}: " + "; ".join(result.failures))


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
