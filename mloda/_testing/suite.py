"""Base class for conformance suites and the pytest-facing helpers.

pytest is imported lazily so the module loads without it.
"""

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from mloda._testing.declarations import DeclarationError, SwitchOff, unknown_key_failures
from mloda._testing.result import CheckResult
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user.python_dict import PythonDictFramework


def _is_suite(klass: type) -> bool:
    return bool(klass.__dict__.get("_conformance_suite", False))


def _contract_tests(cls: type) -> set[str]:
    return {name for klass in cls.__mro__ if _is_suite(klass) for name in klass.__dict__ if name.startswith("test_")}


def check_switched_off(cls: type) -> CheckResult:
    """Check a provider class declares its switched-off tests consistently."""
    contract = _contract_tests(cls)
    switched_off: Mapping[str, SwitchOff] = getattr(cls, "switched_off", {})
    failures: list[str] = []
    for klass in cls.__mro__:
        if klass is object or _is_suite(klass):
            continue
        failures.extend(
            f"{klass.__qualname__} redefines contract test '{name}' without declaring it in switched_off"
            for name, attr in klass.__dict__.items()
            if name in contract and name not in switched_off and not getattr(attr, "_is_skip_stub", False)
        )
        if "pytestmark" in klass.__dict__:
            marks = klass.__dict__["pytestmark"]
            for mark in marks if isinstance(marks, (list, tuple)) else [marks]:
                if getattr(getattr(mark, "mark", mark), "name", None) in ("skip", "skipif"):
                    failures.append(
                        f"{klass.__qualname__} sets a skip pytestmark in its class body, which hides inherited tests"
                    )
    failures.extend(
        f"{cls.__qualname__}: switched_off['{key}'] must be a SwitchOff, got {type(value).__name__}"
        for key, value in switched_off.items()
        if not isinstance(value, SwitchOff)
    )
    failures.extend(unknown_key_failures(switched_off, contract, cls.__qualname__))
    return CheckResult("switched_off", cls.__qualname__, tuple(failures))


def _skip_stub(name: str, declaration: SwitchOff) -> Callable[[Any], None]:
    message = f"{declaration.cause.name}: {declaration.reason}" if declaration.reason else declaration.cause.name

    def stub(self: Any) -> None:
        import pytest

        pytest.skip(message)

    stub.__name__ = name
    stub._is_skip_stub = True  # type: ignore[attr-defined]
    return stub


class ConformanceSuite:
    """Base for suites; declare with `suite=True`, subclass with a Test* name."""

    _conformance_suite: ClassVar[bool] = True
    plugin: ClassVar[type[Any]]
    compute_framework: ClassVar[type[ComputeFramework]] = PythonDictFramework
    switched_off: ClassVar[Mapping[str, SwitchOff]] = {}

    def __init_subclass__(cls, suite: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._conformance_suite = suite
        if suite:
            return
        result = check_switched_off(cls)
        if not result.passed:
            raise DeclarationError(f"{cls.__qualname__}: " + "; ".join(result.failures))
        for name in _contract_tests(cls) - set(cls.switched_off):
            if getattr(getattr(cls, name, None), "_is_skip_stub", False):
                original = next(k.__dict__[name] for k in cls.__mro__ if _is_suite(k) and name in k.__dict__)
                setattr(cls, name, original)
        for name, declaration in cls.switched_off.items():
            setattr(cls, name, _skip_stub(name, declaration))


def assert_passed(result: CheckResult) -> None:
    """Fail the running test when the check did not pass."""
    if result.passed:
        return
    import pytest

    pytest.fail(f"{result.check} failed for {result.subject}: " + "; ".join(result.failures), pytrace=False)
