"""Collection fails for suite hosts that skip contract tests without declaring a cause."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

HEADER = """
import pytest
from mloda._testing.declarations import Cause
from mloda._testing.suite import ConformanceSuite


class Base(ConformanceSuite, suite_base=True):
    def test_one(self):
        pass

    def test_two(self):
        pass

"""

MARKED = "skip, skipif or xfail marks require a declared cause (switched_off) for: "
BOTH = MARKED + "test_one, test_two"

SKIPPED_BASE = (
    "@pytest.mark.skip(reason='x')\n"
    "class SkippedBase(ConformanceSuite, suite_base=True):\n"
    "    def test_one(self):\n        pass\n\n"
    "    def test_two(self):\n        pass\n\n\n"
    "class TestHostSkippedBase(SkippedBase):\n    plugin = object\n"
)
MIXIN = "@pytest.mark.skip(reason='x')\nclass SkipMixin:\n    pass\n\n\nclass TestHostMixin(SkipMixin, Base):\n    plugin = object\n"
MARKED_FUNCTION = "class TestHostFunction(Base):\n    plugin = object\n    test_one = pytest.mark.skip(Base.test_one)\n"

# name -> (host, source, host failure line after "<host>: ")
FAILING = {
    "class_skip": (
        "TestHostSkip",
        "@pytest.mark.skip(reason='x')\nclass TestHostSkip(Base):\n    plugin = object\n",
        BOTH,
    ),
    "class_skipif_false": (
        "TestHostSkipif",
        "@pytest.mark.skipif(False, reason='x')\nclass TestHostSkipif(Base):\n    plugin = object\n",
        BOTH,
    ),
    "module_marks": (
        "TestHostModule",
        "pytestmark = pytest.mark.skipif(True, reason='x')\n\n\nclass TestHostModule(Base):\n    plugin = object\n",
        BOTH,
    ),
    "class_xfail": (
        "TestHostXfail",
        "@pytest.mark.xfail(reason='x')\nclass TestHostXfail(Base):\n    plugin = object\n",
        BOTH,
    ),
    "suite_base_marked_after_definition": ("TestHostSkippedBase", SKIPPED_BASE, BOTH),
    "non_suite_mixin_mark": ("TestHostMixin", MIXIN, "switched off without a declared cause: test_one, test_two"),
    "mark_applied_to_shared_function": ("TestHostFunction", MARKED_FUNCTION, MARKED + "test_one"),
}


def test_undeclared_switched_off_tests_fail_collection(pytester: pytest.Pytester) -> None:
    for name, (_, body, _) in FAILING.items():
        pytester.makepyfile(**{f"test_{name}": HEADER + body})

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    assert result.ret == 2
    output = result.stdout.str()
    missing = [name for name, (host, _, line) in FAILING.items() if f"{host}: {line}" not in output]
    assert missing == []


def test_declared_and_plain_classes_are_untouched(pytester: pytest.Pytester) -> None:
    declared = (
        "@pytest.mark.skip(reason='x')\n"
        "class TestDeclared(Base):\n"
        "    plugin = object\n"
        "    switched_off = {'test_one': Cause.BACKEND, 'test_two': Cause.ENVIRONMENT}\n"
    )
    plain = "@pytest.mark.skip(reason='x')\nclass TestPlain:\n    def test_x(self):\n        pass\n"
    unmarked = "class TestUnmarked(Base):\n    plugin = object\n"
    pytester.makepyfile(test_ok=HEADER + declared + "\n\n" + plain + "\n\n" + unmarked)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    assert result.ret == 0
    result.assert_outcomes(passed=2, skipped=3)
