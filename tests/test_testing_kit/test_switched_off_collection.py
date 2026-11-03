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

FAILING = {
    "class_skip": "@pytest.mark.skip(reason='x')\nclass TestHostSkip(Base):\n    plugin = object\n",
    "class_skipif_false": "@pytest.mark.skipif(False, reason='x')\nclass TestHostSkipif(Base):\n    plugin = object\n",
    "module_marks": "pytestmark = pytest.mark.skipif(True, reason='x')\n\n\nclass TestHostModule(Base):\n    plugin = object\n",
    "class_xfail": "@pytest.mark.xfail(reason='x')\nclass TestHostXfail(Base):\n    plugin = object\n",
}

HOST_NAMES = {
    "class_skip": "TestHostSkip",
    "class_skipif_false": "TestHostSkipif",
    "module_marks": "TestHostModule",
    "class_xfail": "TestHostXfail",
}


def test_undeclared_switched_off_tests_fail_collection(pytester: pytest.Pytester) -> None:
    for name, body in FAILING.items():
        pytester.makepyfile(**{f"test_{name}": HEADER + body})

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    assert result.ret == 2
    output = result.stdout.str()
    for name, host in HOST_NAMES.items():
        assert host in output, name
    assert "test_one" in output and "test_two" in output


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
