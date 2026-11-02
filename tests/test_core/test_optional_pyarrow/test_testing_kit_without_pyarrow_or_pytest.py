"""Verify mloda._testing imports and runs with pyarrow and pytest both blocked."""

import subprocess  # nosec
import sys

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import blocker_preamble

_BODY = """
import importlib
import pkgutil

import mloda._testing

found = set()
for info in pkgutil.iter_modules(mloda._testing.__path__):
    importlib.import_module(f"mloda._testing.{info.name}")
    found.add(info.name)
assert {"suite", "typed_data", "value_fit", "declarations", "result"} <= found, found

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.suite import check_switched_off
from mloda._testing.typed_data import TypedTestData
from mloda._testing.value_fit import check_values_fit

data = TypedTestData({"a": (DataType.INT64, [1, 2]), "b": (DataType.STRING, ["x", "y"])})
print(data.to_python_dict())

fit = check_values_fit({"a": [1, "x"]}, {"a": DataType.INT64})
assert fit.passed is False
print(fit.failures)


class Plain:
    pass


switched = check_switched_off(Plain)
assert switched.passed is True
print(switched)
assert "pyarrow" not in sys.modules
assert "pytest" not in sys.modules
print("OK")
"""


@pytest.mark.timeout(30)
def test_testing_kit_imports_without_pyarrow_or_pytest() -> None:
    script = blocker_preamble("pyarrow") + blocker_preamble("pytest") + _BODY
    result = subprocess.run(  # nosec
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        "Subprocess failed: mloda._testing needs pyarrow or pytest at import time.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.rstrip().endswith("OK")
