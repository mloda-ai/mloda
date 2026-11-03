"""The non-public testing kit imports and runs with neither pyarrow nor pytest importable."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import blocker_preamble, run_blocked

_BODY: str = """
import importlib
import pkgutil
import sys

import mloda._testing

for info in pkgutil.iter_modules(mloda._testing.__path__, "mloda._testing."):
    importlib.import_module(info.name)

from mloda._testing.result import CheckResult
from mloda._testing.suite import ConformanceSuite, check_hooks
from mloda._testing.typed_data import TypedTestData
from mloda._testing.value_fit import check_values_fit
from mloda.core.abstract_plugins.components.data_types import DataType

data = TypedTestData({"a": (DataType.INT8, [1, None])})
fit = check_values_fit(data.to_python_dict(), data.types)
hooks = check_hooks(ConformanceSuite)

assert isinstance(fit, CheckResult), fit
assert isinstance(hooks, CheckResult), hooks
assert "pyarrow" not in sys.modules, "testing kit must not import pyarrow"
assert "pytest" not in sys.modules, "testing kit must not import pytest"

print("OK")
"""


@pytest.mark.timeout(30)
def test_testing_kit_works_without_pyarrow_or_pytest() -> None:
    # run_blocked already blocks pyarrow; the extra preamble blocks pytest.
    result = run_blocked(blocker_preamble("pytest") + _BODY)

    assert result.returncode == 0, f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    assert "OK" in result.stdout
