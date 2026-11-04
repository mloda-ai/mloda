"""The non-public testing kit imports and runs with neither pyarrow nor pytest importable."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import blocker_preamble, run_blocked

_BODY: str = """
import importlib
import pkgutil
import sys

import mloda._testing

# mloda._testing.source holds the pytest mixins and imports pytest and pyarrow on purpose, so it is skipped.
# mloda._testing.compute_framework holds pytest suites, so pytest is let through for it, but pyarrow stays blocked.
_SKIPPED_SUBTREE = "mloda._testing.source"
_PYTEST_SUBTREE = "mloda._testing.compute_framework"


def _in_subtree(name, subtree):
    return name == subtree or name.startswith(subtree + ".")


for info in pkgutil.walk_packages(mloda._testing.__path__, "mloda._testing."):
    if _in_subtree(info.name, _SKIPPED_SUBTREE) or _in_subtree(info.name, _PYTEST_SUBTREE):
        continue
    importlib.import_module(info.name)


_pytest_blocker = sys.meta_path[0]
sys.meta_path.remove(_pytest_blocker)
for info in pkgutil.walk_packages(mloda._testing.__path__, "mloda._testing."):
    if _in_subtree(info.name, _PYTEST_SUBTREE):
        importlib.import_module(info.name)
assert "pyarrow" not in sys.modules, "mloda._testing.compute_framework must not import pyarrow"
for name in [m for m in sys.modules if m == "pytest" or m.startswith(("pytest.", "_pytest"))]:
    del sys.modules[name]
sys.meta_path.insert(0, _pytest_blocker)

from mloda._testing.source_applicability import NO_POINTER, source_applicability
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

from mloda._testing.framework_adapter import PythonDictAdapter

adapter = PythonDictAdapter()
frame = adapter.from_typed(data)
assert adapter.column_names(frame) == ["a"], adapter.column_names(frame)
assert adapter.row_count(frame) == 2
assert adapter.column_values(frame, "a") == [1, None]
assert "pyarrow" not in sys.modules, "PythonDictAdapter must not import pyarrow"

from mloda._testing.feature_group_shape import check_feature_group_shape
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class RunKit(FeatureGroup):
    @classmethod
    def feature_names_supported(cls):
        return {"rk_out"}

    def input_features(self, options, feature_name):
        return {Feature("rk_src")}

    @classmethod
    def return_data_type_rule(cls, feature):
        return DataType.INT64

    @classmethod
    def calculate_feature(cls, data, features):
        data["rk_out"] = [v * 2 for v in data["rk_src"]]
        return data


run = check_feature_group_shape(
    RunKit, input_data=TypedTestData({"rk_src": (DataType.INT64, [1, 2])}), adapter=PythonDictAdapter()
)
assert run.failures == (), run.failures
assert "pyarrow" not in sys.modules, "shape run checks must not import pyarrow"
assert "pytest" not in sys.modules, "shape run checks must not import pytest"

print("OK")
"""


@pytest.mark.timeout(30)
def test_testing_kit_works_without_pyarrow_or_pytest() -> None:
    # run_blocked already blocks pyarrow; the extra preamble blocks pytest.
    result = run_blocked(blocker_preamble("pytest") + _BODY)

    assert result.returncode == 0, f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    assert "OK" in result.stdout
