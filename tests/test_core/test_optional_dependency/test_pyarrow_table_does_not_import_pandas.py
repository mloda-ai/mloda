"""The pyarrow backend does not depend on pandas, so importing its table module must not import pandas.

Planning a PyArrow-only request must not import any heavy backend either, yet the transformer registry
stays complete for the backends a request does use.
"""

from __future__ import annotations

import subprocess  # nosec
import sys

import pytest

_PROBE = (
    "import sys\n"
    "import mloda_plugins.compute_framework.base_implementations.pyarrow.table\n"
    "print('pandas' in sys.modules)\n"
)


def test_importing_pyarrow_table_does_not_import_pandas() -> None:
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0, f"probe failed to import the module:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip() == "False", (
        f"pandas was imported as a side effect of importing the pyarrow table module:\n{result.stdout}\n{result.stderr}"
    )


_HEAVY = ("pandas", "polars", "pyiceberg", "pyspark", "duckdb")
_GATES = ("duckdb", "pyiceberg.table", "pandas", "polars", "pyspark.sql")

_FG_SOURCE = (
    "import sys\n"
    "from typing import Any\n"
    "from mloda.provider import DataCreatorFeatureGroup\n"
    "from mloda.user import mloda, PluginCollector\n"
    "from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable\n"
    "from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import "
    "PythonDictFramework\n"
    "class LazyFG(DataCreatorFeatureGroup):\n"
    "    @classmethod\n"
    "    def calculate_feature(cls, data: Any, features: Any) -> Any:\n"
    "        {body}\n"
    "    @classmethod\n"
    "    def compute_framework_rule(cls) -> Any:\n"
    "        return {{PyArrowTable}}\n"
    "collector = PluginCollector.enabled_feature_groups({{LazyFG}})\n"
)
_PLAN_FG = _FG_SOURCE.format(body="return {'LazyFG': [1, 2, 3]}")
_PLAN = (
    "mloda.explain(['LazyFG'], compute_frameworks=[PyArrowTable, PythonDictFramework], plugin_collector=collector)\n"
)


def _run_probe(body: str) -> subprocess.CompletedProcess[str]:
    # A fresh interpreter importing mloda and planning costs a few seconds.
    return subprocess.run(  # nosec B603
        [sys.executable, "-c", body], capture_output=True, text=True, timeout=55
    )


@pytest.mark.timeout(60)
def test_planning_a_pyarrow_request_imports_no_heavy_backend() -> None:
    body = _PLAN_FG + _PLAN + f"print([m for m in {_HEAVY!r} if m in sys.modules])\n"
    result = _run_probe(body)

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().splitlines()[-1] == "[]", f"backends imported while planning:\n{result.stdout}"


@pytest.mark.timeout(60)
def test_transformer_registry_is_complete_once_every_installed_backend_is_imported() -> None:
    body = (
        _PLAN_FG
        + _PLAN
        + "import importlib, importlib.util\n"
        + f"for gate in {_GATES!r}:\n"
        + "    if importlib.util.find_spec(gate.split('.')[0]) is not None:\n"
        + "        importlib.import_module(gate)\n"
        + "from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import "
        + "ComputeFrameworkTransformer\n"
        + "def keys():\n"
        + "    return sorted(str(k) for k in ComputeFrameworkTransformer().transformer_map)\n"
        + "before = keys()\n"
        + "ComputeFrameworkTransformer.load_all_transformers()\n"
        + "print('SAME' if before == keys() else 'DIFFERENT')\n"
    )
    result = _run_probe(body)

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().splitlines()[-1] == "SAME", f"registry was partial:\n{result.stdout}"


@pytest.mark.timeout(60)
def test_a_lazy_pandas_import_in_calculate_feature_still_converts_to_pyarrow() -> None:
    fg = _FG_SOURCE.format(
        body="import pandas as pd; return pd.DataFrame({'LazyFG': [1, 2, 3]})",
    )
    body = (
        fg
        + "import pyarrow as pa\n"
        + "result = mloda.run_all(['LazyFG'], compute_frameworks=[PyArrowTable], plugin_collector=collector)\n"
        + "print('OK' if isinstance(result[0], pa.Table) and result[0].num_rows == 3 else 'BAD')\n"
    )
    result = _run_probe(body)

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().splitlines()[-1] == "OK"
