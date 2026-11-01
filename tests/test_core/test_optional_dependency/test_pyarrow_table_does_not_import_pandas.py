"""The pyarrow backend does not depend on pandas, so importing its table module must not import pandas.

Planning a PyArrow-only request must not import any heavy backend either, yet the transformer registry
stays complete for the backends a request does use. A standby worker for such a request imports none either.
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


_PYARROW_MODULE = "mloda_plugins.compute_framework.base_implementations.pyarrow.table"
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


_STANDBY_IMPORTS = (
    "import queue\n"
    "from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import "
    "ComputeFrameworkTransformer\n"
    "from mloda.core.runtime.worker.multiprocessing_worker import standby_worker\n"
    "q = queue.Queue()\n"
    "q.put('STOP')\n"
)
_HEAVY_LOADED = f"print([m for m in {_HEAVY!r} if m in sys.modules])\n"


@pytest.mark.timeout(60)
def test_planning_and_standby_for_a_pyarrow_request_import_no_heavy_backend() -> None:
    body = (
        _PLAN_FG
        + _PLAN
        + _HEAVY_LOADED
        + _STANDBY_IMPORTS
        + f"modules = ({_PYARROW_MODULE!r},)\n"
        + "standby_worker(q, q, modules)\n"
        + "print(modules)\n"
        + _HEAVY_LOADED
    )
    result = _run_probe(body)

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    lines = result.stdout.strip().splitlines()
    assert lines[-3] == "[]", f"backends imported while planning:\n{result.stdout}"
    assert lines[-2:] == [str((_PYARROW_MODULE,)), "[]"], f"standby imported backends:\n{result.stdout}"


@pytest.mark.timeout(60)
def test_transformer_registry_is_complete_once_every_installed_backend_is_imported() -> None:
    body = (
        _PLAN_FG
        + _PLAN
        + "import importlib, importlib.util\n"
        + "from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import "
        + "ComputeFrameworkTransformer\n"
        + "early = ComputeFrameworkTransformer()\n"
        + f"for gate in {_GATES!r}:\n"
        + "    if importlib.util.find_spec(gate.split('.')[0]) is not None:\n"
        + "        importlib.import_module(gate)\n"
        + "if importlib.util.find_spec('pandas') is not None:\n"
        + "    import pandas, pyarrow\n"
        + "    print('CHAIN' if early.get_transformation_chain(pandas.DataFrame, pyarrow.Table) is not None else 'NOCHAIN')\n"
        + "def keys():\n"
        + "    return sorted(str(k) for k in ComputeFrameworkTransformer().transformer_map)\n"
        + "before = keys()\n"
        + "ComputeFrameworkTransformer.load_all_transformers()\n"
        + "print('SAME' if before == keys() else 'DIFFERENT')\n"
    )
    result = _run_probe(body)

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "NOCHAIN" not in result.stdout, f"early transformer found no pandas to pyarrow chain:\n{result.stdout}"
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


@pytest.mark.timeout(60)
def test_standby_worker_warms_exactly_the_given_gate() -> None:
    body = (
        "import sys\n"
        + _STANDBY_IMPORTS
        + "standby_worker(q, q, ('pandas',))\n"
        + "print('pandas' in ComputeFrameworkTransformer._lazy_dirs_loaded)\n"
        + _HEAVY_LOADED
    )
    result = _run_probe(body)

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().splitlines()[-2:] == ["True", "['pandas']"], f"unexpected warm:\n{result.stdout}"


@pytest.mark.timeout(60)
def test_worker_unpickling_pyarrow_payload_from_all_backends_parent_imports_no_heavy_backend() -> None:
    """Variant: the FG class stays out of the payload (it lives in A's __main__); steps ship cfw class and features."""
    producer = (
        _PLAN_FG
        + "import importlib, importlib.util\n"
        + f"for gate in {_GATES!r}:\n"
        + "    if importlib.util.find_spec(gate.split('.')[0]) is not None:\n"
        + "        importlib.import_module(gate)\n"
        + "from multiprocessing.reduction import ForkingPickler\n"
        + "from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode\n"
        + "from mloda.core.core.step.feature_group_step import FeatureGroupStep\n"
        + "session = mloda.prepare(['LazyFG'], compute_frameworks=[PyArrowTable], plugin_collector=collector, "
        + "parallelization_modes={ParallelizationMode.MULTIPROCESSING})\n"
        + "plan = session.engine.execution_planner\n"
        + "steps = [s for s in plan if isinstance(s, FeatureGroupStep)]\n"
        + "assert steps\n"
        + "payload = [PyArrowTable()] + [(s.compute_framework, s.features) for s in steps]\n"
        + "print(bytes(ForkingPickler.dumps(payload)).hex())\n"
    )
    produced = _run_probe(producer)
    assert produced.returncode == 0, f"producer failed:\n{produced.stdout}\n{produced.stderr}"
    hex_payload = produced.stdout.strip().splitlines()[-1]

    consumer = (
        "import sys, pickle\n"
        "import mloda.core.runtime.worker.multiprocessing_worker\n"
        f"pickle.loads(bytes.fromhex({hex_payload!r}))\n" + _HEAVY_LOADED
    )
    result = _run_probe(consumer)

    assert result.returncode == 0, f"consumer failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().splitlines()[-1] == "[]", f"worker imported backends:\n{result.stdout}"
