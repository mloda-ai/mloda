"""Core imports pyarrow lazily (issue #737).

pyarrow is an optional backend. The three core modules that use it must not import it at module level,
not even when it is installed: a module-level ``try: import pyarrow`` costs every core-only user the
pyarrow import as soon as the library happens to be present in the environment. Each test runs a fresh
interpreter, so the assertions are about a real cold import, not this session's already-warm sys.modules.
"""

import subprocess  # nosec B404
import sys

import pytest

_DATA_TYPES_MODULE = "mloda.core.abstract_plugins.components.data_types"

# The three core modules that import pyarrow at module level today.
_CORE_MODULES: list[str] = [
    _DATA_TYPES_MODULE,
    "mloda.core.abstract_plugins.compute_framework",
    "mloda.core.runtime.flight.flight_server",
]

_IMPORT_SCRIPT = """
import sys

import {module}

leaked = sorted(name for name in sys.modules if name == "pyarrow" or name.startswith("pyarrow."))
assert not leaked, "import {module} pulled pyarrow into sys.modules: " + repr(leaked)

print("ok")
"""

_INFER_TYPE_SCRIPT = """
import sys

from {module} import DataType

assert "pyarrow" not in sys.modules, "importing DataType pulled pyarrow into sys.modules"

inferred = DataType.infer_type_from_py_type(5)
assert inferred is DataType.INT32, "sanity: infer_type_from_py_type(5) must be INT32, got " + repr(inferred)

leaked = sorted(name for name in sys.modules if name == "pyarrow" or name.startswith("pyarrow."))
assert not leaked, "DataType.infer_type_from_py_type(5) pulled pyarrow into sys.modules: " + repr(leaked)

print("ok")
"""


def _run_isolation_script(script: str) -> subprocess.CompletedProcess[str]:
    # Safe: fixed argv (sys.executable + a script built from module-level constants), no shell, no user input.
    return subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestCoreDoesNotImportPyarrowEagerly:
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("module", _CORE_MODULES)
    def test_core_module_import_does_not_load_pyarrow(self, module: str) -> None:
        result = _run_isolation_script(_IMPORT_SCRIPT.format(module=module))

        assert result.returncode == 0, (
            f"import {module} must not pull pyarrow into sys.modules in a fresh interpreter. pyarrow is an "
            f"optional backend: the module-level import has to move behind the call sites that actually need "
            f"an Arrow type.\nstderr:\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"

    @pytest.mark.timeout(30)
    def test_infer_type_from_py_type_does_not_load_pyarrow(self) -> None:
        """Hot path: inferring the type of a plain Python value must not pay for a pyarrow import either."""
        result = _run_isolation_script(_INFER_TYPE_SCRIPT.format(module=_DATA_TYPES_MODULE))

        assert result.returncode == 0, (
            f"DataType.infer_type_from_py_type(5) must not pull pyarrow into sys.modules: it runs per value on "
            f"plain Python types, so the pyarrow branches must only be reached for values that are not a bool, "
            f"int, float, str, bytes, datetime, date or Decimal.\nstderr:\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"
