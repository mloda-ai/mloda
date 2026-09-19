"""The pyarrow backend does not depend on pandas, so importing its table module must not import pandas."""

from __future__ import annotations

import subprocess  # nosec
import sys

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
