"""RED tests: reading a CSV must NOT require pyarrow once the FileSource seam lands.

Target design (these FAIL today):

  * ``CsvReader`` resolves a CSV into a lightweight ``FileSource`` descriptor, and a
    per-compute-framework transformer materializes it into that framework's native
    type. ``PythonDictFramework`` therefore reads a CSV using only the stdlib, so
    ``mloda.run_all([...], compute_frameworks={PythonDictFramework}, ...)`` succeeds
    even when pyarrow is unavailable.
  * ``CsvReader.get_column_names`` discovers the header with the stdlib ``csv``
    module, so it works with pyarrow absent.

Both bodies run in a subprocess with pyarrow blocked via ``sys.meta_path`` (see
``_pyarrow_blocker.run_blocked``). They fail today because CSV reading currently goes
through ``pyarrow.csv`` (``load_data`` and ``get_column_names`` both import it), which
raises under the block, so no ``OK`` sentinel is printed.
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

# ---------------------------------------------------------------------------
# End-to-end: run_all on a CSV with the dict-backed PythonDict framework, no pyarrow.
# ---------------------------------------------------------------------------
_BODY_RUN_ALL: str = """
import csv
import os
import sys
import tempfile

fd, path = tempfile.mkstemp(suffix=".csv")
os.close(fd)
# A: integer column, B: non-numeric (stays str), C: float column.
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["A", "B", "C"])
    writer.writerow(["1", "x", "1.5"])
    writer.writerow(["2", "y", "2.5"])

try:
    from mloda.user import DataAccessCollection
    from mloda.user import PluginLoader
    from mloda.user import mloda
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
        PythonDictFramework,
    )

    PluginLoader.all()

    result = mloda.run_all(
        ["A", "B", "C"],
        compute_frameworks={PythonDictFramework},
        data_access_collection=DataAccessCollection(files={path}),
    )

    payload = result[0]
    if not isinstance(payload, dict):
        print("WRONG_TYPE:" + type(payload).__name__)
        sys.exit(1)

    a = payload["A"]
    b = payload["B"]
    c = payload["C"]
    # Typed values: without pyarrow, the stdlib transformer must still yield int/float.
    if a != [1, 2] or not all(isinstance(v, int) and not isinstance(v, bool) for v in a):
        print("BAD_A:" + repr(a))
        sys.exit(1)
    if c != [1.5, 2.5] or not all(isinstance(v, float) for v in c):
        print("BAD_C:" + repr(c))
        sys.exit(1)
    if b != ["x", "y"] or not all(isinstance(v, str) for v in b):
        print("BAD_B:" + repr(b))
        sys.exit(1)
    print("OK:TYPED")
    sys.exit(0)
except Exception as e:
    import traceback

    print("RUN_FAILED:" + type(e).__name__ + ":" + str(e))
    traceback.print_exc()
    sys.exit(1)
finally:
    os.remove(path)
"""


# ---------------------------------------------------------------------------
# Stdlib header discovery: CsvReader.get_column_names must work without pyarrow.
# ---------------------------------------------------------------------------
_BODY_GET_COLUMN_NAMES: str = """
import csv
import os
import sys
import tempfile

fd, path = tempfile.mkstemp(suffix=".csv")
os.close(fd)
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["A", "B"])
    writer.writerow(["1", "3"])
    writer.writerow(["2", "4"])

try:
    from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

    cols = list(CsvReader.get_column_names(path))
    print("COLS:" + ",".join(cols))
    if cols == ["A", "B"]:
        print("OK")
    sys.exit(0)
except Exception as e:
    print("FAILED:" + type(e).__name__ + ":" + str(e))
    sys.exit(1)
finally:
    os.remove(path)
"""


@pytest.mark.timeout(30)
def test_csv_reads_into_python_dict_without_pyarrow() -> None:
    """run_all on a CSV into PythonDict must succeed with pyarrow blocked.

    Fails today: CSV reading routes through ``pyarrow.csv``; under the block ``load_data``
    (and header discovery in matching) raises ImportError/AttributeError, so run_all never
    returns a dict-backed result and the ``OK`` sentinel is absent.
    """
    result = run_blocked(_BODY_RUN_ALL)

    assert result.returncode == 0, (
        "Reading a CSV into PythonDict crashed under blocked pyarrow (or values were not typed).\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "OK:TYPED" in result.stdout, (
        "Expected OK:TYPED sentinel (dict-backed, typed A=int/B=str/C=float values). "
        f"Got stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.timeout(30)
def test_get_column_names_uses_stdlib_without_pyarrow() -> None:
    """CsvReader.get_column_names must return ["A","B"] via stdlib with pyarrow blocked.

    Fails today: ``get_column_names`` calls ``pyarrow.csv.ReadOptions`` / ``read_csv``;
    with pyarrow blocked ``pyarrow_csv`` is None, so it raises and no ``OK`` is printed.
    """
    result = run_blocked(_BODY_GET_COLUMN_NAMES)

    assert result.returncode == 0, (
        "get_column_names crashed under blocked pyarrow.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "COLS:A,B" in result.stdout, (
        f"Expected COLS:A,B header sentinel. Got stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout.splitlines(), (
        f"Expected standalone OK sentinel. Got stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
