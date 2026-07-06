"""RED test: a ``FileSource`` descriptor with no resolvable materialization chain must
raise an ACTIONABLE error, not silently leak the descriptor.

When pyarrow is unavailable, a non-PythonDict framework (here ``PandasDataFrame``) cannot
materialize a CSV ``FileSource`` (the only path runs through ``pa.Table``). The fix must
raise an error whose message names the descriptor/target and points at installing pyarrow
(substring ``pyarrow``), instead of leaking the ``FileSource`` object downstream.

The body runs in a subprocess with pyarrow blocked via ``sys.meta_path``.

Fails today: with pyarrow blocked and no chain, ``apply_compute_framework_transformer``
returns ``None`` and ``PandasDataFrame.transform`` falls through to a generic
"Data <FileSource> is not supported by PandasDataFrame" ValueError that never mentions
pyarrow, so the ``OK:ACTIONABLE`` sentinel is absent.
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
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

try:
    from mloda.user import ParallelizationMode
    from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    fw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    source = FileSource(path=path, format="csv", columns=("A", "B"))

    try:
        result = fw.transform(source, {"A", "B"})
    except Exception as e:
        msg = str(e)
        if "pyarrow" in msg.lower() and "FileSource" in msg:
            print("OK:ACTIONABLE")
        else:
            print("WRONG_MESSAGE:" + type(e).__name__ + ":" + msg)
        sys.exit(0)

    # No error at all: the FileSource leaked instead of raising.
    print("NO_RAISE:" + type(result).__name__ + ":" + repr(result)[:120])
    sys.exit(0)
except Exception as e:
    import traceback

    print("SETUP_FAILED:" + type(e).__name__ + ":" + str(e))
    traceback.print_exc()
    sys.exit(1)
finally:
    os.remove(path)
"""


@pytest.mark.timeout(30)
def test_file_source_without_chain_raises_actionable_pyarrow_error() -> None:
    result = run_blocked(_BODY)

    assert result.returncode == 0, (
        f"Body crashed unexpectedly.\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "OK:ACTIONABLE" in result.stdout, (
        "Expected an actionable error naming the FileSource descriptor and mentioning "
        "pyarrow when no materialization chain can be resolved.\n"
        f"Got stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
