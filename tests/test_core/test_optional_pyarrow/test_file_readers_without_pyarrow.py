"""Tests that file-reader modules import cleanly under blocked pyarrow and that their
load_data methods raise ImportError mentioning pyarrow (not a top-level crash).

Current state (red for import tests): each reader has a hard top-level
    from pyarrow import <submodule>
which causes ModuleNotFoundError at import time under the blocker, so the
IMPORTED sentinel is never printed and the test fails.
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
_BODY_CSV_IMPORT: str = """
import sys

try:
    from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
"""

_BODY_CSV_LOAD: str = """
import sys

try:
    from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__)
    sys.exit(0)

try:
    CsvReader.load_data("/nonexistent/path.csv", None)
    print("NO_RAISE")
except ImportError as e:
    if "pyarrow" in str(e).lower():
        print("IMPORTERROR")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__)
"""

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
_BODY_JSON_IMPORT: str = """
import sys

try:
    from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
"""

# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------
_BODY_PARQUET_IMPORT: str = """
import sys

try:
    from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
"""

# ---------------------------------------------------------------------------
# Feather
# ---------------------------------------------------------------------------
_BODY_FEATHER_IMPORT: str = """
import sys

try:
    from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
"""

# ---------------------------------------------------------------------------
# ORC
# ---------------------------------------------------------------------------
_BODY_ORC_IMPORT: str = """
import sys

try:
    from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
"""


@pytest.mark.timeout(30)
def test_csv_reader_imports_without_pyarrow() -> None:
    """CsvReader module must import successfully even when pyarrow is absent.

    Current (red): top-level 'from pyarrow import csv' raises ModuleNotFoundError
    so IMPORTED sentinel is never printed.
    """
    result = run_blocked(_BODY_CSV_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel after blocking pyarrow. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_csv_reader_load_data_raises_import_error_without_pyarrow() -> None:
    """CsvReader.load_data must raise ImportError mentioning pyarrow when pyarrow absent.

    Current (red): module fails to import at all, so this test can only be meaningful
    after the import test goes green (module restructured). Assert IMPORTERROR.
    """
    result = run_blocked(_BODY_CSV_LOAD)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_json_reader_imports_without_pyarrow() -> None:
    """JsonReader module must import successfully even when pyarrow is absent.

    Current (red): top-level 'from pyarrow import json' raises ModuleNotFoundError.
    """
    result = run_blocked(_BODY_JSON_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_parquet_reader_imports_without_pyarrow() -> None:
    """ParquetReader module must import successfully even when pyarrow is absent.

    Current (red): top-level 'from pyarrow import parquet' raises ModuleNotFoundError.
    """
    result = run_blocked(_BODY_PARQUET_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_feather_reader_imports_without_pyarrow() -> None:
    """FeatherReader module must import successfully even when pyarrow is absent.

    Current (red): top-level 'from pyarrow import feather' raises ModuleNotFoundError.
    """
    result = run_blocked(_BODY_FEATHER_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_orc_reader_imports_without_pyarrow() -> None:
    """OrcReader module must import successfully even when pyarrow is absent.

    Current (red): top-level 'from pyarrow import orc' raises ModuleNotFoundError.
    """
    result = run_blocked(_BODY_ORC_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
