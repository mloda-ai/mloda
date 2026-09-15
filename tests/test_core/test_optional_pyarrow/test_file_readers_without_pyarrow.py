"""Tests that file-reader modules import cleanly under blocked pyarrow and that
CsvReader.load_data resolves a backend-neutral FileSource descriptor without pyarrow.
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
    from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
    from mloda.provider import FeatureSet
    from mloda.user import Feature
    from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__)
    sys.exit(0)

features = FeatureSet()
features.add(Feature("A"))

result = CsvReader.load_data("/nonexistent/path.csv", features)
if isinstance(result, FileSource) and result.format == "csv" and result.columns == ("A",):
    print("FILESOURCE")
else:
    print("WRONG:" + type(result).__name__)
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
    """CsvReader module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_CSV_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel after blocking pyarrow. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_csv_reader_load_data_returns_file_source_without_pyarrow() -> None:
    """CsvReader.load_data returns a lightweight FileSource descriptor even when pyarrow
    is absent; a per-framework transformer materializes it later.
    """
    result = run_blocked(_BODY_CSV_LOAD)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "FILESOURCE" in result.stdout, (
        f"Expected FILESOURCE sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_json_reader_imports_without_pyarrow() -> None:
    """JsonReader module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_JSON_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_parquet_reader_imports_without_pyarrow() -> None:
    """ParquetReader module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_PARQUET_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_feather_reader_imports_without_pyarrow() -> None:
    """FeatherReader module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_FEATHER_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_orc_reader_imports_without_pyarrow() -> None:
    """OrcReader module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_ORC_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Feather: match still assumes a plain name present, load_data raises ImportError
# ---------------------------------------------------------------------------
_BODY_FEATHER_MATCH_AND_LOAD: str = """
import sys

from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader

match_result = FeatherReader.match_read_file_data_access(["/nonexistent/x.feather"], ["a"])

features = FeatureSet()
features.add(Feature("a"))

try:
    FeatherReader.load_data("/nonexistent/x.feather", features)
    load_result = "NO_RAISE"
except ImportError as e:
    load_result = "IMPORT_ERROR:" + str(e)
except Exception as e:
    load_result = "OTHER:" + type(e).__name__ + ":" + str(e)

print("MATCH:" + str(match_result))
print("LOAD:" + load_result)
"""


@pytest.mark.timeout(30)
def test_feather_reader_match_and_load_without_pyarrow() -> None:
    """Regression guard: real pyarrow absence still matches the plain name and load_data raises ImportError."""
    result = run_blocked(_BODY_FEATHER_MATCH_AND_LOAD)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "MATCH:/nonexistent/x.feather" in result.stdout, (
        f"Expected match sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
    assert "LOAD:IMPORT_ERROR:" in result.stdout, (
        f"Expected import-error sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
    assert "mloda[pyarrow]" in result.stdout, (
        f"Expected mloda[pyarrow] install hint. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
