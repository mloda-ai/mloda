"""Tests that file format group modules import cleanly under blocked pyarrow and that
CsvFG.load_neutral resolves a backend-neutral FileSource descriptor without pyarrow.
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
    from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
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
    from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
    from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
except Exception as e:
    print("IMPORT_FAILED:" + type(e).__name__)
    sys.exit(0)

features = FeatureSet()
features.add(Feature("A"))

result = CsvFG.load_neutral(SourceMatch(source="/nonexistent/path.csv", access="/nonexistent/path.csv"), features)
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
    from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
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
    from mloda_plugins.feature_group.input_data.file_formats.parquet_fg import ParquetFG
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
    from mloda_plugins.feature_group.input_data.file_formats.feather_fg import FeatherFG
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
    from mloda_plugins.feature_group.input_data.file_formats.orc_fg import OrcFG
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
"""


@pytest.mark.timeout(30)
def test_csv_fg_imports_without_pyarrow() -> None:
    """CsvFG module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_CSV_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel after blocking pyarrow. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_csv_fg_load_neutral_returns_file_source_without_pyarrow() -> None:
    """CsvFG.load_neutral returns a lightweight FileSource descriptor even when pyarrow
    is absent; a per-framework transformer materializes it later.
    """
    result = run_blocked(_BODY_CSV_LOAD)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "FILESOURCE" in result.stdout, (
        f"Expected FILESOURCE sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_json_fg_imports_without_pyarrow() -> None:
    """JsonFG module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_JSON_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_parquet_fg_imports_without_pyarrow() -> None:
    """ParquetFG module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_PARQUET_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_feather_fg_imports_without_pyarrow() -> None:
    """FeatherFG module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_FEATHER_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_orc_fg_imports_without_pyarrow() -> None:
    """OrcFG module must import successfully even when pyarrow is absent."""
    result = run_blocked(_BODY_ORC_IMPORT)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Feather: match declines with a rejection naming mloda[pyarrow], load_neutral raises ImportError
# ---------------------------------------------------------------------------
_BODY_FEATHER_DECLINE_AND_LOAD: str = """
import os
import sys
import tempfile

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.feature_group.input_data.file_formats.feather_fg import FeatherFG

fd, path = tempfile.mkstemp(suffix=".feather")
os.close(fd)
with open(path, "wb") as f:
    f.write(b"placeholder")

window = {}
token = MATCH_REJECTION_REASONS.set(window)
try:
    matched = FeatherFG.match_feature_group_criteria("a", Options(), DataAccessCollection(files={path}))
finally:
    MATCH_REJECTION_REASONS.reset(token)

features = FeatureSet()
features.add(Feature("a"))

try:
    FeatherFG.load_neutral(SourceMatch(source=path, access=path), features)
    load_result = "NO_RAISE"
except ImportError as e:
    load_result = "IMPORT_ERROR:" + str(e)
except Exception as e:
    load_result = "OTHER:" + type(e).__name__ + ":" + str(e)
finally:
    os.remove(path)

print("MATCH:" + str(matched))
rejection = window.get("FeatherFG")
print("REJECTION:" + ("NONE" if rejection is None else rejection.reason))
print("LOAD:" + load_result)
"""


@pytest.mark.timeout(30)
def test_feather_fg_declines_with_an_install_hint_and_load_raises_without_pyarrow() -> None:
    """Without pyarrow the Feather group declines at match time, recording why, and load_neutral raises ImportError."""
    result = run_blocked(_BODY_FEATHER_DECLINE_AND_LOAD)
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "MATCH:False" in result.stdout, f"Expected a decline. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    rejection_lines = [line for line in result.stdout.splitlines() if line.startswith("REJECTION:")]
    assert rejection_lines and "mloda[pyarrow]" in rejection_lines[0], (
        f"Expected a rejection naming mloda[pyarrow]. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
    assert "LOAD:IMPORT_ERROR:" in result.stdout, (
        f"Expected import-error sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
    assert "mloda[pyarrow]" in result.stdout


# ---------------------------------------------------------------------------
# YAML: the module must import without PyYAML, and only raise once produce_document runs
# ---------------------------------------------------------------------------
_BODY_YAML_IMPORT_AND_LOAD: str = """
import os
import sys
import tempfile

try:
    from mloda_plugins.feature_group.input_data.read_files.yaml_document_reader import YamlDocumentReader
    print("IMPORTED")
except ImportError as e:
    print("IMPORT_ERROR:" + str(e))
    sys.exit(0)
except Exception as e:
    print("IMPORT_OTHER:" + type(e).__name__ + ":" + str(e))
    sys.exit(0)

fd, path = tempfile.mkstemp(suffix=".yaml")
os.close(fd)
with open(path, "w", encoding="utf-8") as f:
    f.write("key: value\\n")

try:
    YamlDocumentReader.produce_document(path)
    load_result = "NO_RAISE"
except ImportError as e:
    load_result = "IMPORT_ERROR:" + str(e)
except Exception as e:
    load_result = "OTHER:" + type(e).__name__ + ":" + str(e)
finally:
    os.remove(path)

print("LOAD:" + load_result)
"""


@pytest.mark.timeout(30)
def test_yaml_document_reader_imports_without_yaml_and_load_raises() -> None:
    result = run_blocked(_BODY_YAML_IMPORT_AND_LOAD, module="yaml")
    assert result.returncode == 0, f"Body crashed.\nstderr:\n{result.stderr}"
    assert "IMPORTED" in result.stdout, (
        f"Expected IMPORTED sentinel after blocking yaml. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
    assert "LOAD:IMPORT_ERROR:" in result.stdout, (
        f"Expected import-error sentinel from produce_document. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
    assert "mloda[yaml]" in result.stdout, (
        f"Expected mloda[yaml] install hint. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
