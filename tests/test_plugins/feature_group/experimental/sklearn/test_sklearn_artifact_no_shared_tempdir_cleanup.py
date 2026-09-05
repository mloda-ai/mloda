"""No test may clean up artifact files it did not itself create by globbing the shared system temp dir."""

from __future__ import annotations

from pathlib import Path


SIBLING_TEST_FILE = Path(__file__).resolve().parent / "test_sklearn_artifact.py"

# A blanket glob against this pattern deletes every matching file in /tmp, including files
# written concurrently by other tests or pytest-xdist workers, not just files this test created.
UNSCOPED_GLOB_PATTERN = "/tmp/sklearn_artifact_*.joblib"  # nosec


def test_test_sklearn_artifact_has_no_unscoped_glob_cleanup() -> None:
    source = SIBLING_TEST_FILE.read_text()
    assert UNSCOPED_GLOB_PATTERN not in source, (
        f"{SIBLING_TEST_FILE.name} must not glob-delete '{UNSCOPED_GLOB_PATTERN}': "
        "this pattern matches artifact files owned by other, concurrently-running tests/workers, "
        "not just files this test created. Clean up only the exact path(s) this test saved."
    )
