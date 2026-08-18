"""Shared file list for the docs/docs guard tests."""

from pathlib import Path

# Anchored to this file, not the cwd: a relative glob run from elsewhere yields nothing,
# so these guards would pass while checking zero files (issue #937).
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs" / "docs"


def doc_files() -> list[Path]:
    """Every markdown file under DOCS_ROOT; never an empty list, so a bad path fails loudly at import time."""
    files = sorted(DOCS_ROOT.rglob("*.md"))
    if not files:
        raise RuntimeError(f"no markdown files found under {DOCS_ROOT}, doc guard tests would check nothing")
    return files
