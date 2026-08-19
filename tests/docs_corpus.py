"""Shared file list and fence-tag vocabulary for the docs/docs guard tests."""

import re
from pathlib import Path

# Anchored to this file, not the cwd: a relative glob run from elsewhere yields nothing,
# so these guards would pass while checking zero files (issue #937).
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs" / "docs"

RUNNABLE_TAG = "python"
ILLUSTRATIVE_TAG = "py"
OUTPUT_TAG = "text"

PYTHON_LOOKING = re.compile(r"i?py(thon)?[0-9]*(-repl)?|pycon", re.IGNORECASE)

PYTHON_BLOCK_PATTERN = re.compile(rf"```(?:{RUNNABLE_TAG}|{ILLUSTRATIVE_TAG})\n(.*?)```", re.DOTALL)


def doc_files(root: Path = DOCS_ROOT) -> list[Path]:
    """Every markdown file under root; never an empty list, so a bad path fails loudly at import time."""
    files = sorted(root.rglob("*.md"))
    if not files:
        raise RuntimeError(f"no markdown files found under {root}, doc guard tests would check nothing")
    return files


def doc_id(fpath: Path) -> str:
    """Repo-relative posix path used as a pytest parametrize id."""
    return fpath.relative_to(REPO_ROOT).as_posix()


def is_python_fence(info: str) -> bool:
    """Whether a fence's info string (the text after the opening backticks) tags it as Python."""
    parts = info.strip().split()
    tag = parts[0] if parts else ""
    return bool(PYTHON_LOOKING.fullmatch(tag))
