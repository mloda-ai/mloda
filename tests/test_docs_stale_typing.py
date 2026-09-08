"""Old-style Union/Optional/Set/Type generics swept from docs/docs/in_depth and the marimo examples.

None of these words legitimately appear bracketed elsewhere in this doc tree, so a raw text scan is
enough to catch them, no fence parsing needed.
"""

import re
from pathlib import Path

import pytest

from tests.docs_corpus import DOCS_ROOT, REPO_ROOT, doc_files, doc_id

STALE_TOKENS = ("Union", "Optional", "Set", "Type")

TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {token: re.compile(rf"\b{token}\[") for token in STALE_TOKENS}

# Marimo notebook scripts whose mo.md(r\"\"\"...\"\"\") prose blocks carry the same stale forms.
EXAMPLE_FILES = [
    REPO_ROOT / "docs" / "docs" / "examples" / "base_usage.py",
    REPO_ROOT / "docs" / "docs" / "examples" / "mloda_basics" / "3_ml_data_feature_feature_groups.py",
    REPO_ROOT / "docs" / "docs" / "examples" / "mloda_basics" / "4_ml_data_providers_user_steward.py",
]

TARGET_FILES = doc_files(DOCS_ROOT / "in_depth") + EXAMPLE_FILES


def _stale_token_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    counts = {token: len(pattern.findall(text)) for token, pattern in TOKEN_PATTERNS.items()}
    return {token: count for token, count in counts.items() if count}


@pytest.mark.parametrize("path", TARGET_FILES, ids=[doc_id(path) for path in TARGET_FILES])
def test_no_stale_typing_forms_remain(path: Path) -> None:
    counts = _stale_token_counts(path)
    found = ", ".join(f"{token}[ x{count}" for token, count in counts.items())
    assert not counts, (
        f"{doc_id(path)} still uses old-style typing form(s): {found}. "
        f"Replace with PEP 604/585 syntax (X | Y, X | None, set[X], type[X])."
    )
