"""CLAUDE.md and AGENTS.md are one instruction set: every line not tagged claude-only must be identical."""

import difflib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Naming both files here makes tests/test_ci_paths_ignore.py treat them as protected: a change to
# either must run the gate, so neither may stay in the ci.yaml docs-only paths-ignore list.
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"

CLAUDE_ONLY_MARKER = "<!-- claude-only -->"
SHARED_HEADING = "### mypy iteration notes"


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _is_claude_only(line: str) -> bool:
    return line.rstrip().endswith(CLAUDE_ONLY_MARKER)


def _shared_lines() -> list[str]:
    """The part of CLAUDE.md that is not Claude-specific, so it belongs to both agents."""
    return [line for line in _lines(CLAUDE_MD) if not _is_claude_only(line)]


def _contains_heading(lines: list[str], heading: str) -> bool:
    return any(line.rstrip() == heading for line in lines)


def test_shared_part_of_claude_md_matches_agents_md() -> None:
    shared = _shared_lines()
    agents = _lines(AGENTS_MD)
    diff = "\n".join(
        difflib.unified_diff(shared, agents, fromfile="CLAUDE.md (shared part)", tofile="AGENTS.md", lineterm="")
    )
    assert shared == agents, (
        f"CLAUDE.md and AGENTS.md have drifted:\n{diff}\n\n"
        "Keep the two files identical. If a line is genuinely only true for Claude Code, end it with the "
        f"{CLAUDE_ONLY_MARKER} marker; every other line must appear in AGENTS.md verbatim."
    )


def test_agents_md_has_no_claude_only_marker() -> None:
    tagged = [
        f"{number}: {line}" for number, line in enumerate(_lines(AGENTS_MD), start=1) if CLAUDE_ONLY_MARKER in line
    ]
    found = "\n".join(tagged)
    assert not tagged, (
        f"AGENTS.md must never contain the {CLAUDE_ONLY_MARKER} marker: the marker flags content that stays "
        f"out of AGENTS.md, so a tagged line there contradicts itself. Drop the marker or keep the line in "
        f"CLAUDE.md only:\n{found}"
    )


def test_mypy_iteration_notes_stay_shared() -> None:
    sources = (("CLAUDE.md (shared part)", _shared_lines()), ("AGENTS.md", _lines(AGENTS_MD)))
    missing = [name for name, lines in sources if not _contains_heading(lines, SHARED_HEADING)]
    assert not missing, (
        f"The '{SHARED_HEADING}' subsection must be in both files, missing from: {missing}. It describes the "
        "tox gate rather than Claude Code, so it must not be tagged claude-only and must not be dropped."
    )
