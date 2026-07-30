"""CLAUDE.md and AGENTS.md are one instruction set: every line not tagged claude-only must be identical."""

import difflib
from collections.abc import Callable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Naming both files here makes tests/test_ci_paths_ignore.py treat them as protected: a change to
# either must run the gate, so neither may stay in the ci.yaml docs-only paths-ignore list.
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"

CLAUDE_ONLY_MARKER = "<!-- claude-only -->"
SHARED_HEADING = "### mypy iteration notes"

# Every CLAUDE.md line allowed out of the shared instruction set, by prefix: rewording a bullet body
# stays free, tagging a new line does not. Widening this is the visible signal that content stopped
# being shared with AGENTS.md.
CLAUDE_ONLY_PREFIXES: tuple[str, ...] = ("- **Session root**:",)


def _lines(path: Path) -> list[str]:
    assert path.is_file(), (
        f"{path.name} must exist at the project root: CLAUDE.md and AGENTS.md are one instruction set and "
        "each agent reads only its own file, so dropping one drops every shared rule for that agent."
    )
    # Bytes plus split("\n"), not read_text().splitlines(): universal-newline translation and splitlines()
    # each hide drift that `diff CLAUDE.md AGENTS.md` reports (CRLF, U+2028, a lost trailing newline).
    return path.read_bytes().decode("utf-8").split("\n")


def _is_claude_only(line: str) -> bool:
    return line.rstrip().endswith(CLAUDE_ONLY_MARKER)


def _shared_lines() -> list[str]:
    """The part of CLAUDE.md that is not Claude-specific, so it belongs to both agents."""
    return [line for line in _lines(CLAUDE_MD) if not _is_claude_only(line)]


def _numbered(path: Path, keep: Callable[[str], bool]) -> list[str]:
    return [f"{number}: {line}" for number, line in enumerate(_lines(path), start=1) if keep(line)]


def _section_body(lines: list[str], heading: str) -> list[str]:
    """The lines below `heading`, up to the next markdown heading. Empty when `heading` is absent."""
    body: list[str] = []
    seen = False
    for line in lines:
        if not seen:
            seen = line.rstrip() == heading
        elif line.startswith("#"):
            break
        else:
            body.append(line)
    return body


def _diff(shared: list[str], agents: list[str]) -> str:
    """Diff of repr-ed lines, so whitespace-only drift is not reported as two identical-looking lines."""
    return "\n".join(
        difflib.unified_diff(
            [repr(line) for line in shared],
            [repr(line) for line in agents],
            fromfile="CLAUDE.md (shared part)",
            tofile="AGENTS.md",
            lineterm="",
        )
    )


def test_shared_part_of_claude_md_matches_agents_md() -> None:
    shared = _shared_lines()
    agents = _lines(AGENTS_MD)
    assert shared == agents, (
        f"CLAUDE.md and AGENTS.md have drifted:\n{_diff(shared, agents)}\n\n"
        "AGENTS.md carries no content of its own: whatever belongs there belongs in CLAUDE.md too. Keep the "
        "two files identical; only a line that is genuinely true for Claude Code alone may stay out, by "
        f"ending it with the {CLAUDE_ONLY_MARKER} marker and listing its prefix in CLAUDE_ONLY_PREFIXES."
    )


def test_agents_md_has_no_claude_only_marker() -> None:
    tagged = _numbered(AGENTS_MD, lambda line: CLAUDE_ONLY_MARKER in line)
    assert not tagged, (
        f"AGENTS.md must never contain the {CLAUDE_ONLY_MARKER} marker: the marker flags content that stays "
        "out of AGENTS.md, so a tagged line there contradicts itself. Drop the marker or keep the line in "
        "CLAUDE.md only:\n" + "\n".join(tagged)
    )


def test_claude_only_marker_ends_the_line() -> None:
    misplaced = _numbered(CLAUDE_MD, lambda line: CLAUDE_ONLY_MARKER in line and not _is_claude_only(line))
    assert not misplaced, (
        f"The {CLAUDE_ONLY_MARKER} marker must be the last thing on the line. On these CLAUDE.md lines it sits "
        "somewhere else, where it excludes nothing: move it to the end of the line, or keep the line in "
        "AGENTS.md verbatim:\n" + "\n".join(misplaced)
    )


def test_claude_only_lines_match_the_allowlist() -> None:
    tagged = [line for line in _lines(CLAUDE_MD) if _is_claude_only(line)]
    unlisted = [line for line in tagged if not line.startswith(CLAUDE_ONLY_PREFIXES)]
    assert not unlisted, (
        "Tagging a line claude-only takes it out of the instruction set every non-Claude agent reads, so the "
        "marker alone is not enough. Add its prefix to CLAUDE_ONLY_PREFIXES in this file, so the diff shows "
        "content leaving the shared contract:\n" + "\n".join(unlisted)
    )
    unused = [prefix for prefix in CLAUDE_ONLY_PREFIXES if not any(line.startswith(prefix) for line in tagged)]
    assert not unused, (
        f"CLAUDE_ONLY_PREFIXES entries that tag no CLAUDE.md line: {unused}. Drop the stale entry so the "
        "allowlist keeps naming exactly the lines that are not shared with AGENTS.md."
    )


def test_mypy_iteration_notes_stay_shared() -> None:
    sources = (("CLAUDE.md (shared part)", _shared_lines()), ("AGENTS.md", _lines(AGENTS_MD)))
    bodies = {name: _section_body(lines, SHARED_HEADING) for name, lines in sources}
    empty = [name for name, body in bodies.items() if not any(line.strip() for line in body)]
    assert not empty, (
        f"The '{SHARED_HEADING}' subsection must carry its body in both files, missing or empty in: {empty}. "
        "It describes the tox gate rather than Claude Code, so it must not be tagged claude-only, dropped, or "
        "reduced to a bare heading."
    )
