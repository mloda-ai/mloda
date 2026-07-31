"""CLAUDE.md and AGENTS.md are one instruction set for different agents, so the two files must be identical."""

import difflib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Naming both files here makes tests/test_ci_paths_ignore.py treat them as protected: a change to
# either must run the gate, so neither may stay in the ci.yaml docs-only paths-ignore list.
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"


def _lines(path: Path) -> list[str]:
    assert path.is_file(), f"{path.name} must exist at the project root: each agent reads only its own file."
    # Bytes, not read_text(): universal-newline translation would let a CRLF rewrite compare equal.
    return path.read_bytes().decode("utf-8").split("\n")


def _diff(claude: list[str], agents: list[str]) -> str:
    """Diff of repr-ed lines, so whitespace-only drift is not printed as two identical-looking lines."""
    return "\n".join(
        difflib.unified_diff(
            [repr(line) for line in claude],
            [repr(line) for line in agents],
            fromfile="CLAUDE.md",
            tofile="AGENTS.md",
            lineterm="",
        )
    )


def test_claude_md_and_agents_md_are_identical() -> None:
    claude, agents = _lines(CLAUDE_MD), _lines(AGENTS_MD)
    assert claude == agents, (
        f"CLAUDE.md and AGENTS.md have drifted:\n{_diff(claude, agents)}\n\n"
        "Neither file carries content of its own: whatever belongs in one belongs in the other, verbatim."
    )
