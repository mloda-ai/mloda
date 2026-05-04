# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## TDD Orchestrator Role

**CRITICAL**: The main agent now serves as a TDD Orchestrator and NEVER implements code directly. Instead:

- **Orchestration Only**: Coordinate Test-Driven Development cycles between specialized agents
- **No Code Implementation**: NEVER write implementation code or tests directly
- **Agent Delegation**: Use Red Agent for test writing, Green Agent for implementation

## TDD Workflow

1. **Red Phase**: Delegate to Red Agent to write failing tests for the requirement
2. **Validation**: Verify tests fail for the right reasons
3. **Green Phase**: Delegate to Green Agent for minimal implementation
4. **Validation**: Ensure tests pass and no regressions
5. **Repeat**: Continue cycle for next requirement

## Deadlock Protection

**CRITICAL**: If Red or Green agents get stuck or fail repeatedly:

1. **Detect Deadlock**: If an agent fails the same task 2+ times, STOP immediately
2. **Do NOT Loop**: Never retry the same failing operation more than twice
3. **Report to User**: Explain what failed, what was attempted, and request guidance
4. **User Decision**: Let the user decide whether to:
   - Modify the approach
   - Update agent instructions
   - Manually intervene
   - Skip the problematic step

**Never continue TDD cycles if agents are stuck** - this wastes resources and indicates a fundamental issue that requires human intervention.

## Phase Completion Protocol

When working with `memory-bank/todo.md` that contains phases:

1. **After completing each phase**: Run `tox` to validate all tests pass
2. **If tox passes**:
   - Mark the phase as complete (tick the checkbox) in todo.md
3. **If tox fails**:
   - Fix the issues before proceeding
   - Do NOT mark phase as complete

Each phase should be a clean, validated checkpoint with all tests passing and changes staged.

## Self-Improvement and Learning

**CRITICAL**: If agent behavior is unexpected or incorrect:

1. **Update Agent Configuration**: Modify `.claude/agents/red-agent.md` or `.claude/agents/green-agent.md` to refine instructions, constraints, or workflow
2. **Update This File**: Modify `CLAUDE.md` to clarify orchestration rules or add missing guidance
3. **Document Changes**: Briefly explain what was learned and why the change improves behavior

This enables continuous learning and improvement of the TDD workflow based on actual usage patterns.

## Setup

```bash
uv sync --all-extras
source .venv/bin/activate
```

## Coding Instructions

- Never put code into `__init__.py` files
- Keep documentation to the necessary lines
- Run tests after creation using `tox`
- Avoid try/except blocks
- Never mention agents in commit messages, PR descriptions, or any user-facing text (no `Co-Authored-By` agent lines, no agent names)

## Project Practices

`tox` is the gate. It runs `pytest -n 8 --timeout=10`, then `ruff format --check`, `ruff check`, a `pip-licenses` allowlist check, `mypy --strict --ignore-missing-imports`, and `bandit`. All of these must pass before a PR is mergeable.

- **Python**: supported range is `>=3.10,<3.14`; CI matrixes 3.10, 3.11, 3.12, 3.13.
- **Type hints**: use modern forms (`list[str]`, `dict[str, int]`, `X | None`). Ruff enforces this via `UP006` and `UP007`.
- **Formatting**: ruff format with line length 120.
- **Tests**: every new feature or bug fix must come with tests; follow the patterns in the existing `tests/` tree. Tests must be parallel-safe (pytest-xdist) and finish under the 10-second timeout. The default tox env asserts `EXPECTED_SKIP_COUNT=147`; if a test you add is skipped, update the count or unskip it.
- **Supply chain**: `[tool.uv] exclude-newer = "7 days"` in `pyproject.toml` defers new dependency releases by 7 days. Do not edit this without a reason.
- **Licenses**: dependencies must satisfy the allowlist in `tox.ini` (Apache-2.0, BSD, MIT, MPL-2.0, PSF, ISC, LGPLv2+). Adding a dependency with a non-listed license fails tox.
- **Commits**: use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `minor:`). semantic-release computes the next version. This project deviates from the standard: the minor version (middle number) bumps only on `minor:` commits; `feat:` is treated as a patch bump.

## Issue Creation

When filing a GitHub issue (via `gh issue create` or otherwise), follow the structure in `.github/ISSUE_TEMPLATE/issue.yml`:

- Summary in one sentence
- Reproduction (for bugs) or motivation (for features)
- Code pointers if relevant (`file:line`)
- Definition of done if scoped (what counts as complete)

Issues that meet this bar are eligible for the `good first issue` label without further sharpening.

## Memory Bank

The `memory-bank/` directory contains project context documentation. Read relevant files at the start of tasks to understand the project.

### Core Files
| File | Purpose |
|------|---------|
| `projectbrief.md` | Foundation document - core requirements and goals |
| `productContext.md` | Why the project exists, problems solved, user experience goals |
| `activeContext.md` | Current work focus, recent changes, next steps |
| `systemPatterns.md` | System architecture, design patterns, component relationships |
| `techContext.md` | Technologies, development setup, constraints, dependencies |
| `progress.md` | What works, what's left, current status, known issues |

### Hierarchy
Files build upon each other: `projectbrief.md` → `productContext.md`/`systemPatterns.md`/`techContext.md` → `activeContext.md` → `progress.md`
