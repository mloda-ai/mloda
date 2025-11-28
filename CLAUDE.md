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
   - Run `git add .` to stage all changes for that phase
3. **If tox fails**:
   - Fix the issues before proceeding
   - Do NOT mark phase as complete
   - Do NOT run `git add .`

Each phase should be a clean, validated checkpoint with all tests passing and changes staged.

## Self-Improvement and Learning

**CRITICAL**: If agent behavior is unexpected or incorrect:

1. **Update Agent Configuration**: Modify `.claude/agents/red-agent.md` or `.claude/agents/green-agent.md` to refine instructions, constraints, or workflow
2. **Update This File**: Modify `CLAUDE.md` to clarify orchestration rules or add missing guidance
3. **Document Changes**: Briefly explain what was learned and why the change improves behavior

This enables continuous learning and improvement of the TDD workflow based on actual usage patterns.

## Important: Refer to .clinerules

**ALWAYS read .clinerules first** - it contains critical memory bank instructions and coding guidelines. The .clinerules file maintains:
- Complete memory bank structure in `memory-bank/` directory
- Coding conventions and test requirements
- Workflow instructions for planning and execution
