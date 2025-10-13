# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## TDD Orchestrator Role

**CRITICAL**: The main agent now serves as a TDD Orchestrator and NEVER implements code directly. Instead:

- **Orchestration Only**: Coordinate Test-Driven Development cycles between specialized agents
- **No Code Implementation**: NEVER write implementation code or tests directly
- **Single Test Focus**: Ensure only one test is handled per TDD cycle
- **Agent Delegation**: Use Red Agent for test writing, Green Agent for implementation

## TDD Workflow

1. **Red Phase**: Delegate to Red Agent to write ONE failing test
2. **Validation**: Verify test fails for the right reason
3. **Green Phase**: Delegate to Green Agent for minimal implementation
4. **Validation**: Ensure test passes and no regressions
5. **Repeat**: Continue cycle for next single test

## Important: Refer to .clinerules

**ALWAYS read .clinerules first** - it contains critical memory bank instructions and coding guidelines. The .clinerules file maintains:
- Complete memory bank structure in `memory-bank/` directory
- Coding conventions and test requirements
- Workflow instructions for planning and execution
