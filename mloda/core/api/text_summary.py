"""Plain-text rendering of result tables and plan steps, shared by ``RunResult.summary`` and ``mloda demo``."""

from collections.abc import Iterable, Sequence
from typing import Any


def aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-align each column to its widest cell, two spaces between columns, trailing whitespace stripped."""
    widths = [max(len(cell) for cell in column) for column in zip(*rows, strict=True)]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows]


def table_lines(names: Sequence[str], rows: Iterable[Sequence[Any]], hidden_rows: int = 0) -> list[str]:
    """Header plus ``str`` cells for every row given; ``hidden_rows > 0`` appends a ``... and N more rows`` line."""
    lines = aligned([list(names), *([str(value) for value in row] for row in rows)])
    if hidden_rows > 0:
        lines.append(f"... and {hidden_rows} more rows")
    return lines


def plan_detail(features: Sequence[str], source: str | None, inputs: Sequence[str]) -> str:
    """The step's feature names, then its api_data source alias or, failing that, its input feature names."""
    detail = ", ".join(features)
    if source:
        return f"{detail}  (source {source})"
    if inputs:
        return f"{detail}  (from {', '.join(inputs)})"
    return detail


def plan_lines(steps: Iterable[tuple[int, str, str]]) -> list[str]:
    """One aligned ``step N  label  detail`` line per (number, label, detail)."""
    return aligned([[f"step {number}", label, detail] for number, label, detail in steps])


def result_and_plan(tables: Sequence[str], steps: Sequence[str]) -> str:
    """``Result``, the tables separated by a blank line, a blank line, ``Plan`` and the step lines; no trailing newline."""
    return "\n".join(["Result", "\n\n".join(tables), "", "Plan", *steps])
