"""GitHub-flavored Markdown pipe tables of result frames and plan steps.

Shared by ``RunResult.summary`` and ``mloda demo``.
"""

from collections.abc import Iterable, Sequence
from typing import Any

PLAN_HEADER = ("step", "plugin", "features", "from")


def pipe_table(rows: Sequence[Sequence[str]]) -> list[str]:
    """Header row, dash separator, then the rest; ``|`` escaped, each column padded to its widest cell."""
    cells = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    widths = [max(len(cell) for cell in column) for column in zip(*cells, strict=True)]
    lines = [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) + " |" for row in cells
    ]
    return [lines[0], "|" + "|".join("-" * (width + 2) for width in widths) + "|", *lines[1:]]


def table_lines(names: Sequence[str], rows: Iterable[Sequence[Any]], hidden_rows: int = 0) -> list[str]:
    """Pipe table of ``str`` cells for every row given; ``hidden_rows > 0`` appends a ``... and N more rows`` line."""
    lines = pipe_table([list(names), *([str(value) for value in row] for row in rows)])
    if hidden_rows > 0:
        lines.append(f"... and {hidden_rows} more rows")
    return lines


def plan_cells(
    kind: str,
    feature_group: str | None,
    source_feature_group: str | None,
    join_type: str | None,
    features: Sequence[str],
    source: str | None,
    inputs: Sequence[str],
) -> tuple[str, str, str]:
    """The ``plugin``, ``features`` and ``from`` cells of one plan step."""
    if kind == "transform":
        return "transform", "", f"{source_feature_group} -> {feature_group}"
    if kind == "join":
        return "join", "", f"{feature_group} + {source_feature_group} ({join_type})"
    origin = f"source {source}" if source else ", ".join(inputs)
    return feature_group or kind, ", ".join(features), origin


def plan_table(steps: Iterable[tuple[int, Sequence[str]]]) -> list[str]:
    """One ``step | plugin | features | from`` row per (number, cells); no lines for an empty plan."""
    rows = [[str(number), *cells] for number, cells in steps]
    return pipe_table([list(PLAN_HEADER), *rows]) if rows else []


def result_and_plan(tables: Sequence[str], steps: Sequence[str]) -> str:
    """``Result``, the tables separated by a blank line, a blank line, ``Plan`` and the step lines; no trailing newline."""
    return "\n".join(["Result", "\n\n".join(tables), "", "Plan", *steps])
