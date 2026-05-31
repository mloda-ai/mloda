"""mkdocs hook: export the marimo example notebooks (.py) to .ipynb with outputs so
mkdocs-jupyter can render them. Regenerates only when the .py is newer than the .ipynb."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any


def _is_marimo_notebook(path: Path) -> bool:
    return "import marimo" in path.read_text(encoding="utf-8")


def on_pre_build(config: Any, **kwargs: Any) -> None:
    examples_dir = Path(config["docs_dir"]) / "examples"
    if not examples_dir.is_dir():
        return

    for source in sorted(examples_dir.rglob("*.py")):
        if not _is_marimo_notebook(source):
            continue

        notebook = source.with_suffix(".ipynb")
        if notebook.exists() and notebook.stat().st_mtime >= source.stat().st_mtime:
            continue

        subprocess.run(  # nosec B603
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(source),
                "-o",
                str(notebook),
                "--include-outputs",
                "--sort",
                "top-down",
                "-f",
            ],
            check=True,
        )
