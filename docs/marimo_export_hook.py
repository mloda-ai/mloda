"""mkdocs hook that renders marimo notebooks.

The example notebooks under ``docs/docs/examples`` are authored as marimo (``.py``)
files so they stay diff-friendly in version control. This hook exports each one to a
Jupyter ``.ipynb`` with executed outputs before the build, so the ``mkdocs-jupyter``
plugin can render them exactly as before. The generated ``.ipynb`` files are build
artifacts (git-ignored); only the ``.py`` sources are tracked.

The export runs in ``on_pre_build`` (before mkdocs collects files) and regenerates a
notebook only when its ``.py`` source is newer than the previously generated ``.ipynb``,
so ``mkdocs serve`` does not rebuild in a loop.
"""

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
