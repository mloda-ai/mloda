## Development Setup (for Contributors)

Thanks for your interest in contributing to mloda. The canonical contributor guide is [CONTRIBUTING.md](https://github.com/mloda-ai/mloda/blob/main/CONTRIBUTING.md), and all participants are expected to follow our [Code of Conduct](https://github.com/mloda-ai/mloda/blob/main/CODE_OF_CONDUCT.md). This page covers two topics that are specific to working on the docs and the dev environment.

#### Dev Container (Optional)

Prerequisite: Docker.

- Open the project in a dev container (in VS Code or a compatible tool).
- The dev container includes all necessary dependencies and tools to work on mloda.

If you don't use Docker, follow the local setup in [CONTRIBUTING.md](https://github.com/mloda-ai/mloda/blob/main/CONTRIBUTING.md#local-development-setup) (`uv sync --all-extras`, then `source .venv/bin/activate`).

#### Building the docs locally

After completing the setup in CONTRIBUTING.md (`uv sync --all-extras`), you can preview docs changes with:

```bash
mkdocs serve --config-file docs/mkdocs.yml
```

The build executes the example notebooks to capture their outputs, so it needs the example notebooks' compute backends. `uv sync --all-extras` (or installing `mloda[docs]`) provides them.

#### Authoring example notebooks (marimo)

The example notebooks under `docs/docs/examples/` are [marimo](https://marimo.io) notebooks stored as plain Python (`.py`), so they stay diff-friendly and lintable. Edit one with:

```bash
marimo edit docs/docs/examples/base_usage.py
```

A few things to keep in mind:

- marimo is reactive: a variable may be defined in only one cell, and names prefixed with `_` are private to their cell. Give values that are shared across cells distinct, non-underscore names.
- A notebook is valid when it runs end to end as a script: `python docs/docs/examples/base_usage.py` must exit `0`. This is exactly what CI checks.
- Run `ruff format` on the file before committing.
- The rendered `.ipynb` (with executed outputs) is generated at build time by `docs/marimo_export_hook.py` and is git-ignored. Do not commit `.ipynb` files; add new notebooks as `.py` and reference them with their `.ipynb` name in `docs/mkdocs.yml`.

#### Where to start

- Plugin contributions: see the [mloda-registry guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides) and the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template).
- Core framework contributions: browse issues labeled [`good first issue`](https://github.com/mloda-ai/mloda/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [`help wanted`](https://github.com/mloda-ai/mloda/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
- Bugs and feature requests: open an [issue](https://github.com/mloda-ai/mloda/issues/).
