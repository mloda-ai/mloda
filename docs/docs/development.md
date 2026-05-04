## Development Setup (for Contributors)

Thanks for your interest in contributing to mloda. The canonical contributor guide is [CONTRIBUTING.md](https://github.com/mloda-ai/mloda/blob/main/CONTRIBUTING.md), and all participants are expected to follow our [Code of Conduct](https://github.com/mloda-ai/mloda/blob/main/CODE_OF_CONDUCT.md). This page covers two topics that are specific to working on the docs and the dev environment.

#### Dev Container (Optional)

Prerequisite: Docker.

- Open the project in a dev container (in VS Code or a compatible tool).
- The dev container includes all necessary dependencies and tools to work on mloda.

If you don't use Docker, follow the local setup in [CONTRIBUTING.md](https://github.com/mloda-ai/mloda/blob/main/CONTRIBUTING.md#local-development-setup) (`uv sync --all-extras`, then `source .venv/bin/activate`).

#### Building the docs locally

After completing the setup in CONTRIBUTING.md, you can preview docs changes with:

```bash
mkdocs serve --config-file docs/mkdocs.yml
```

#### Where to start

- Plugin contributions: see the [mloda-registry guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides) and the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template).
- Core framework contributions: browse issues labeled [`good first issue`](https://github.com/mloda-ai/mloda/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [`help wanted`](https://github.com/mloda-ai/mloda/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
- Bugs and feature requests: open an [issue](https://github.com/mloda-ai/mloda/issues/).
