# Contributing to mloda

We welcome contributions from the community. Whether you are fixing bugs, adding features, or developing plugins, your input is invaluable.

## Getting Started

### Prerequisites

- Python 3.10 or higher (tested on 3.10, 3.11, 3.12, 3.13)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [tox](https://tox.wiki/) as the test runner (installed via uv)

### Local Development Setup

1. Clone the repository:

```bash
git clone https://github.com/mloda-ai/mloda.git
cd mloda
```

2. Install dependencies (including all extras needed for development):

```bash
uv sync --all-extras
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Verify your setup by running the full test suite:

```bash
tox
```

## Code Style

All code must pass the automated checks enforced by tox. The toolchain includes:

- **ruff format** for code formatting (line length: 120 characters)
- **ruff check** for linting
- **mypy --strict** for static type checking
- **bandit** for security scanning

### Conventions

- No code in `__init__.py` files.
- Avoid `try/except` blocks unless absolutely necessary.
- Keep documentation to the necessary minimum.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`fix:`, `feat:`, `chore:`, etc.).

### Running Checks Locally

Run the full suite (linting, formatting, type checking, security, and tests):

```bash
tox
```

Run only the tests:

```bash
pytest -n auto --timeout=10
```

## Pull Request Workflow

1. Create a feature branch from `main`:

```bash
git checkout -b fix/short-description
```

2. Make your changes and ensure `tox` passes locally.
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) format.
4. Push your branch and open a pull request targeting `main`.
5. CI runs the full tox suite on Python 3.10, 3.11, 3.12, and 3.13. All checks must pass before merge.

## Ways to Contribute

### Core Development

For contributions to the mloda core framework, see our [Documentation](https://mloda-ai.github.io/mloda/).

### Plugin Development

The easiest way to extend mloda is by creating plugins. The [mloda-registry](https://github.com/mloda-ai/mloda-registry) contains 40+ guides organized as a step-by-step journey:

1. **Using plugins**: Learn to use and discover existing plugins.
2. **Creating plugins**: Build a plugin in your project, then package it.
3. **Sharing plugins**: Distribute via private repos, publish to the community registry, or contribute to official plugins.
4. **Advanced patterns**: Deep dives into feature groups, compute frameworks, and extenders, with dedicated pattern catalogs for each.

Start with the [Plugin Journey overview](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/index.md) to find the right guide for your level.

To scaffold a new standalone plugin package with pre-configured CI/CD, use the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template).

### Report Issues

Found a bug or have a feature request? [Open an issue](https://github.com/mloda-ai/mloda/issues/).

## License

By contributing, you agree that your contributions will be licensed under the [Apache License, Version 2.0](https://github.com/mloda-ai/mloda/blob/main/LICENSE).
