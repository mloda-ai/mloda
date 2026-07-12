# Contributing to mloda

We welcome contributions from the community. Whether you are fixing bugs, adding features, or developing plugins, your input is invaluable.

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Show Support

The simplest way to support mloda is to star the repository on GitHub. It costs nothing, takes a second, and helps others discover the project.

## Getting Started

### Prerequisites

- Python 3.10 or higher (tested on 3.10, 3.11, 3.12, 3.13, 3.14)
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

Run the full suite (linting, formatting, type checking, security, and tests). Always run tox before submitting a pull request:

```bash
tox
```

For quick iteration during development, you can run only the tests. Note that this skips linting, type checking, and security checks, so it is not a substitute for tox:

```bash
pytest -n auto --timeout=10
```

When multiple processes run `tox` on the same host (e.g., parallel agents), their `pytest` step serializes on `$TOX_LOCK_PATH` (default `/tmp/mloda-tox.lock`) when `flock` is available. The file is created world-writable so any local user can share the lock; if you prefer a per-user lock, point `TOX_LOCK_PATH` at a path under `$HOME` (literal path, not `~`). Set `TOX_LOCK_PATH=` (empty) to disable. On hosts without `flock` (macOS: `brew install flock`; Windows: use WSL) or where the lock file is not writable (read-only `/tmp`, foreign ownership), the lock is skipped automatically and `tox` runs unwrapped.

## Ways to Contribute

### Plugin Development

The easiest way to extend mloda is by creating plugins. The [mloda-registry](https://github.com/mloda-ai/mloda-registry) hosts 40+ guides under [`docs/guides/`](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides), organized as a step-by-step journey:

1. **Using plugins**: Learn to use and discover existing plugins.
2. **Creating plugins**: Build a plugin in your project, then package it.
3. **Sharing plugins**: Distribute via private repos, publish to the community registry, or contribute to official plugins.
4. **Advanced patterns**: Deep dives into feature groups, compute frameworks, and extenders, with dedicated pattern catalogs for each.

Start with the [Plugin Journey overview](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/index.md) to find the right guide for your level.

To scaffold a new standalone plugin package with pre-configured CI/CD, use the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template).

### Core Development

For contributions to the mloda core framework, see our [Documentation](https://mloda-ai.github.io/mloda/).

### Report Issues

Found a bug or have a feature request? [Open an issue](https://github.com/mloda-ai/mloda/issues/). The issue template will prompt you for a summary, reproduction or motivation, optional code pointers, and an optional definition of done.

## Pull Request Workflow

1. Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/mloda.git
cd mloda
```

2. Create a feature branch from `main`:

```bash
git checkout -b fix/short-description
```

3. Make your changes and ensure `tox` passes locally.
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/) format.
5. Push your branch to your fork and open a pull request targeting `main`.
6. CI runs the full tox suite on Python 3.10, 3.11, 3.12, 3.13, and 3.14. All checks must pass before merge.

### Renaming or Removing Public API

Open issues cite public symbols as code pointers. When a PR renames or removes one (a `FeatureGroup` hook, a `ComputeFramework` method, an extender type, a `DefaultOptionKeys` member), sweep the backlog for the old name in the same PR:

```bash
gh issue list --repo mloda-ai/mloda --state open --search "<old name> in:body"
```

Update each hit to the new name, or close the issue if the change made it moot. A stale pointer costs the next contributor discovery time before any real work starts.

Confirm the symbol is really gone by grepping the public surface, not the whole tree. A retired symbol usually lives on in tests that assert its removal, so a repo-wide grep still reports hits and hides the rename:

```bash
git grep -w "<old name>" -- mloda/ mloda_plugins/   # no output: gone from the public surface
```

## License

By contributing, you agree that your contributions will be licensed under the [Apache License, Version 2.0](https://github.com/mloda-ai/mloda/blob/main/LICENSE).
