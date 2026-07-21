# Technical Context

## Language and Runtime

- Python `>=3.10,<3.15`; CI matrixes 3.10, 3.11, 3.12, 3.13, 3.14.
- Modern type hints only (`list[str]`, `X | None`), enforced by ruff `UP006` / `UP007`.
- The core declares no runtime dependencies; the PythonDict backend also needs none, while every other backend is an optional extra.

## Optional Extras (backends)

`pandas`, `pyarrow`, `polars`, `duckdb`, `sqlite`, `iceberg`, `spark`, `sklearn` (scikit-learn + joblib), `text_cleaning` (nltk), `otel` (OpenTelemetry), `docs` (mkdocs, mkdocs-jupyter, marimo, mermaid-py).

## Tooling

- **uv**: dependency management; the committed `uv.lock` is installed by the gate so each commit resolves reproducibly.
- **tox**: the single gate. It runs, in order: `pytest -n 8 --timeout=10`, `ruff format --check` (line length 120), `ruff check`, a `pip-licenses` allowlist check, `mypy --strict --ignore-missing-imports`, and `bandit`. All must pass.
- Tests assert `EXPECTED_SKIP_COUNT=170`; a newly skipped test must update the count or be unskipped.
- `pip-audit` runs in a dedicated CVE-scan env.

## Development Setup

```bash
uv sync --all-extras
source .venv/bin/activate
tox
```

## Constraints

- Tests must be parallel-safe (pytest-xdist) and finish under the 10-second timeout.
- Supply chain: `[tool.uv] exclude-newer = "7 days"` defers new releases; dependency licenses must satisfy the `tox.ini` allowlist.
- `attribution/ATTRIBUTION.md` is regenerated on every `tox` run and committed alongside dependency changes.
- The package ships `py.typed` (typed distribution).
