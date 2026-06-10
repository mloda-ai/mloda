# Policy conformance is mandatory

Every new FeatureGroup-level policy flag (anything in the spirit of
`allow_empty_result()`: a class method on `FeatureGroup` that changes how the pipeline
treats a result) MUST ship a `run_all`-driven conformance suite.

## What "conformance suite" means

Build it on `PolicyRunAllTestBase` (see `policy_run_all_test_base.py`). The suite must:

- exercise the flag END-TO-END through `mloda.run_all`, not in isolation;
- cover ALL built-in compute frameworks (PyArrow, Pandas, Polars, DuckDB, SQLite,
  PythonDict, Spark, Iceberg) via per-framework subclasses;
- cover at least SYNC plus one worker-based parallelization mode (e.g. THREADING);
- express each case as a `PolicySuccess` or `PolicyRaises` expectation.

## Template

- `policy_run_all_test_base.py` is the generic base.
- `empty_result_run_all_test_base.py` is the canonical first consumer; copy its shape.

## Why this bar exists

`allow_empty_result()` had THREE enforcement surfaces. The third one, in
`DataLifecycleManager.get_result_data`, shipped broken. Unit tests against the first two
surfaces all passed. The defect was only caught because the full pipeline runs for every
framework and every parallelization mode through `run_all`. A policy flag that is not
driven through the whole pipeline, on every framework, is not actually verified.
