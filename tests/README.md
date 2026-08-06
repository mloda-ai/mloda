# mloda Tests

## Overview
Testing is a critical aspect of the mloda framework, ensuring reliability, correctness, and maintainability. The testing approach combines unit tests, integration tests, and documentation tests to provide comprehensive coverage of the codebase.

## Directory Structure

Directories to three levels deep; regenerate with
`find tests -type d -not -path "*__pycache__*" | sort`.

```
tests/
├── conftest.py                     # Common pytest fixtures and configuration
├── registry_isolation.py           # Helper: run a body in an isolated plugin registry
├── registry_isolation_probe.py     # Helper: subprocess probe for registry isolation
├── test_agent_docs_sync.py         # Agent-facing docs stay in sync with the code
├── test_attributions.py            # Third-party attributions are complete
├── test_ci_paths_ignore.py         # CI path filters do not skip real changes
├── test_docs_fences.py             # Fenced code blocks in docs are well-formed
├── test_docs_taught_values.py      # Values taught in docs match the code
├── test_gc_freeze_contract.py      # GC freeze contract around plugin loading
├── test_mloda_imports.py           # Public import surface of the package
├── test_packaging_metadata_guard.py  # Packaging metadata stays consistent
├── test_parallelization_modes_support.py  # Every parallelization mode is supported
├── test_project_structure.py       # Repository layout invariants
├── test_registry_isolation.py      # Plugin registry isolation between tests
├── helpers/                        # Shared stub plugins tests import (see plugin_stubs.py)
├── test_core/                      # Tests for core functionality
│   ├── test_abstract_plugins/      # Tests for abstract plugin interfaces
│   │   ├── test_components/        # Tests for component implementations
│   │   ├── test_feature_group/     # Tests for the feature group base class
│   │   ├── test_plugin_loader/     # Tests for plugin loading mechanism
│   │   └── test_plugin_registry/   # Tests for the plugin registry
│   ├── test_api/                   # Tests for API functionality
│   │   └── feature_config/         # Tests for feature configuration
│   ├── test_artifacts/             # Tests for artifact handling
│   ├── test_core/                  # Tests for core engine
│   │   └── test_step/              # Tests for execution steps
│   ├── test_filter/                # Tests for filtering mechanisms
│   ├── test_flight/                # Tests for flight server
│   ├── test_index/                 # Tests for indexing functionality
│   ├── test_integration/           # Core integration tests
│   │   └── test_core/              # Tests for core integration scenarios
│   ├── test_mask/                  # Tests for the mask engine
│   ├── test_optional_dependency/   # Behaviour when an optional dependency is absent
│   ├── test_optional_pyarrow/      # Behaviour when pyarrow is absent
│   ├── test_plugin_collector/      # Tests for plugin collection
│   ├── test_prepare/               # Tests for the planning/preparation phase
│   │   └── test_validators/        # Tests for planning validators
│   ├── test_resolution_parity/     # Parity between resolution paths
│   ├── test_runtime/               # Tests for runtime behaviour
│   ├── test_setup/                 # Tests for setup procedures
│   └── test_tooling/               # Tests for internal test tooling
├── test_documentation/             # Tests for documentation examples
├── test_examples/                  # Tests for example code
│   ├── mloda_basics/               # Tests for basic mloda examples
│   └── sklearn_integration/        # Tests for the sklearn integration example
└── test_plugins/                   # Tests for plugin implementations
    ├── api/                        # Tests for plugin-facing API
    ├── compute_framework/          # Tests for compute framework plugins
    │   ├── base_implementations/   # Tests per framework (pandas, polars, pyarrow, ...)
    │   └── test_tooling/           # Shared compute framework test tooling
    ├── extender/                   # Tests for extender plugins
    ├── feature_group/              # Tests for feature group plugins
    │   ├── experimental/           # Tests for experimental feature groups
    │   ├── input_data/             # Tests for input data handling
    │   └── src/                    # Fixture sources for feature group tests
    └── integration_plugins/        # Tests for plugin integration
        ├── chainer/                # Tests for feature chaining
        └── test_validate_features/ # Tests for feature validation
```

Test dependencies are declared in the `test` extra in `pyproject.toml`.
