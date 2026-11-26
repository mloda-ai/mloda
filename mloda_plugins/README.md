# mloda Plugins

## Overview
The plugin system is a fundamental part of the mloda framework, providing extensibility and flexibility. Plugins allow for customization of feature definitions, computation technologies, and metadata extraction processes, enabling the framework to adapt to various use cases and environments.

## Plugin Types

- **Feature Groups** define feature dependencies and calculations, with automatic dependency resolution.
- **Compute Frameworks** define the technology stack (e.g. Pandas, PyArrow) that executes them, keeping feature definitions independent of it.
- **Function Extenders** add metadata extraction, governance and traceability without modifying core behavior.
- **Implementations** serve one (format or operation Feature Group, Compute Framework) pair with their own `calculate`, for example loading a CSV file into a PyArrow table.

## Directory Structure
```text
mloda_plugins/
├── compute_framework/   # Compute framework implementations
├── feature_group/       # Feature group implementations
└── function_extender/   # Extension point for custom Extenders (bundled: TimingExtender)
```
