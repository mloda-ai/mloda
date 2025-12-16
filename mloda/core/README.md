# mloda Core

## Overview
The mloda Core is the central engine of the mloda framework, orchestrating the interactions between plugins and managing the data transformation pipeline. It handles dependencies between features and computations by coordinating linking, joining, filtering, and ordering operations to ensure optimized data processing.

## Architecture
The core architecture consists of several key components that work together to provide a flexible and resilient framework for data and feature engineering:

- **Core Engine**: Orchestrates the execution of feature transformations and manages the data flow between different components.
- **API**: Provides interfaces for interacting with the mloda framework.
- **Runtime**: Handles the execution of transformations in different environments.
- **Filter**: Manages data filtering operations.
- **Prepare**: Handles the preparation and planning of execution steps.

## Directory Structure
```
mloda/core/
├── abstract_plugins/    # Abstract base classes for plugins
├── api/                 # API interfaces
├── core/                # Core engine implementation
├── filter/              # Filtering mechanisms
├── prepare/             # Preparation and planning logic
└── runtime/             # Execution runtime
```

## Key Components

### Abstract Plugins
Defines the interfaces and abstract base classes for the plugin system, including Feature Groups, Compute Frameworks, and Function Extenders.

### API
Provides interfaces for interacting with the mloda framework, including request handling and response formatting.

### Core
Implements the core engine functionality, including dependency resolution, execution planning, and orchestration of plugins.

### Filter
Manages data filtering operations, allowing for efficient querying and processing of complex features.

### Prepare
Handles the preparation and planning of execution steps, including graph building, link resolution, and compute framework selection.

### Runtime
Manages the execution of transformations in different environments, including local execution and distributed processing.

## How It Works
The mloda Core works by:

1. Receiving feature requests through the API
2. Resolving feature dependencies using the plugin system
3. Building an execution graph to optimize processing
4. Selecting appropriate compute frameworks for each step
5. Executing the transformations in the runtime environment
6. Returning the processed features to the requester

This approach focuses on defining transformations rather than static states, facilitating smooth transitions between development phases and reducing redundant work.

## Development
When extending or modifying the core functionality, consider the following guidelines:

- Maintain backward compatibility with existing plugins
- Follow the established design patterns and architecture
- Write comprehensive tests for new functionality
- Document changes and additions thoroughly
- Consider performance implications, especially for large-scale data processing
