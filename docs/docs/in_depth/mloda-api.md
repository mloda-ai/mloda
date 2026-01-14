## mlodaAPI

The **mlodaAPI** class serves as the primary interface for interacting with the core system, streamlining the setup and execution of computational workflows

#### Major components (and steps)

1. **Configuration**

    Initialize the mloda with features, compute frameworks, and other settings to configure your environment.

2.  **Engine Setup**

    Create an execution plan by setting up the engine, which orchestrates the computation based on the defined features and configurations.

3.  **Runner Setup**

    Prepare the runner that will execute the plan, ensuring that all components are ready for computation.

4.  **Run Engine Computation**

    Apply the runner to execute the computations based on the prepared execution plan, managing the lifecycle of the computational process.

This means, depending on your needs, you can run them all at once (**batch run**) or split them up, e.g. for realtime needs (**inference**).

#### Configuration for mlodaAPI

-   **requested_features**: Specify the features to process (as names, Feature objects, or a Features container).
-   **compute_frameworks** (optional): Limit the compute frameworks using framework types or names.
-   **links** (optional): Define dataset merging links with Link objects.
-   **data_access_collection** (optional): Provide data sources for feature identification.

#### Runner & Execution Configuration

-   **function_extender** (optional): Add function extenders to customize computations.
-   parallelization_modes (optional): Choose between sync, threading, or multiprocessing modes. (Default: sync)
-   flight_server (optional): Specify a flight server for multiprocessing only.

#### Plugin Discovery

To discover available plugins (feature groups, compute frameworks, extenders), use the functions `get_feature_group_docs()`, `get_compute_framework_docs()`, and `get_extender_docs()` from `mloda.steward`:

```python
from mloda.steward import get_feature_group_docs, get_compute_framework_docs, get_extender_docs
```

