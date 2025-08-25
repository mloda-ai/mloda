# Compute Frameworks

## Overview

Compute Frameworks define the technology stack for executing feature transformations, decoupling feature definitions from specific computation technologies.

```mermaid
graph TD
    subgraph Frameworks
        PD[PandasDataframe]
        PA[PyarrowTable]
        PY[PythonDict]
        PL[PolarsDataframe]
        DB[DuckDBFramework]
        IC[IcebergFramework]
        SP[SparkFramework]
    end
    
    subgraph Components
        ME[MergeEngine]
        FE[FilterEngine]
        TR[Transformers]
    end
    
    Frameworks --> Components
    Components --> Exec[Execution]
    
    style Frameworks fill:#bbf,stroke:#333,stroke-width:2px
```

## Available Frameworks

### Core Frameworks
- **PandasDataframe**: Rich data manipulation, good for development
- **PyarrowTable**: Memory-efficient columnar format, production-ready
- **PythonDict**: Dependency-free, List[Dict] structure

### Advanced Frameworks
- **PolarsDataframe**: High-performance Rust-based, lazy evaluation
- **DuckDBFramework**: SQL interface, OLAP workloads
- **IcebergFramework**: Data lake management, schema evolution
- **SparkFramework**: Distributed processing, big data

## Framework Selection

```mermaid
flowchart LR
    F1[Feature Definition] --> |Specify| CF[Compute Framework]
    F2[Feature Group Rules] --> |Limit| CF
    F3[API Request] --> |Override| CF
    
    CF --> Auto[Automatic Selection]
```

## Framework Transformers

Enable seamless conversion between data representations:

```python
# Automatic transformation when needed
PandasDataframe <--> PyarrowTable
PyarrowTable <--> PolarsDataframe
PyarrowTable <--> SparkDataframe
```

## Integration Pattern

```python
@classmethod
def compute_framework_rule(cls):
    return {PandasDataframe}  # Specific framework
    # Or return True for all frameworks
