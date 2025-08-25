# Progress

## Status Overview

```mermaid
pie title Implementation Status
    "Completed" : 85
    "In Progress" : 10
    "Planned" : 5
```

## Working Features

```mermaid
mindmap
  root((mloda))
    Feature Groups
      Core
        AggregatedFeatureGroup
        TimeWindowFeatureGroup
        MissingValueFeatureGroup
      Analytics
        ClusteringFeatureGroup
        DimensionalityReduction
        ForecastingFeatureGroup
        NodeCentralityFeatureGroup
      Processing
        TextCleaningFeatureGroup
        GeoDistanceFeatureGroup
        SklearnPipelineFeatureGroup
    Compute Frameworks
      PandasDataframe
      PyarrowTable
      PythonDict
      PolarsDataframe
      DuckDBFramework
      IcebergFramework
      SparkFramework
    Core Systems
      Options Refactoring
      PROPERTY_MAPPING
      Feature Chaining
      Artifact Storage
```

## Recent Achievements
- ✅ Options object refactoring with group/context separation
- ✅ All Feature Groups modernized to PROPERTY_MAPPING
- ✅ Sklearn Pipeline with artifact storage
- ✅ Complete compute framework implementations

## Known Issues

```mermaid
flowchart TD
    Issue1[FeatureSet Creation] --> |Unintuitive| Fix1[feature_set.add syntax]
    Issue2[Options.data Dictionary] --> |Unhashable| Fix2[JSON serialization workaround]
    
    style Issue1 fill:#fbb,stroke:#333
    style Issue2 fill:#fbb,stroke:#333
```

- **FeatureSet creation**: Requires `FeatureSet()` then `.add(feature)`
- **Options.data dictionaries**: Can't hash dict values, use JSON serialization

