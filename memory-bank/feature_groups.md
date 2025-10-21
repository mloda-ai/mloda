# Feature Groups

## Overview

Feature Groups are core components that define feature dependencies and calculations using a modern configuration-based architecture.

```mermaid
graph TD
    AFG[AbstractFeatureGroup] --> |Base Class| Implementations
    
    subgraph Implementations
        AGG[AggregatedFeatureGroup]
        TW[TimeWindowFeatureGroup]
        MV[MissingValueFeatureGroup]
        CL[ClusteringFeatureGroup]
        SK[SklearnPipelineFeatureGroup]
    end
    
    AGG --> |Framework Specific| PA[PandasAggregated]
    AGG --> |Framework Specific| PY[PyArrowAggregated]
    
    style AFG fill:#f9f,stroke:#333,stroke-width:4px
```

## Implemented Feature Groups

### Core Feature Groups
- **AggregatedFeatureGroup**: Sum, avg, min, max aggregations
- **TimeWindowFeatureGroup**: Time-based window operations (e.g., 7-day averages)
- **MissingValueFeatureGroup**: Mean, median, mode, constant, ffill, bfill imputation

### Analytics Feature Groups
- **ClusteringFeatureGroup**: K-means, DBSCAN, hierarchical clustering
- **DimensionalityReductionFeatureGroup**: PCA, t-SNE, ICA, LDA, Isomap
- **ForecastingFeatureGroup**: Linear, random forest, SVR time series forecasting
- **NodeCentralityFeatureGroup**: Degree, betweenness, closeness, pagerank centrality

### Processing Feature Groups
- **TextCleaningFeatureGroup**: Text normalization, stopword removal, punctuation cleaning
- **GeoDistanceFeatureGroup**: Haversine, euclidean, manhattan distance calculations
- **SklearnPipelineFeatureGroup**: Sklearn transformers and pipelines

## Key Architecture

### PROPERTY_MAPPING Configuration

```mermaid
flowchart LR
    PM[PROPERTY_MAPPING] --> GP[Group Parameters]
    PM --> CP[Context Parameters]
    
    GP --> |Affect| Split[Feature Group Splitting]
    CP --> |Don't Affect| Meta[Metadata Only]
    
    style PM fill:#bbf,stroke:#333,stroke-width:2px
    style Split fill:#fbb,stroke:#333,stroke-width:2px
    style Meta fill:#bfb,stroke:#333,stroke-width:2px
```

**Context Parameters** (Don't affect resolution):
- Algorithm parameters (aggregation_type, window_function)
- Source features (mloda_source_features)
- Configuration settings

**Group Parameters** (Affect resolution):
- Data source isolation
- Environment parameters
- Version boundaries

### Core Methods

```python
class AbstractFeatureGroup:
    # Required Implementation
    def calculate_feature(features: FeatureCollection) -> DataCreator
    
    # Meta Methods
    def description() -> str
    def version() -> str
    def match_feature_group_criteria(feature_name, options) -> bool
    
    # Configuration
    def input_features(options, feature_name) -> Set[Feature]
    def compute_framework_rule() -> Set[ComputeFramework]
```

## Feature Patterns

### 1. Aggregated Pattern

**Convention**: `{aggregation_type}_aggr__{source_feature}`

```mermaid
graph LR
    sales[sales] --> |sum| sum_aggr[sum_aggr__sales]
    price[price] --> |avg| avg_aggr[avg_aggr__price]
```

```python
PROPERTY_MAPPING = {
    AGGREGATION_TYPE: {
        **AGGREGATION_TYPES_DICT,
        DefaultOptionKeys.mloda_context: True,  # Context parameter
    },
    DefaultOptionKeys.mloda_source_features: {
        DefaultOptionKeys.mloda_context: True,
    },
}
```

### 2. Time Window Pattern

**Convention**: `{function}_{size}_{unit}_window__{source_feature}`

```mermaid
graph LR
    temp[temperature] --> |7-day avg| avg_7[avg_7_day_window__temperature]
    humid[humidity] --> |5-day max| max_5[max_5_day_window__humidity]
```

### 3. Feature Chaining

Complex features through composition:

```mermaid
graph LR
    price --> |impute| imp[mean_imputed__price]
    imp --> |window| win[sum_7_day_window__mean_imputed__price]
    win --> |aggregate| final[max_aggr__sum_7_day_window__mean_imputed__price]
    
    style price fill:#ffd,stroke:#333
    style final fill:#dfd,stroke:#333,stroke-width:2px
```

## Implementation Guide

### Dual Approach Support

```python
# String-based (Legacy)
feature = Feature("sum_aggr__sales")

# Configuration-based (Modern)
feature = Feature(
    "placeholder",
    Options(context={
        "aggregation_type": "sum",
        DefaultOptionKeys.mloda_source_features: "sales"
    })
)
```

### Feature Group Authority

Each Feature Group defines its own parameter categorization:
