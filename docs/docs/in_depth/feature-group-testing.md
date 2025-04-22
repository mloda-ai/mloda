# Feature Group Testing


Most feature groups were generated automatically by using an agent. It is good practise to give the agents examples of other tests. Some agents tend to skip tests. Please be aware of that.

This guide outlines key aspects to test in feature groups and provides brief examples.

## What to Test

### 1. Feature Name Pattern Matching

Test that your feature group correctly identifies feature names it should handle.

**Example:**
```python
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_core.abstract_plugins.components.options import Options

# Test valid and invalid feature names
assert ClusteringFeatureGroup.match_feature_group_criteria("cluster_kmeans_5__customer_behavior", Options())
assert not ClusteringFeatureGroup.match_feature_group_criteria("invalid_name", Options())
```

### 2. Input Feature Extraction

Test that your feature group correctly extracts source features from feature names.

**Example:**
``` python
# Test extracting source features from a feature name
input_features = feature_group.input_features(Options(), FeatureName("sum_aggr__sales"))
assert Feature("sales") in input_features
```

### 3. Calculation Logic

Test that your feature group correctly transforms input data into output features.

**Example:**
``` python
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClusteringFeatureGroup

# Test calculation with sample data
feature_set = FeatureSet()
feature_set.add(Feature("cluster_kmeans_2__feature1,feature2"))
result = PandasClusteringFeatureGroup.calculate_feature(sample_data, feature_set)
assert "cluster_kmeans_2__feature1,feature2" in result.columns
```

### 4. Configuration-Based Feature Creation

Test that your feature group correctly parses features from configuration options.

**Example:**
``` python
# Test creating features from options
options = Options({
    "aggregation_type": "sum",
    "mloda_source_feature": "sales"
})
feature_name = parser_config.parse_from_options(options)
assert feature_name == "sum_aggr__sales"
```

### 5. Integration with mloda API

Test that your feature group works correctly with the mloda API.

**Example:**
``` python
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe

features = ["source_feature", "my_operation__source_feature"]
result = mlodaAPI.run_all(features, compute_frameworks={PandasDataframe})
assert "my_operation__source_feature" in result[0].columns
```

## Test Organization

Organize tests into three categories:

1. **Base Class Tests**: Test feature name parsing, input feature extraction, and configuration
2. **Framework Implementation Tests**: Test calculation logic for specific compute frameworks
3. **Integration Tests**: Test with the mloda API and other components
