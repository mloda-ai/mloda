# Feature Group Testing

This guide outlines key aspects to test in feature groups and provides brief examples.

## What to Test

### 1. Feature Name Pattern Matching

Test that your feature group correctly identifies feature names it should handle.

**Example:**
```python
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda.user import Options

# Test valid and invalid feature names
assert ClusteringFeatureGroup.match_feature_group_criteria("customer_behavior__cluster_kmeans_5", Options())
assert not ClusteringFeatureGroup.match_feature_group_criteria("invalid_name", Options())
```

The reader veto gate reads the engine's per-candidate rejection window, so a direct `match_feature_group_criteria` call without an active window does not exercise it; engine-shaped assertions go through `IdentifyFeatureGroupClass.evaluate`.

### 2. Input Feature Extraction

Test that your feature group correctly extracts source features from feature names.

**Example:**
```python
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

# Test extracting source features from a feature name
input_features = PandasAggregatedFeatureGroup().input_features(Options(), FeatureName("sales__sum_aggr"))
assert Feature("sales") in input_features
```

`input_features` is an instance method, so instantiate the feature group before calling it.

### 3. Calculation Logic

Test that your feature group correctly transforms input data into output features.

**Example:**
```python
import pandas as pd
from mloda.user import Feature
from mloda.provider import FeatureSet

# Test calculation with sample data
sample_data = pd.DataFrame({"sales": [10.0, 20.0, 30.0]})
feature_set = FeatureSet()
feature_set.add(Feature("sales__sum_aggr"))
result = PandasAggregatedFeatureGroup.calculate_feature(sample_data, feature_set)
assert "sales__sum_aggr" in result.columns
```

### 4. Configuration-Based Feature Creation

Test that your feature group correctly parses features from configuration options.

**Example:**
```python
# Test that the options a request carries are enough for the feature group to match
options = Options(context={
    "aggregation_type": "sum",
    "in_features": "sales"
})
assert PandasAggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", options)

# Assert the negative too, or the test passes on a name the group never claims
assert not PandasAggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", Options(context={
    "aggregation_type": "not_an_aggregation"
}))
```

### 5. Integration with mloda API

Test that your feature group works correctly with the mloda API.

**Example:**
```python
from mloda.user import mloda
from mloda.user.pandas import PandasDataFrame

features = ["sales__sum_aggr"]
result = mloda.run_all(
    features,
    compute_frameworks={PandasDataFrame},
    api_data={"SalesData": {"sales": [10.0, 20.0, 30.0]}},
)
assert "sales__sum_aggr" in result[0].columns
```

### 6. Testing with Mock Input Data

When testing a FeatureGroup that depends on another FeatureGroup, you can inject mock data by combining `disabled_feature_groups` with `api_data`:

**Example:**
```python
from typing import Any

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda


class HandGenerator(FeatureGroup):
    """The expensive dependency under test, replaced by mock data below."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"hand"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"hand": ["72o"]}


class HandScore(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
    ) -> bool:
        return str(feature_name) == "hand_score"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("hand")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["hand_score"] = [len(hand) for hand in data["hand"]]
        return data


# Disable the real dependency FeatureGroup
collector = PluginCollector.disabled_feature_groups({HandGenerator})

# Inject mock data and run your derived feature
results = mloda.run_all(
    features=["hand_score"],  # Your derived feature
    compute_frameworks=["PandasDataFrame"],
    api_data={"hand": {"hand": ["AA", "KK", "QQ"]}},  # Mock the dependency
    plugin_collector=collector,
)

assert "hand_score" in results[0].columns
```

This pattern is useful when:
- Testing derived features without running expensive upstream computations
- Providing controlled test data for reproducible tests
- Isolating the feature under test from its dependencies

## Test Organization

Organize tests into three categories:

1. **Base Class Tests**: Test feature name parsing, input feature extraction, and configuration
2. **Framework Implementation Tests**: Test calculation logic for specific compute frameworks
3. **Integration Tests**: Test with the mloda API and other components
