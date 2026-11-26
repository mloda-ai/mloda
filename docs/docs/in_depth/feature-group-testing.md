# Feature Group Testing

Start with the conformance kit, which checks the contract every feature group shares, then add unit tests for what only your group does.

## The conformance kit

The kit is `mloda.testing`, from the `mloda-testing` package. It needs pytest; adapters that build frames through Arrow need the `mloda-testing[pyarrow]` extra. Each `check_*` function returns a `CheckResult` (`failures`, `notes`, `passed`); its `assert_*` twin fails the running test instead.

### Shape check

`check_feature_group_shape(cls, base_options=None, *, input_data=None, adapter=None, require_run=False, requests=None)` from `mloda.testing.feature_group_shape` probes the match contract: the class name and declared names match, a foreign name and an undeclared option do not, declared inputs are a set of `Feature`, dropping a required option records a `required_option_absent` rejection naming it, a strict option rejects an unknown value, and `version()`, `compute_framework_rule()` and secret-shaped options are declared soundly. Given `input_data` (a `TypedTestData` from `mloda.testing.typed_data`), it also runs the names on the `adapter`'s compute framework (Python dict by default) and checks outputs, types, zero rows, error values and determinism. `assert_feature_group_shape` defaults to `require_run=True`, so a run that cannot happen fails instead of leaving a note.

Each failure starts with its sub-check: `names:`, `foreign:`, `inputs:`, `required:`, `strict:`, `secrets:`, `declaration:`, `run:`, `outputs:`, `type:`, `zero_rows:`, `error_values:`, `determinism:`.

```py
from mloda.testing.feature_group_shape import assert_feature_group_shape
from mloda.testing.framework_adapter import PandasAdapter
from mloda.testing.typed_data import TypedTestData
from mloda.user import DataType
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation  # noqa: F401


def test_aggregation_shape() -> None:
    assert_feature_group_shape(
        AggregatedFeatureGroup,
        requests=["x__sum_aggr", "x__count_aggr", "x__median_aggr"],
        input_data=TypedTestData({"x": (DataType.DOUBLE, [1.0, 2.0, 3.0, 4.0])}),
        adapter=PandasAdapter(),
    )
```

An operation group's serving Implementation must be imported (or plugins loaded) before the check runs it.

- **`requests=`** probes and runs these names instead of the class name and declared names; a name-path group needs it. Pass a sequence of names, not a string, and check names that bind different option keys in separate calls.
- **Configuration path**: with `in_features` in `base_options`, a mixin group without `NAME_ONLY_MATCH` matches any name, so the foreign probe drops `in_features` and notes it; a group that still matches without it fails `foreign:`. A dropped required key records no rejection on this path, so `required:` passes when resolution names the key, and when the request no longer selects the group the silent decline is only noted.
- **`secrets:`** is described in [Declaring a credential handle](named-data-access-handles.md#declaring-a-credential-handle); `check_feature_group_secrets` runs that sub-check alone; `check_feature_group_result_types` runs only the run and type checks and reports no stale records.
- **Type notes**: a `type` [divergence record](divergence-records.md) of the group or its serving Implementation turns an exact mismatch, a lost type, a value misfit or a zero-row type change into a `type:` note naming the record. Only a `gap` record excuses an unresolved type (exact readback, but the declared type cannot be resolved), which otherwise fails. For example, a Pandas `object` column declared DATE, STRING, LIST, BINARY or DECIMAL is unresolved. A `type` record on the serving Implementation whose columns all match exactly on a typed framework fails as stale; on the group it is a note.
- **Zero-row input types**: an input column that changes type on zero rows fails `zero_rows:` unless the adapter frames it differently by row count (a pandas int or bool column with nulls), the with-rows type is the framework's own Arrow framing (float64 or object), and the zero-row type is the one the adapter handed. An output whose zero-row type only follows an int input with nulls, framed as float64 with rows, is excused too.
- **Pass-through inputs**: the `outputs:` check accepts an input column the adapter reads back differently from the source values, as long as the result matches that readback.

### Other checks

- Subclass scope (`mloda.testing.subclass_scope`), operation group shape (`mloda.testing.operation_group_shape`) and the source group suites (`mloda.testing.source`): see [A loaded subclass replaces its parent](troubleshooting/feature-group-resolution-errors.md#a-loaded-subclass-replaces-its-parent).
- Time zone computation rule (cases and check in `mloda.testing.time_rule`, suite in `mloda.testing.compute_framework.time_window_suite`): see [Testing against the rule](timezone-computation.md#testing-against-the-rule).
- ComputeFramework suites and framework adapters: see [Testing](compute-framework-integration.md#testing).

## Unit tests next to the kit

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

A bare direct call records nothing; wrap one call in `collect_match_rejections()` to read the reasons it recorded (see section 4), and full engine-shaped assertions go through `IdentifyFeatureGroupClass.evaluate`.

### 2. Input Feature Extraction

Test that your feature group correctly extracts source features from feature names.

**Example:**
```python
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

# Test extracting source features from a feature name
input_features = AggregatedFeatureGroup().input_features(Options(), FeatureName("sales__sum_aggr"))
assert Feature("sales") in input_features
```

`input_features` is an instance method, so instantiate the feature group before calling it. For an abstract base, call `declared_input_features(options, feature_name)` on the abstract base class itself to get the declared inputs.

### 3. Calculation Logic

Test that your feature group correctly transforms input data into output features.

**Example:**
```python
import pandas as pd
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation

# Test calculation with sample data
sample_data = pd.DataFrame({"sales": [10.0, 20.0, 30.0]})
feature_set = FeatureSet()
feature_set.add(Feature("sales__sum_aggr"))
result = PandasAggregation.calculate(AggregatedFeatureGroup, sample_data, feature_set, None)
assert "sales__sum_aggr" in result.columns
```

### 4. Configuration-Based Feature Creation

Test that your feature group correctly parses features from configuration options.

**Example:**
```python
from mloda.provider import collect_match_rejections
from mloda.user import Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

# Test that the options a request carries are enough for the feature group to match
options = Options(context={
    "aggregation_type": "sum",
    "in_features": "sales"
})
assert AggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", options)

# Assert the negative too, or the test passes on a name the group never claims; collect the recorded reason
with collect_match_rejections() as reasons:
    assert not AggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", Options(context={
        "aggregation_type": "not_an_aggregation"
    }))
assert "not_an_aggregation" in reasons["AggregatedFeatureGroup"].reason
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
    compute_frameworks=[PandasDataFrame],
    api_data={"SalesData": {"sales": [10.0, 20.0, 30.0]}},
)
assert "sales__sum_aggr" in result[0].columns
```

### 6. Testing with Mock Input Data

When testing a FeatureGroup that depends on another FeatureGroup, you can inject mock data by combining `disabled_feature_groups` with `api_data`:

**Example:**
```python
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda


class HandGenerator(DataCreatorFeatureGroup):
    """The expensive dependency under test, replaced by mock data below."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"hand"}

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
