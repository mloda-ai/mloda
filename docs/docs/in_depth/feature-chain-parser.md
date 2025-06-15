# Feature Chain Parser

## Overview

The Feature Chain Parser system enables feature groups to work with chained feature names and configuration-based feature creation. This document explains the key concepts and how to use them in your custom feature groups.

## Key Concepts

### Feature Chaining

Feature chaining allows feature groups to be composed, where the output of one feature group becomes the input to another. This is reflected in the feature name using a double underscore pattern:

```
{operation}__{source_feature}
```

For example:
- `sum_aggr__sales` - Simple feature
- `max_aggr__sum_7_day_window__mean_imputed__price` - Chained feature

### FeatureChainParser

The `FeatureChainParser` class provides utilities for working with feature names:

- **Extracting source features**: Get the input feature from a feature name
- **Validating feature names**: Check if a name follows the expected pattern
- **Detecting chained features**: Determine if a feature is part of a chain

### Configuration-Based Feature Creation

The `FeatureChainParserConfiguration` allows features to be created from configuration options rather than explicit names:

```python
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options


# Instead of explicitly naming features:
feature = Feature("sum_aggr__sales")

# Use configuration-based creation:
feature = Feature(
    "PlaceHolder",  # Will be replaced
    Options({
        "aggregation_type": "sum",
        "mloda_source_feature": "sales"
    })
)
```

## Implementation in Feature Groups

### 1. Define a Prefix Pattern

```python
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

class MyFeatureGroup(AbstractFeatureGroup):
    # Define the prefix pattern for feature name parsing
    PREFIX_PATTERN = r"^my_operation__"
```

### 2. Extract Source Features

```python
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser

def input_features(self, options, feature_name):
    # Extract source features using the FeatureChainParser
    source_feature = FeatureChainParser.extract_source_feature(
        feature_name.name, self.PREFIX_PATTERN)
    return {Feature(source_feature)}
```

### 3. Enable Configuration-Based Creation

```python
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    create_configurable_parser
)


@classmethod
def configurable_feature_chain_parser(cls):
    return create_configurable_parser(
        parse_keys=[
            "operation_param",
            "mloda_source_feature"
        ],
        feature_name_template="my_operation_{operation_param}__{mloda_source_feature}",
        validation_rules={
            "operation_param": lambda x: x in ["valid", "values"]
        }
    )
```

## Multiple Result Columns with ~ Pattern

Some feature groups produce multiple result columns from a single input feature. The `~` pattern allows accessing individual columns:

```python
# OneHot encoding creates multiple columns
base_feature = "onehot_encoded__category"  # Creates all columns
specific_column = "onehot_encoded__category~0"  # Access first column
another_column = "onehot_encoded__category~1"  # Access second column
```

**Implementation Note**: Feature groups handle this pattern in their `input_features()` method to extract the base feature name, and in `calculate_feature()` to create the appropriately named result columns.

## Benefits

- **Consistent Naming**: Enforces naming conventions across feature groups
- **Composability**: Enables building complex features through chaining
- **Configuration-Based Creation**: Simplifies feature creation in client code
- **Validation**: Ensures feature names follow expected patterns
- **Multi-Column Support**: Handle transformations that produce multiple result columns
