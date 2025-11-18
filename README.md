# mloda: Make data, feature and context engineering shareable

[![Website](https://img.shields.io/badge/website-mloda.ai-blue.svg)](https://mloda.ai)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://mloda-ai.github.io/mloda/)
[![PyPI version](https://badge.fury.io/py/mloda.svg)](https://badge.fury.io/py/mloda)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mloda-ai/mloda/blob/main/LICENSE.TXT)
[![Tox](https://img.shields.io/badge/tested_with-tox-blue.svg)](https://tox.readthedocs.io/)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **⚠️ Early Version Notice**: mloda is in active development. Some features described below are still being implemented. We're actively seeking feedback to shape the future of the framework. [Share your thoughts!](https://github.com/mloda-ai/mloda/issues/)

## 🍳 Think of mloda Like Cooking Recipes

**Traditional Data Pipelines** = Making everything from scratch
- Want pasta? Make noodles, sauce, cheese from raw ingredients
- Want pizza? Start over - make dough, sauce, cheese again
- Want lasagna? Repeat everything once more
- Can't share recipes easily - they're mixed with your kitchen setup

**mloda** = Using recipe components
- Create reusable recipes: "tomato sauce", "pasta dough", "cheese blend"
- Use same "tomato sauce" for pasta, pizza, lasagna
- Switch kitchens (home → restaurant → food truck) - same recipes work
- Share your "tomato sauce" recipe with friends - they don't need your whole kitchen

**Result**: Instead of rebuilding the same thing 10 times, build once and reuse everywhere!

### Installation
```bash
pip install mloda
```

### 1. The Core API Call - Your Starting Point

**Complete Working Example with DataCreator**

```python
# Step 1: Create a sample data source using DataCreator
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from typing import Any, Optional
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
import pandas as pd

class SampleData(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"customer_id", "age", "income"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25, 30, 35, None, 45],
            'income': [50000, 75000, None, 60000, 85000]
        })

# Step 2: Load mloda plugins and run pipeline
from mloda_core.api.request import mlodaAPI
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe

PluginLoader.all()

result = mlodaAPI.run_all(
    features=[
        "customer_id",                    # Original column
        "age",                            # Original column
        "standard_scaled__income"         # Transform: scale income to mean=0, std=1
    ],
    compute_frameworks={PandasDataframe}
)

# Step 3: Get your processed data
data = result[0]
print(data.head())
# Output: DataFrame with customer_id, age, and scaled income
```

**What just happened?**
1. **SampleData class** - Created a data source using DataCreator (generates data in-memory)
2. **PluginLoader.all()** - Loaded all available transformations (scaling, encoding, imputation, etc.)
3. **mlodaAPI.run_all()** - Executed the feature pipeline:
   - Got data from `SampleData`
   - Extracted `customer_id` and `age` as-is
   - Applied StandardScaler to `income` → `standard_scaled__income`
4. **result[0]** - Retrieved the processed pandas DataFrame

> **Key Insight**: The syntax `standard_scaled__income` is mloda's **feature chaining**. Behind the scenes, mloda creates a chain of **feature group** objects (`StandardScalingFeatureGroup` → `SourceFeatureGroup`), automatically resolving dependencies. See [Section 2](#2-understanding-feature-chaining-transformations) for full explanation of chaining syntax and [Section 4](#4-advanced-feature-objects-for-complex-configurations) to learn about the underlying feature group architecture.

### 2. Understanding Feature Chaining (Transformations)

**The Power of Double Underscore `__` Syntax**

As mentioned in Section 1, feature chaining (like `standard_scaled__income`) is syntactic sugar that mloda converts into a chain of **feature group objects**. Each transformation (`standard_scaled`, `mean_imputed`, etc.) corresponds to a specific feature group class.

mloda's chaining syntax lets you compose transformations using `__` as a separator:

```python
# Pattern examples (these show the syntax):
#   "standard_scaled__income"                     # Scale income column
#   "mean_imputed__age"                           # Fill missing age values with mean
#   "onehot_encoded__category"                    # One-hot encode category column
#
# You can chain transformations!
# Pattern: {transform2}__{transform1}__{source}
#   "standard_scaled__mean_imputed__income"       # First impute, then scale

# Real working example:
_ = ["standard_scaled__income", "mean_imputed__age"]  # Valid feature names
```

**Available Transformations:**

| Transformation | Purpose | Example |
|---------------|---------|---------|
| `standard_scaled__` | StandardScaler (mean=0, std=1) | `standard_scaled__income` |
| `minmax_scaled__` | MinMaxScaler (range [0,1]) | `minmax_scaled__age` |
| `robust_scaled__` | RobustScaler (median-based, handles outliers) | `robust_scaled__price` |
| `mean_imputed__` | Fill missing values with mean | `mean_imputed__salary` |
| `median_imputed__` | Fill missing values with median | `median_imputed__age` |
| `mode_imputed__` | Fill missing values with mode | `mode_imputed__category` |
| `onehot_encoded__` | One-hot encoding | `onehot_encoded__state` |
| `label_encoded__` | Label encoding | `label_encoded__priority` |

> **Key Insight**: Transformations are read right-to-left. `standard_scaled__mean_imputed__income` means: take `income` → apply mean imputation → apply standard scaling.

**When You Need More Control**

Most of the time, simple string syntax is enough:
```python
# Example feature list (simple strings)
example_features = ["customer_id", "standard_scaled__income", "onehot_encoded__region"]
```

But for advanced configurations, you can explicitly create `Feature` objects with custom options (covered in Section 3).

### 3. Advanced: Feature Objects for Complex Configurations

**Understanding the Feature Group Architecture**

Behind the scenes, chaining like `standard_scaled__income` creates feature group objects:

```python
# When you write this string:
"standard_scaled__income"

# mloda creates this chain of feature groups:
# StandardScalingFeatureGroup (reads from) → IncomeSourceFeatureGroup
```

**Explicit Feature Objects**

For truly custom configurations, you can use `Feature` objects:

```python
# Example (for custom feature configurations):
# from mloda_core.abstract_plugins.components.feature import Feature
# from mloda_core.abstract_plugins.components.options import Options
#
# features = [
#     "customer_id",                                   # Simple string
#     Feature(
#         "custom_feature",
#         options=Options({
#             "custom_param": "value",
#             "mloda_source_features": "source_column",
#         })
#     ),
# ]
#
# result = mlodaAPI.run_all(
#     features=features,
#     compute_frameworks={PandasDataframe}
# )
```

> **Deep Dive**: Each transformation type (`standard_scaled__`, `mean_imputed__`, etc.) maps to a feature group class in `mloda_plugins/feature_group/`. For example, `standard_scaled__` uses `ScalingFeatureGroup`. When you chain transformations, mloda builds a dependency graph of these feature groups and executes them in the correct order. This architecture makes mloda extensible - you can create custom feature groups for your own transformations!

### 4. Data Access - Where Your Data Comes From

**Three Ways to Provide Data**

mloda supports multiple data access patterns depending on your use case:

**1. DataCreator** - For testing and demos (used in our examples)
```python
# Perfect for creating sample/test data in-memory
# See Section 1 for the SampleData class definition using DataCreator:
#
# class SampleData(AbstractFeatureGroup):
#     @classmethod
#     def input_data(cls) -> Optional[BaseInputData]:
#         return DataCreator({"customer_id", "age", "income"})
#
#     @classmethod
#     def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
#         return pd.DataFrame({
#             'customer_id': ['C001', 'C002'],
#             'age': [25, 30],
#             'income': [50000, 75000]
#         })
```

**2. DataAccessCollection** - For production file/database access
```python
# Example (requires actual files/databases):
# from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
#
# # Read from files, folders, or databases
# data_access = DataAccessCollection(
#     files={"customers.csv", "orders.parquet"},           # CSV/Parquet/JSON files
#     folders={"data/raw/"},                                # Entire directories
#     credential_dicts={"host": "db.example.com"}           # Database credentials
# )
#
# result = mlodaAPI.run_all(
#     features=["customer_id", "standard_scaled__income"],
#     compute_frameworks={PandasDataframe},
#     data_access_collection=data_access
# )
```

**3. ApiData** - For runtime data injection (web requests, real-time predictions)
```python
# Example (for API endpoints and real-time predictions):
# from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import ApiInputDataCollection
#
# api_input_data_collection = ApiInputDataCollection()
# api_data = api_input_data_collection.setup_key_api_data(
#     key_name="PredictionData",
#     api_input_data={"customer_id": ["C001", "C002"], "age": [25, 30]}
# )
#
# result = mlodaAPI.run_all(
#     features=["customer_id", "standard_scaled__age"],
#     compute_frameworks={PandasDataframe},
#     api_input_data_collection=api_input_data_collection,
#     api_data=api_data
# )
```

> **Key Insight**: Use **DataCreator** for demos, **DataAccessCollection** for batch processing from files/databases, and **ApiData** for real-time predictions and web services.

### 5. Compute Frameworks - Choose Your Processing Engine

**Using Different Data Processing Libraries**

mloda supports multiple compute frameworks (pandas, polars, pyarrow, etc.). Most users start with pandas:

```python
# Using the SampleData class from Section 1
# Default: Everything processes with pandas
result = mlodaAPI.run_all(
    features=["customer_id", "standard_scaled__income"],
    compute_frameworks={PandasDataframe}  # Use pandas for all features
)

data = result[0]  # Returns pandas DataFrame
print(type(data))  # <class 'pandas.core.frame.DataFrame'>
```

**Why Compute Frameworks Matter:**
- **Pandas**: Best for small-to-medium datasets, rich ecosystem, familiar API
- **Polars**: High performance for larger datasets
- **PyArrow**: Memory-efficient, great for columnar data
- **Spark**: Distributed processing for big data

> **For most use cases**: Start with `compute_frameworks={PandasDataframe}` and switch to others only if you need specific performance characteristics.

### 6. Putting It All Together - Complete ML Pipeline

**Real-World Example: Customer Churn Prediction**

Let's build a complete machine learning pipeline with mloda:

```python
# Step 1: Extend SampleData with more features for ML
# (Reuse the same class to avoid conflicts)
SampleData._original_calculate = SampleData.calculate_feature

@classmethod
def _extended_calculate(cls, data: Any, features: FeatureSet) -> Any:
    import numpy as np
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'customer_id': [f'C{i:03d}' for i in range(n)],
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(30000, 120000, n),
        'account_balance': np.random.randint(0, 10000, n),
        'subscription_tier': np.random.choice(['Basic', 'Premium', 'Enterprise'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'customer_segment': np.random.choice(['New', 'Regular', 'VIP'], n),
        'churned': np.random.choice([0, 1], n)
    })

SampleData.calculate_feature = _extended_calculate
SampleData._input_data_original = SampleData.input_data()

@classmethod
def _extended_input_data(cls) -> Optional[BaseInputData]:
    return DataCreator({"customer_id", "age", "income", "account_balance",
                       "subscription_tier", "region", "customer_segment", "churned"})

SampleData.input_data = _extended_input_data

# Step 2: Run feature engineering pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

result = mlodaAPI.run_all(
    features=[
        "customer_id",
        "standard_scaled__age",
        "standard_scaled__income",
        "robust_scaled__account_balance",
        "label_encoded__subscription_tier",
        "label_encoded__region",
        "label_encoded__customer_segment",
        "churned"
    ],
    compute_frameworks={PandasDataframe}
)

# Step 3: Prepare for ML
processed_data = result[0]
if len(processed_data.columns) > 2:  # Check we have features besides customer_id and churned
    X = processed_data.drop(['customer_id', 'churned'], axis=1)
    y = processed_data['churned']

    # Step 4: Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"🎯 Model Accuracy: {accuracy:.2%}")
else:
    print("⚠️ Skipping ML - extend SampleData first with more features!")
```

**What mloda Did For You:**
1. ✅ Generated sample data with DataCreator
2. ✅ Scaled numeric features (StandardScaler & RobustScaler)
3. ✅ Encoded categorical features (Label encoding)
4. ✅ Returned clean DataFrame ready for sklearn

> **🎉 You now understand mloda's complete workflow!** The same transformations work across pandas, polars, pyarrow, and other frameworks - just change `compute_frameworks`.

## 📖 Documentation

- **[Getting Started](https://mloda-ai.github.io/mloda/chapter1/installation/)** - Installation and first steps
- **[sklearn Integration](https://mloda-ai.github.io/mloda/examples/sklearn_integration_basic/)** - Complete tutorial
- **[Feature Groups](https://mloda-ai.github.io/mloda/chapter1/feature-groups/)** - Core concepts
- **[Compute Frameworks](https://mloda-ai.github.io/mloda/chapter1/compute-frameworks/)** - Technology integration
- **[API Reference](https://mloda-ai.github.io/mloda/in_depth/mloda-api/)** - Complete API documentation

## 🤝 Contributing

We welcome contributions! Whether you're building plugins, adding features, or improving documentation, your input is invaluable.

- **[Development Guide](https://mloda-ai.github.io/mloda/development/)** - How to contribute
- **[GitHub Issues](https://github.com/mloda-ai/mloda/issues/)** - Report bugs or request features
- **[Email](mailto:info@mloda.ai)** - Direct contact

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://github.com/mloda-ai/mloda/blob/main/LICENSE.TXT).
---
