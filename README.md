# mloda: Make data and feature engineering shareable

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

**The One Command That Does Everything**

```python
# This is the heart of mloda. You describe what you want and mloda resolves the dependencies.
from mloda_core.api.request import mlodaAPI

result = mlodaAPI.run_all(
    features=["age", "standard_scaled__weight"]
)

# That's it! You get processed data back
data = result[0]
print(data.head())
```

**What just happened?**
- mloda found your data automatically
- Applied transformations (scaling, encoding)
- Returned clean, ready-to-use DataFrame

> **Key Insight**: As long as the plugins and data accesses exist, mloda can derive any feature automatically.

### 2. Setting Up Your Data

**Using DataCreator - The mloda Way**

```python
# DataCreator: Perfect for testing, demos, and prototyping
# Use this when you need synthetic data or want to test mloda without external files
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

class SampleDataFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls):
        # Define what columns your data will have
        return DataCreator({
            "age", "weight", "state", "income", "target"
        })
    
    @classmethod 
    def calculate_feature(cls, data, features):
        # Generate sample data that matches your DataCreator specification
        # This is where you'd normally load from files, databases, or APIs
        return {
            'age': [25, 30, 35, None, 45, 28, 33],
            'weight': [150, 180, None, 200, 165, 140, 175], 
            'state': ['CA', 'NY', 'TX', 'CA', 'FL', 'NY', 'TX'],
            'income': [50000, 75000, 85000, 60000, None, 45000, 70000],
            'target': [1, 0, 1, 0, 1, 0, 1]
        }
```

**When to Use DataCreator vs Other Data Access Methods:**

- **DataCreator**: For testing, demos, synthetic data, or when you want to generate data programmatically within mloda
- **File Access** (`DataAccessCollection` with files): When your data lives in CSV, JSON, Parquet, etc.
- **Database Access** (`DataAccessCollection` with credentials): When connecting to SQL databases, data warehouses
- **API Access**: When fetching data from REST APIs or other web services

> **Key Insight**: DataCreator is mloda's built-in data generation tool - perfect for getting started without external dependencies. Once you're ready for production, switch to file or database access methods.

**Quick Start with Your Own Data:**
```python
# Replace DataCreator with real data access
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection

# For files
data_access = DataAccessCollection(files={"your_data.csv"})

# For databases  
data_access = DataAccessCollection(
    credential_dicts=[{"host": "your-db.com", "username": "user"}]
)
```

### 3. Understanding What You Get Back

**The Result Structure**

```python
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe

result = mlodaAPI.run_all(features, compute_frameworks={PandasDataframe})

# result is always a LIST of result objects
data_list = result  
# Each object matches your compute framework type: pd.DataFrame, spark.DataFrame, etc.

# Access your processed data
data = result[0]  # Most common case: single result
print(f"Shape: {data.shape}, Columns: {list(data.columns)}")
```

> **Key Insight**: mloda returns a list of results. Most simple cases return a single DataFrame that you access with `result[0]`.

### 4. The Features Parameter

**Feature Object Syntax**

```python
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

# Load all available plugins (required before using features)
PluginLoader.all()

features = [
    "age",                                    # Simple string
    Feature(
            "weight_replaced",
            options=Options(
                group={
                    "imputation_method": "mean",
                    "mloda_source_features": "weight",
                }
            ),
        ),
    "onehot_encoded__state"                  # Chaining syntax
]
```

**Three Ways to Define Features:**
- **Simple strings**: For basic columns like "age"
- **Feature objects**: For explicit configuration and advanced options  
- **Chaining syntax**: Convenient shorthand for transformations

### 5. Compute Frameworks

**Choose Your Processing Engine**

```python
# Different processing engines
features = [
    Feature("age", compute_framework=PandasDataframe.get_class_name()),
    Feature("weight", compute_framework=PolarsDataframe.get_class_name()),
]

# Mixed - familiar, extensive ecosystem
result = mlodaAPI.run_all(features)
```

### 6. Data Access

**Tell mloda Where Your Data Lives**

```python
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection

# Configure data sources
data_access = DataAccessCollection(
    files={"data/customers.csv"},                    # Specific files
    folders={"data/archive/"},                       # Entire directories
    credential_dicts=[{"host": "db.example.com"}]    # Database credentials
)

result = mlodaAPI.run_all(
    features=["age", "standard_scaled__income"],
    compute_frameworks={PandasDataframe},
    data_access_collection=data_access
)
```

> **Key Insight**: Configure data access once globally, and all features can use it automatically.

### 7. Putting It All Together

**Real-World Feature Engineering Pipeline**

```python
# Complete mlodaAPI call
result = mlodaAPI.run_all(
    # What you want
    features=[
        "standard_scaled__age",
        "onehot_encoded__state", 
        "mean_imputed__income"
    ],
    
    # How to process it
    compute_frameworks={PandasDataframe},
    
    # Where to get it
    data_access_collection=DataAccessCollection(files={"data/customers.csv"})
)

# Get your results
processed_data = result[0]
print(f"✅ Created {len(processed_data.columns)} features from {len(processed_data)} rows")

# Use in your ML pipeline
from sklearn.model_selection import train_test_split
X = processed_data.drop('target', axis=1)
y = processed_data['target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

> **🎉 You now understand mloda's core workflow!**

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
