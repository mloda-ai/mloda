"""
Feature Configuration Plugin
-----------------------------
Provides declarative configuration for mloda features.

This plugin enables defining features through JSON/YAML configuration files instead of Python code,
making feature engineering more maintainable and accessible to non-developers.

Supported Patterns
------------------
- **Simple features**: Plain string feature names (e.g., "age", "salary")
- **Features with options**: Configure feature behavior with options dict
- **Chained features**: Multi-step transformations using `__` syntax (e.g., "scaled__imputed__age")
- **Group/context options**: Separate performance and runtime options for optimization
- **Multi-column access**: Select specific columns from multi-output features using `~` syntax
- **Feature references**: Reference other features as sources using `@feature_name` syntax
- **Multiple sources**: Features requiring multiple inputs (e.g., distance from lat/lon)

Input Formats
-------------
- JSON: Standard JSON format
- YAML: Coming soon

Example Configuration
--------------------
See `tests/test_plugins/config/feature/test_config_features.json` for a comprehensive example
demonstrating all supported patterns.

Basic Usage
-----------
```python
from mloda_plugins.config.feature.loader import load_features_from_config

# Load features from JSON string
config_str = '''[
    "age",
    {"name": "scaled_age", "mloda_source": "age", "options": {"method": "standard"}},
    {"name": "distance", "mloda_sources": ["lat", "lon"], "options": {"type": "euclidean"}}
]'''

features = load_features_from_config(config_str, format="json")
```

Schema Export
-------------
Get the JSON schema for validation:
```python
from mloda_plugins.config.feature.models import feature_config_schema
schema = feature_config_schema()
```

Installation
------------
Install dependencies: pip install -r mloda_plugins/config/feature/requirements.txt
"""
