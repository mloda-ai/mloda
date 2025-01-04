## Domain

The domain concept represents scenarios, where similar data is originating from different domains, be it business domains, logical domains or simply testing domains.

The framework itself will match the domains of features and feature groups. This means the domains are essentially **filters**! 

However, we only apply them **if they are set for features**. As default, this is not the case!

**Feature parameter**
```python
from mloda_core.abstract_plugins.components.feature import Feature

Feature(name="Revenue",
        domain="Sales"
)
```

**Feature option**
```python
Feature(name="Revenue",
        options={"domain": "Fraud"}
)
```

The domain can also be set in the feature group.

**Feature group**
```python
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.domain import Domain

class ExampleFeature(AbstractFeatureGroup):
    @classmethod
    def get_domain(cls) -> Domain:
        """This function should return the domain for the feature group"""
        return "example_domain"
```
For feature groups, the default domain is default_domain.
