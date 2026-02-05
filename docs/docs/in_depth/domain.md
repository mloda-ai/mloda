## Domain

The domain concept represents scenarios, where similar data is originating from different domains, be it business domains, logical domains or simply testing domains.

The framework itself will match the domains of features and feature groups. This means the domains are essentially **filters**! 

However, we only apply them **if they are set for features**. As default, this is not the case!

**Feature parameter**
```python
from mloda.user import Feature

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
from mloda.provider import FeatureGroup
from mloda.user import Domain

class ExampleFeature(FeatureGroup):
    @classmethod
    def get_domain(cls) -> Domain:
        """This function should return the domain for the feature group"""
        return "example_domain"
```
For feature groups, the default domain is default_domain.

## Domain Propagation

When a feature depends on other features (via `input_features()`), the parent's domain propagates automatically to child features. You can override this by setting an explicit domain on each dependent feature.

```
+------------------------------------------+---------------+------------------+
| Child Definition                         | Parent Domain | Result           |
+------------------------------------------+---------------+------------------+
| "child" (string)                         | "Sales"       | Inherits "Sales" |
| Feature("child")                         | "Sales"       | Inherits "Sales" |
| Feature("child", domain="Finance")       | "Sales"       | Keeps "Finance"  |
| Any                                      | None          | No domain        |
+------------------------------------------+---------------+------------------+
```

**Example: String-based features inherit domain**
```python
Feature("Revenue", domain="Sales")

class SalesRevenueGroup(FeatureGroup):
    def input_features(self, options, feature_name):
        return {"base_amount", "currency"}  # Both inherit "Sales"
```

**Example: Override domain for cross-domain dependency**
```python
class SalesRevenueGroup(FeatureGroup):
    def input_features(self, options, feature_name):
        return {
            "base_amount",                              # Inherits "Sales"
            Feature("exchange_rate", domain="Finance"), # Uses "Finance"
        }
```
