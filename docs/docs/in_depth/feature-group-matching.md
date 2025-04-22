# Feature Group Matching Criteria

## Overview

The mloda framework uses a matching system to determine which feature group should handle a given feature. This document explains how this matching works.

## Matching Process

When a feature is requested, the system checks all available feature groups to find the one that should handle the feature. This is done through the `match_feature_group_criteria` method in each feature group.

## Default Matching Criteria

By default, a feature group matches a feature if any of these conditions are true:

1. **Root Feature with Matching Input Data**: The feature group is a root feature (has no dependencies) and its input data matches the feature.

2. **Class Name Match**: The feature name exactly matches the feature group's class name.
   ```python
   feature_name == FeatureGroup.get_class_name()
   ```

3. **Prefix Match**: The feature name starts with the feature group's class name as a prefix.
   ```python
   feature_name.startswith(FeatureGroup.prefix())  # Default prefix is "ClassName_"
   ```

4. **Explicitly Supported**: The feature name is in the set of explicitly supported feature names.
   ```python
   feature_name in FeatureGroup.feature_names_supported()
   ```

## Custom Matching

Feature groups can override the `match_feature_group_criteria` method to implement custom matching logic:

```python
@classmethod
def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
    # Convert FeatureName to string if needed
    if isinstance(feature_name, FeatureName):
        feature_name = feature_name.name
        
    # Custom pattern matching
    return re.match(r"^pattern_\w+__source$", feature_name) is not None
```

## Example

For a `ClusteringFeatureGroup`:

- Matches: `cluster_kmeans_5__customer_behavior` (prefix match)
- Matches: `ClusteringFeatureGroup` (class name match)
- Doesn't match: `kmeans_cluster_5__customer_behavior` (wrong pattern)
