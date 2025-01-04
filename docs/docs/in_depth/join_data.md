## Join datasets

Combining datasets from various feature groups is crucial for building comprehensive and scalable data processing pipelines. The framework automatically handles data merging in the background to accommodate the following scenarios:

-   **Different Compute Frameworks**: Merging data from feature groups that utilize different underlying compute technologies.
-   **Same Compute Framework, Different Sources**: Combining datasets that use the same compute framework but originate from different data sources.
-   **Same Feature Group, Different Feature Options**: Integrating data from the same feature group configured with different feature options.


_**If we have a feature, which is dependent on a aforementioned setup,**_

-   we need to define a Link,
-   and the ComputeFramework must support the according merge!


We will first discuss the basic building blocks of the **Links** (**Index**, **Join Types**) and then the **merge** in the ComputeFramework.

#### Index

The Index class defines the keys (columns) used to merge datasets. An index is a tuple of strings.

Properties:
    
-   Multi-Index Support: Supports multi-column keys for complex merges.
-   Comparison Operations: Methods to compare indexes and determine subset relationships.
-   Utility Methods: Provides methods to check if the index is a multi-index.

Example:

```python
from mloda_core.abstract_plugins.components.index.index import Index

# Create an Index with a single column
single_column_index = Index(('user_id',))

# Create an Index with multiple columns (multi-index)
multi_column_index = Index(('user_id', 'timestamp'))

# Check if an index is a multi-index
is_multi = multi_column_index.is_multi_index()  # Returns True

# Check if single column is a part of a composite index
is_a_part_of = single_column_index.is_a_part_of_(multi_column_index) # Returns True
```

#### JoinType

Join Types specify how two datasets are merged based on their keys. The framework supports the following join types:

-   Inner Join,
-   Left Join,
-   Outer Join,
-   Right Join (use sparingly; prefer left joins when possible).

```python
from enum import Enum

class JoinType(Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"

join_type = JoinType.INNER
```

#### Link

A Link specifies the relationship of:

-   join type, 
-   the feature groups involved, 
-   and the indexes to use for the join.

```python
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

# Assume FeatureGroupA and FeatureGroupB are defined feature groups
feature_group_a = AbstractFeatureGroup()
feature_group_b = AbstractFeatureGroup()

# Define indexes for each feature group
left_index = Index(('id',))
right_index = Index(('feature_a_id',))

# Create a Link using an inner join
link = Link(
    jointype=JoinType.INNER,
    left=(feature_group_b, left_index),
    right=(feature_group_b, right_index)
)

# Alternatively, use the class method for an inner join
link = Link.inner(
    left=(feature_group_b, left_index),
    right=(feature_group_b, right_index)
)
```

#### mlodaAPI

``` python
from mloda_core.api.request import mlodaAPI

set_of_links = {link}

mlodaAPI.run_all(
    features=["Feature_of_FeatureGroupA", "Feature_of_FeatureGroupB"],
    links=set_of_links
    )
```

mloda will then use the merge implementation in the compute framework and use the given links to join datasets, if needed.

In the following section, we will see how this can look like.

#### Merging Data in the Compute Framework

The compute framework can implement the following methods to handle data merging:

-   **merge_inner**,
-   **merge_left**,
-   **merge_right**,
-   **merge_full_outer**.

These methods are invoked via the final implementation in the abstract **ComputeFramework** class.

Abstract Method Implementation:
``` python
@final
def merge(self, right_data: Any, jointype: JoinType, left_index: Index, right_index: Index) -> Any:
    if jointype == JoinType.INNER:
        return self.merge_inner(right_data, left_index, right_index)
    if jointype == JoinType.LEFT:
        return self.merge_left(right_data, left_index, right_index)
    if jointype == JoinType.RIGHT:
        return self.merge_right(right_data, left_index, right_index)
    if jointype == JoinType.OUTER:
        return self.merge_full_outter(right_data, left_index, right_index)
```

A simplified implementation in a compute framework can look like:
``` python
def merge_inner(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
    return self.framework_merge_function("inner", right_data, left_index, right_index, JoinType.INNER)

def merge_left(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
    return self.framework_merge_function("left outer", right_data, left_index, right_index, JoinType.LEFT)

# ... Implement merge_right and merge_full_outer similarly

def framework_merge_function(
    self, join_type: str, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
) -> Any:
    if left_index == right_index:
        self.data = self.data.join(right_data, keys=left_index.index[0], join_type=join_type)
        return self.data
    else:
        raise ValueError(
            f"JoinType {join_type} with indexes {left_index} and {right_index} are not yet implemented in {self.__class__.__name__}"
        )
```

Key Components:

-   **join_type**: Type of join operation (e.g., "inner", "left outer").
-   **right_data**: Dataset to join with the left dataset.
-   **left_index** and **right_index**: Indexes specifying join keys.
-   **jointype**: Instance of JoinType.

By implementing these merge functions, the compute framework automatically handles data merging operations in the background, aligning with the relationships defined by **Index**, **JoinType**, and **Link**.
