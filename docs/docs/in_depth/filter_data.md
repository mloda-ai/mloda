## Filter

In the data and machine learning world, a filter is a technique used to narrow down or preprocess data by removing irrelevant or unwanted information based on specific conditions, improving the quality and relevance of the dataset for analysis or modeling.

That however means that a data project typically contains multiple filters. For this, this project uses the **GlobalFilter** as a container for multiple **SingleFilters**. 

#### SingleFilter

-    filter_feature: It can be a Feature or a feature name as string.
-    parameter: A dictionary of parameters to filter. Example: {"min": 2, "max": 3}
-    filter_type: It can be a str or a FilterTypeEnum.

#### FilterTypeEnum

This class is supposed to created similarity in the framework and by framework users.

``` python
class FilterTypeEnum(Enum):
    min = "min"
    max = "max"
    equal = "equal"
    range = "range"
    regex = "regex"
    categorical_inclusion = "categorical_inclusion"
```

#### GlobalFilter

The GlobalFilter provides methods to add filters to the collection. The prefered way to use the GlobalFilter is by using the following functions.

**add_filter**: Adds a single filter to the GlobalFilter object.

-   filter_feature
-   filter_type
-   parameter

**add_time_and_time_travel_filters**: Adds time and time travel filtering to the GlobalFiltering. This is a convenience API. Due to the complexity of time in data/ml/ai projects, this function should be used.

This method is useful for **filtering data based on time ranges** (event) and **validity periods** (valid).

**Event Time Filter**: For historical data (e.g., checking if a customer had a valid contract at the event time), only the event time filter is needed.
    
**Time Travel Filter**: If prior actions (e.g., payments made before the event) are relevant, the time travel filter is required.

Typically, **valid_to matches the event timestamp**, but in cases like payment plans, where payments occur after creation, some payments may be excluded based on the valid_to data.

Parameters:

-   event_from (datetime): Start of the time range (with timezone).
-   event_to (datetime): End of the time range (with timezone).
-   valid_from (Optional[datetime]): Start of the validity period (optional, with timezone).
-   valid_to (Optional[datetime]): End of the validity period (optional, with timezone).
-   max_exclusive (bool): If True, **valid_to** values are treated as exclusive.
-   time_filter_feature: the feature description for the time filter. Default is "time_filter".
-   time_travel_filter_feature: the feature description for the time travel filter. Default is "time_travel_filter".

The **single_filters** created will be converted to UTC as ISO 8601 formatted strings to ensure consistency
    across time zones and avoid ambiguity when comparing or processing time-based data.

#### How to create a collection of single filters (GlobalFilter)

In this example, we simply instantiate a GlobalFilter and add a SingleFilter.

```python
from mloda_core.filter.global_filter import GlobalFilter

global_filter = GlobalFilter()
global_filter.add_filter("example_order_id", "eq", {"value": 1})

global_filter.filters
```

Result

``` python
{<SingleFilter(feature_name=example_order_id, type=eq, parameters=(('value', 1),))>}
```

#### How to deal with time filters

In this example, we show how one can manage datetime relations.

```python
from datetime import datetime, timezone

event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)

global_filter.add_time_and_time_travel_filters(event_from, event_to, valid_from, valid_to)

global_filter.filters
```

Result

``` python
{<SingleFilter(feature_name=time_travel_filter, type=range, parameters=(('max', '2022-12-31T00:00:00+00:00'), ('max_exclusive', True), ('min', '2022-01-01T00:00:00+00:00')))>, 
 <SingleFilter(feature_name=time_filter, type=range, parameters=(('max', '2023-12-31T00:00:00+00:00'), ('max_exclusive', True), ('min', '2023-01-01T00:00:00+00:00')))>}
```


#### Example access to Filters in the Feature Group

Now, we need to also use a FeatureGroup which supports it. In this example, we show where we can access the SingleFilters.
The implementation of the concrete filters is dependent on the feature group. This example is rather complex as we filter a python dictionary. 

Further, the feature is a data creator, so we create the data here itself. 

```python
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from typing import Any, Union, Set, Type, Optional
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_core.api.request import mlodaAPI

class ExampleOrderFilter(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "example_order_id"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        _data_creator = {cls.get_class_name(): [1, 2, 3],
                         "example_order_id": [2, 1, 1]
                         }
        # The following algorithm is naive and rather should show an example than a normal use case.
        # The filter implementation highly depends on the feature group!
        # Extract the filter value and filter_name information from the filters.
        for filter in features.filters:
            filter_value = filter.parameter[0]
            filter_name = filter.filter_feature.name
            break
        # Create the order_id filter
        order_id_filter = [i for i, order_id in enumerate(_data_creator[filter_name]) if order_id == filter_value[1]]
        # Apply the filter
        filtered_data = {
            key: [value[i] for i in order_id_filter]
            for key, value in _data_creator.items()
            if key != filter_name
        }
        return filtered_data
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}

result = mlodaAPI.run_all(
    ["ExampleOrderFilter"], 
    global_filter=global_filter
)
result[0]
```
Expected Output

```python
ExampleOrderFilter: [[2,3]]
```

Although this example is complex, it is noteworthy, that the framework considers filters as features and setup as that in the framework. 

### Summary

This filtering system improves data preprocessing through GlobalFilter and SingleFilters, allowing flexible, condition-based refinement, including time-based filtering. It maintains consistency using FilterTypeEnum and supports complex machine learning use cases. If you encounter a commonly used filter not yet included, feel free to open an issue or submit a pull request.