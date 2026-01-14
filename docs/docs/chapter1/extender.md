## Overview

The **Extender** class is an abstract base class (ABC) that provides an extensible framework for enhancing and wrapping functions with additional capabilities. It is especially useful for automating and monitoring various operations such as metadata harvesting, messaging integration, and event logging. This class offers a standardized approach to augmenting functions with critical features like performance monitoring, audit trails, and impact analysis.

In the following example, we will reuse the previous feature group example and demonstrate how to monitor the execution time of the **calculate_feature** function using a custom extender.

**Monitoring Execution Time**

We will create a DokuExtender class to monitor and log the time taken for the calculate_feature function of the feature group to execute.
#### 1. Define the Extender
```python
from typing import Set, Any
import time
from mloda.steward import Extender, ExtenderHook
import logging

logger = logging.getLogger(__name__)
```

A simple DokuExtender class:

``` python
class DokuExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}
    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result
```
#### 2. Run the Example with the Extender
We will now run the **mlodaAPI** call, including our custom **DokuExtender** to monitor the execution time of the **calculate_feature** function.
```python
from mloda.user import mloda
from mloda.user import DataAccessCollection
from tests.test_documentation.test_documentation import DokuExtender

file_path = "tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"
data_access_collection = DataAccessCollection(files={file_path})

feature_list = ["id","V1","V2","V3"]

example_feature_list = [f"ExampleB_{f}" for f in feature_list]


mloda.run_all(
    feature_list,
    compute_frameworks={"PyArrowTable"},
    data_access_collection=data_access_collection,
    function_extender={DokuExtender()}
)
```
Expected Output (Logged Execution Times)
``` python
ERROR    test_getting_started:test_getting_started.py:29 Time taken: 0.00454258918762207
ERROR    test_getting_started:test_getting_started.py:29 Time taken: 0.001033782958984375
```

#### 3. Summary

With this simple extender, you can easily log and monitor the execution time of any functionality within feature groups. By extending the Extender class, you can wrap additional behavior—such as performance monitoring, logging, or auditing—around critical functions to enhance observability and traceability in your data processing workflows.

When multiple extenders are provided, they are automatically chained and executed in priority order (lower values first).

#### 4. Discovering Extenders

To list all available extenders and their documentation, use the `get_extender_docs()` function from `mloda.steward`.
