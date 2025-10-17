from typing import Any, Optional
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData


class ApiInputDataFeature(AbstractFeatureGroup):
    """
    Base class for API-based input data feature groups.

    This feature group enables data input through API calls, allowing features to be
    sourced from external APIs rather than static files or databases. It provides a
    flexible mechanism for real-time data integration and dynamic feature retrieval.

    ## Supported Operations

    - `api_data_access`: Access data through API endpoints with configurable parameters
    - `dynamic_feature_mapping`: Map API response fields to feature names
    - `real_time_retrieval`: Fetch data on-demand during feature calculation

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features can reference API-sourced data columns directly by name:

    Examples:
    ```python
    features = [
        "user_profile",          # Direct reference to API column
        "transaction_history",   # Reference to transaction data from API
        "real_time_metrics"      # Real-time metrics from API endpoint
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options to specify API data access configuration:

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options
    from mloda_core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData

    feature = Feature(
        name="user_profile",
        options=Options(
            context={
                ApiInputData.data_access_name(): {
                    "endpoint_name": ["user_profile", "account_info"],
                }
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic API Data Access

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    # Simple API feature reference
    feature = Feature(name="api_user_score")
    ```

    ### Configuration-Based with Endpoint Mapping

    ```python
    from mloda_core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData

    # Map multiple API response fields
    feature = Feature(
        name="customer_data",
        options=Options(
            context={
                "ApiInputData": {
                    "customer_api": ["name", "email", "customer_id"],
                    "preferences_api": ["theme", "language"]
                }
            }
        )
    )
    ```

    ### Real-Time Data Integration

    ```python
    # Feature accessing real-time market data
    feature = Feature(
        name="stock_price",
        options=Options(
            context={
                "ApiInputData": {
                    "market_data_api": ["current_price", "volume", "change_percent"]
                }
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `ApiInputData`: Dictionary mapping API endpoint names to lists of column names
    - API endpoint configuration is passed through the options context

    ### Group Parameters
    Currently none for ApiInputDataFeature. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Requirements

    - API endpoints must be configured in the DataAccessCollection
    - API responses must be in a format compatible with the data processing pipeline
    - Feature names must match column names in the API response data
    - Authentication credentials (if required) must be configured for API access

    ## Additional Notes

    - This feature group acts as a pass-through, returning data as-is from the API
    - API data is matched against feature names using the ApiInputData.matches() method
    - Supports both feature-scoped and global-scoped API data access
    - API calls are typically made during the data loading phase, not feature calculation
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ApiInputData()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data
