import os
from typing import Any

import pyarrow as pa
import sqlite3

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB


class SQLITEReader(ReadDB):
    """
    Base class for SQLite database reading feature groups.

    This feature group enables reading data from SQLite database files, providing
    a flexible mechanism for loading features from local database files. It handles
    connection management, query building, and data conversion to PyArrow format.

    ## Supported Operations

    - `database_connection`: Connect to SQLite database files using file paths
    - `query_building`: Automatically build SQL queries based on requested features
    - `table_discovery`: Automatically discover and match features to tables
    - `data_conversion`: Convert SQLite results to PyArrow tables

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features reference column names from SQLite database tables:

    Examples:
    ```python
    features = [
        "customer_id",      # Column from customers table
        "order_total",      # Column from orders table
        "product_name"      # Column from products table
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with database credentials and configuration:

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options
    from mloda_core.abstract_plugins.components.hashable_dict import HashableDict

    feature = Feature(
        name="customer_name",
        options=Options(
            context={
                "BaseInputData": (
                    SQLITEReader,
                    HashableDict({"sqlite": "/path/to/database.db"})
                )
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic SQLite Feature Access

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.hashable_dict import HashableDict

    # Simple column reference from SQLite database
    feature = Feature(
        name="user_email",
        options=Options(
            context={
                "BaseInputData": (
                    SQLITEReader,
                    HashableDict({"sqlite": "users.db"})
                )
            }
        )
    )
    ```

    ### Multiple Features from Same Database

    ```python
    # Load multiple columns from the same database
    feature1 = Feature(
        name="customer_id",
        options=Options(
            context={
                "BaseInputData": (
                    SQLITEReader,
                    HashableDict({"sqlite": "sales.db"})
                )
            }
        )
    )

    feature2 = Feature(
        name="purchase_amount",
        options=Options(
            context={
                "BaseInputData": (
                    SQLITEReader,
                    HashableDict({"sqlite": "sales.db"})
                )
            }
        )
    )
    ```

    ### Using DataAccessCollection

    ```python
    from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection

    # Configure database access at the collection level
    data_access = DataAccessCollection(
        credential_dicts=HashableDict({"sqlite": "/data/analytics.db"})
    )

    # Features will automatically use the configured database
    feature = Feature(name="revenue")
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `sqlite`: File path to the SQLite database file
    - `table_name`: Automatically determined based on feature name lookup

    ### Group Parameters
    Currently none for SQLITEReader. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Requirements

    - SQLite database file must exist at the specified path
    - Database must be readable and accessible
    - Feature names must match column names in the database tables
    - All features must use the same database file
    - PyArrow library must be installed for data conversion

    ## Additional Notes

    - The reader automatically discovers which table contains requested columns
    - Built queries use SELECT statements for requested columns
    - Results are converted to PyArrow Table format for efficient processing
    - Connection validation occurs before attempting to read data
    - Table names are automatically cached after first feature lookup
    """

    @classmethod
    def db_path(cls) -> str:
        return "sqlite"

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        if isinstance(credentials, HashableDict):
            credential = credentials.data[cls.db_path()]
            return sqlite3.connect(credential)
        else:
            raise ValueError("Credentials must be an HashableDict.")

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        query = "select "

        options = None
        for feature in features.features:
            query += f"{feature.get_name()}, "
            options = feature.options

        query = query[:-2] + " "  # last comma is removed

        query += "from "

        if options is None:
            raise ValueError(
                "Options were not set. Call this after adding a feature to ensure Options are initialized."
            )

        query += f"{cls.get_table(options)};"

        if query is None:
            raise ValueError("query cannot be None")
        elif len(query) == 0:
            raise ValueError("query cannot be empty")
        return query

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        """Checks if the given dictionary is a valid credentials object."""

        if isinstance(credentials, HashableDict):
            db_path = credentials.data.get(cls.db_path(), None)
        else:
            db_path = None

        if not isinstance(db_path, str):
            return False

        if not os.path.isfile(db_path):
            raise ValueError(f"Database file {db_path} does not exist, but key is given.")
        return True

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        query = cls.build_query(features)
        result, column_names = cls.read_db(data_access, query)
        return cls.read_as_pa_data(result, column_names)

    @classmethod
    def read_as_pa_data(cls, result: Any, column_names: Any) -> Any:
        schema = pa.schema([(column_names[i], DataType.infer_arrow_type(result[0][i])) for i in range(len(result[0]))])
        data_dicts = [{column_names[i]: row[i] for i in range(len(row))} for row in result]
        table = pa.Table.from_pylist(data_dicts, schema=schema)
        return table

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        # get tables in the database
        result, _ = cls.read_db(data_access, query="SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [table[0] for table in result]

        # check if the feature_name is in the tables
        for table in table_names:
            result, _ = cls.read_db(data_access, query=f"PRAGMA table_info({table});")
            column_names = [column[1] for column in result]
            if feature_name in column_names:
                cls.set_table_name(data_access, table)
                return True
        return False

    @classmethod
    def set_table_name(cls, data_access: Any, table_name: str) -> None:
        if data_access.data.get("table_name"):
            if data_access.data["table_name"] != table_name:
                raise ValueError(f"Table name is already set to {data_access.data['table_name']} and not {table_name}.")
            return

        data_access.data["table_name"] = table_name

    @classmethod
    def get_table(cls, options: Options | None) -> Any:
        if options is None:
            raise ValueError("Options were not set.")
        return options.get("BaseInputData")[1].data["table_name"]
