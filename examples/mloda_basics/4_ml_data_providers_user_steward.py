import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Provider, User, Steward in mloda
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Roles in mloda

    In this notebook, we will describe three roles which exists in the mloda framework.

    - Providers: Provide access to raw data, business-layer data, or aggregated data.
      - A data scientist or analyst might create simpler datasets or analytical outputs
      - A data engineer might design access to complex data infrastructures, such as data lakes or warehouses
      - Shares plugins and with it access to data

    - Data User: Interacts with mloda by applying plugins while making requests via the mlodaAPI.
      - A data scientist or analyst who needs data and data transformations (features)
      - Consumes features from other parts of the organizations

    - Steward: Ensures lifetime value, availability and governance of data and features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Provider

    Let us look closer into the role of a provider.

    What could a provider create?
    """)
    return


@app.cell
def _():
    # We reuse the data from the first example and just rerun it for the sake of the example, but just in one cell.

    import os
    from mloda.user import mloda
    from mloda.user import Feature, DataAccessCollection, PluginLoader
    from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    plugin_loader = PluginLoader.all()

    # Initialize a DataAccessCollection object
    data_access_collection = DataAccessCollection()

    # Define the folders containing the data
    # Note: We use two paths to accommodate different possible root locations as it depends where the code is executed.
    base_data_path = os.path.join(os.getcwd(), "docs", "docs", "examples", "mloda_basics", "base_data")
    if not os.path.exists(base_data_path):
        base_data_path = os.path.join(os.getcwd(), "base_data")

    # Add the folder to the DataAccessCollection
    data_access_collection.add_folder(base_data_path)

    # As a db cannot work with a folder, we need to add a connection for the db.
    data_access_collection.add_credentials({SQLITEReader.db_path(): os.path.join(base_data_path, "example.sqlite")})

    order_features: list[str | Feature] = ["order_id", "product_id", "quantity", "item_price"]
    payment_features: list[str | Feature] = ["payment_id", "payment_type", "payment_status", "valid_datetime"]
    location_features: list[str | Feature] = ["user_location", "merchant_location", "update_date"]
    categorical_features: list[str | Feature] = ["user_age_group", "product_category", "transaction_type"]
    all_features = order_features + payment_features + location_features + categorical_features

    from mloda.user import PluginCollector
    from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature

    mloda.run_all(
        all_features,
        data_access_collection=data_access_collection,
        compute_frameworks={PyArrowTable},
        plugin_collector=PluginCollector.disabled_feature_groups({ReadDocumentFeature}),
    )
    return (
        PluginCollector,
        PyArrowTable,
        data_access_collection,
        mloda,
        order_features,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### FeatureGroup mloda (plugin) in short

    In mloda, a provider defines access by creating feature groups. Here's an example implementation:

    ```python
    class FeatureGroupClass(FeatureGroup):

        # Root feature definition
        @classmethod
        def input_data(...)
            ...

        # Features derived from other features
        def input_features(...)
            ...

        @classmethod
        def calculate_feature(...)
            ...
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simple example implementation of the FeatureGroup mloda

    In the background, mloda loads the plugins, which were created before, like this one.

    ```python
    class ReadFileFeature(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return ReadFile()

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            reader = cls.input_data()
            if reader is not None:
                data = reader.load(features)
                return data
            raise ValueError(f"Reading file failed for feature {features.get_name_of_one_feature()}.")
    ```

    We use composition to read different data sources. A ReadFile object looks like this:

    ```python
    class CsvReader(ReadFile):
        @classmethod
        def suffix(cls) -> Tuple[str, ...]:
            return (
                ".csv",
                ".CSV",
            )

        @classmethod
        def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
            result = pyarrow_csv.read_csv(data_access)
            return result.select(list(features.get_all_names()))

        @classmethod
        def get_column_names(cls, file_name: str) -> Any:
            read_options = pyarrow_csv.ReadOptions(skip_rows_after_names=1)
            table = pyarrow_csv.read_csv(file_name, read_options=read_options)
            return table.schema.names
    ```

    As you can see, the implementation is flexible in the sense that if you need something, you can adjust it quite easily. The other files like .json, .parquet and the sqlite access are implemented in a similar fashion.
    """)
    return


@app.cell
def _():
    # In the following, we will just adjust a bit the CsvReader to handle a different delimiter.

    from typing import Any, Optional

    from pyarrow import csv as pyarrow_csv

    from mloda.provider import FeatureSet, BaseInputData
    from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
    from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

    class CsvReader2(CsvReader):
        # Adjusted CsvReader2 to handle the new delimiter
        _parse_options = pyarrow_csv.ParseOptions(
            delimiter=",",  # Default delimiter
            quote_char='"',  # Handles quoted strings
            ignore_empty_lines=True,  # Skips empty lines
        )

        @classmethod
        def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
            result = pyarrow_csv.read_csv(data_access, parse_options=cls._parse_options)
            print("We used CsvReader2 to load the data.")
            return result.select(list(features.get_all_names()))

    class ReadFileFeature2(ReadFileFeature):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return CsvReader2()

        @classmethod
        def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
            for column_name in features.get_all_names():
                if column_name in data.column_names:
                    column = data[column_name]
                    if column.null_count == column.length:
                        raise ValueError(f"Column '{column_name}' contains only null values.")

    return (ReadFileFeature2,)


@app.cell
def _(
    PluginCollector,
    PyArrowTable,
    ReadFileFeature2,
    data_access_collection,
    mloda,
    order_features: "list[str | Feature]",
):
    # We can see that the data was loaded using the new CsvReader2.
    # However, this is a rather simple use case. In a real-world scenario, we would have more complex data and more complex operations.
    result = mloda.run_all(
        order_features,
        data_access_collection=data_access_collection,
        compute_frameworks={PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups({ReadFileFeature2}),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Complex plugins

    These can be quite varied:

    - aggregate features
    - entity features
    - historical features

    Additionally, one can also write feature groups for:

    - using feature stores
    - using orchestrator steps
    - lazy evaluated functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quality

    The producer must optimize quality, which includes:

    - defining input and output validators
    - manage the storage and retrieval of artifacts
    - implementing software testing

    An integration test could be done by using mlodamloda.run_all and custom data.

    An example of a unit test could look like:
    ```python
    def test_csv_reader_2(self) -> None:
       def test_parse_options_are_customized(self, mock_read_csv):
            # Ensure the parse options are as expected
            expected_parse_options = pyarrow_csv.ParseOptions(
                delimiter=",",
                quote_char='"',
                ignore_empty_lines=True
            )

            # Call the method to trigger parse options usage
            CsvReader2.load_data(Mock(), Mock(spec=FeatureSet))

            # Verify that the _parse_options in CsvReader2 are customized
            self.assertEqual(CsvReader2._parse_options, expected_parse_options)
    ```

    This allows us to apply software engineering practices consistently throughout the entire data workflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Consequences

    Within mloda, the provider is empowered as the primary driver, owing to the extensive and customizable range of available plugins.

    Unlike traditional data toolchains, mloda provides providers with the flexibility to define their specific start and end points. This enables the versatile application of mloda across different parts of machine learning lifecycle, such as prototyping, training data preparation, or real-time result monitoring.

    This includes the autonomy to define the boundaries of the provider's domain and to govern the outflow of data. Both aspects are managed through feature groups, which remain under the direct control of the provider.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data User Role

    The Data User plays a pivotal role in configuring and utilizing the mloda API for machine learning and data workflows. The mloda API offers flexible configurations to cater to diverse use cases across the ML lifecycle. Below is an outline of the configurations and features that define the Data User's role:

    ```python
    class mlodaAPI:
        def __init__(
            self,
            requested_features: Union[Features, list[Union[Feature, str]]],
            compute_frameworks: Union[Set[Type[ComputeFramework]], Optional[list[str]]] = None,
            links: Optional[Set[Link]] = None,
            data_access_collection: Optional[DataAccessCollection] = None,
            global_filter: Optional[GlobalFilter] = None,
            api_input_data_collection: Optional[ApiInputDataCollection] = None,
            plugin_collector: Optional[PluginCollector] = None,
        ) -> None:

    data = mlodamloda.run_all(requested_feature,...)
    ```

    Let's use the mloda to further explain the Data User role. As shown above, there are several configurations to consider. The key ones are:

    - Which features to request and if the compute_frameworks should be limited?
    - How data is linked?
    - What specific access rights and permissions does the user have?
    - How data is refined to meet the requirements of the use case?

    With all the given configurations, the mloda core is designed, whenever feasible, to follow the process:

    - First, formulate an optimized execution plan
    - Second, to execute the plan accordingly

    What the user mostly gains is that the process is repeatable and can be run in most environments, as long as the plugins are available and the accesses exist (firewalls, credentials).

    The data user could run mloda API in following scenarios:

    - POC notebooks
    - Production code scenarios (model training or realtime prediction)
    - Micro service endpont
    - KPI or QA test data ingestion

    With this, the whole ml lifecycle is represented and plugins can be reused in a testable and repeatable way along this cycle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steward Role

    Stewards typically operate at various levels within an organization.

    It can be the one who produces the data, the business stakeholder responsible for the service, or, in some cases, may not be explicitly defined.

    In mloda, the steward is the one in control of the governance. However, as to date, this system is not included in this open-source offering, as this platform is reserved for development until the plugin ecosystem has a higher degree of maturity.

    We have the plugin functionalities to integrate governance and operations systems in place. Two simple examples can be:

    #### Using organization wide logging

    ```python
    class OtelExtender(Extender):
        def __init__(self) -> None:
            if trace is None:
                return

            # Function to be wrapped by the Extender
            self.wrapped = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

        def wraps(self) -> Set[ExtenderHook]:
            return self.wrapped

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            logger.warning("OtelExtender")
            result = func(*args, **kwargs)
            return result
    ```

    #### Logging data size

    ```python
    class LogSizeOfData(Extender):

        def wraps(self) -> Set[ExtenderHook]:
            # Function to be wrapped by the Extender
            return {ExtenderHook.VALIDATE_INPUT_FEATURE}

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            size = sys.getsizeof(result)
            print(f"Size: {size}")
            return result
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    In this notebook, we explored the roles of Providers, Data Users, and Stewards within mloda. We delved into the responsibilities and functionalities associated with each role, highlighting how they contribute to the overall data lifecycle.

    - Providers are responsible for implementing the plugins, ensuring the accuracy of data access processes, and defining relevant configuration options

    - Data Users create usage configuration and apply the mlodaAPI to receive data

    - Stewards, while not fully covered in this open-source offering, they are critical for governance and ensuring the ongoing availability and maintenance of essential plugins

    Understanding these roles and their interactions, shows how mloda's modular and extensible design is vital in bringing the change to efficient data management practices and processing throughout the machine learning lifecycle
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    While mloda holds a great potential to become a unified portal, we face a key challenge: its plugin coverage is not comprehensive enough to integrate seamlessly across all available tools and technologies. Therefore, active community contributions are absolutely essential to accelerate both its adoption and its ability to transform data management practices
    """)
    return


if __name__ == "__main__":
    app.run()
