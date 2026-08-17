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
    # Intro to the core interfaces of mloda
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    mloda is a robust and flexible data framework tailored for professionals to efficiently manage data and feature engineering. It enables users to abstract processes away from data, in contrast to the current industry setup where processes are usually bound to specific data sets.

    This introductory notebook provides a practical demonstration of how MLoda helps machine learning data workflows by emphasizing data processes over raw data manipulation.

    - It begins by loading data from various sources, such as order, payment, location, and categorical datasets.
    - Next, we showcase mloda's versatility in handling diverse compute frameworks, including PyArrow tables and Pandas DataFrames.
    - Then we leverage mloda's advanced capabilities to integrate data from various sources into cohesive and unified feature sets (details on feature sets are covered in chapter 3).

    Finally, we will conclude by discussing the broader implications of what was done.
    """)
    return


@app.cell
def _():
    # Load all available plugins into the python environment
    from mloda.user import PluginLoader

    plugin_loader = PluginLoader.all()

    # Since there are potentially many plugins loaded, we'll focus on specific categories for clarity.
    # Here, we demonstrate by listing the available 'read' and 'sql' plugins.
    print(plugin_loader.list_loaded_modules("read"))
    print(plugin_loader.list_loaded_modules("sql"))
    return


@app.cell
def _():
    # Optional!
    # We use synthetic dummy data to demonstrate the basic usage.
    # You can run this cell in your own jupyter notebook.
    # They are however not relevant for further understanding.
    #
    # from examples.mloda_basics import create_synthetic_data

    # create_synthetic_data.create_ml_lifecylce_data()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We should see 4 files in the base_data folder. One sqlite example for a db and 3 different file formats.

    Now we want to load the data to look at the content, so we can look at the data.
    """)
    return


@app.cell
def _():
    # Step 1: We want to load typical order information like order_id, product_id, quantity, and item_price.
    from mloda.user import Feature

    order_features: list[str | Feature] = ["order_id", "product_id", "quantity", "item_price"]

    payment_features: list[str | Feature] = ["payment_id", "payment_type", "payment_status", "valid_datetime"]

    location_features: list[str | Feature] = ["user_location", "merchant_location", "update_date"]

    categorical_features: list[str | Feature] = ["user_age_group", "product_category", "transaction_type"]
    return (
        Feature,
        categorical_features,
        location_features,
        order_features,
        payment_features,
    )


@app.cell
def _():
    # Step 2: We specify the data sources to load
    import os
    from mloda.user import DataAccessCollection
    from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader

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
    return (data_access_collection,)


@app.cell
def _(
    categorical_features: "list[str | Feature]",
    data_access_collection,
    location_features: "list[str | Feature]",
    order_features: "list[str | Feature]",
    payment_features: "list[str | Feature]",
):
    # Step 3: Request Data Using the Defined Access Collection and Desired Features
    from mloda.user import mloda
    from mloda.user.pyarrow import PyArrowTable

    all_features = order_features + payment_features + location_features + categorical_features
    _result = mloda.run_all(
        all_features, data_access_collection=data_access_collection, compute_frameworks={PyArrowTable}
    )
    for _data in _result:
        # Retrieve data based on the specified feature list and access collection
        # Display the first five entries of each result table and its type
        print(_data[:2], type(_data))
    return all_features, mloda


@app.cell
def _(all_features, data_access_collection, mloda):
    # The data is initially loaded as a Pyarrow table. However, we can easily load it also as a PandasDataFrame.
    from mloda.user.pandas import PandasDataFrame

    _result = mloda.run_all(
        all_features, data_access_collection=data_access_collection, compute_frameworks={PandasDataFrame}
    )
    # Request data using the Pandas compute framework
    for _data in _result:
        # Display the first five entries of each result table and its type
        print(_data[:2], type(_data))
    return


@app.cell
def _(Feature, data_access_collection, mloda):
    # Define features with specific compute frameworks
    order_id = Feature(name="order_id", compute_framework="PandasDataFrame")
    product_id = Feature(name="product_id", compute_framework="PyArrowTable")
    specific_framework_feature_list: list[Feature | str] = [order_id, product_id]
    _result = mloda.run_all(specific_framework_feature_list, data_access_collection=data_access_collection)
    # Request data for the defined features
    for res in _result:
        print("The resulting data structure differs based on the compute framework:")
        # Display the first few rows and data types of the results
        print("\n", res[:3], type(res))
    return


@app.cell
def _(Feature, data_access_collection, mloda):
    from typing import Any, Optional
    from mloda.provider import FeatureGroup, FeatureSet
    from mloda.user import FeatureName, Options, Index, Link, JoinSpec
    from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature

    index = Index(("order_id",))

    class ReadFileFeatureJoin(ReadFileFeature):
        @classmethod
        def index_columns(cls) -> Optional[list[Index]]:
            return [index]

    link = Link.inner(JoinSpec(ReadFileFeatureJoin, index), JoinSpec(ReadFileFeatureJoin, index))

    class ExampleMlLifeCycleJoin(FeatureGroup):
        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            quantity = Feature(name="quantity", compute_framework="PandasDataFrame")
            product_id = Feature(name="product_id", compute_framework="PyArrowTable")
            return {product_id, quantity}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            print(
                "Data from two different sources is now combined into one feature within one data technology: \n",
                data,
                type(data),
                "\n",
            )
            return {"ExampleMlLifeCycleJoin": [1, 2, 3]}

    _result = mloda.run_all(["ExampleMlLifeCycleJoin"], data_access_collection=data_access_collection, links={link})
    print(
        "Final result: ",
        _result[0],
        "\nNote: As no specific compute framework was defined for the result, the output could be in either format.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What Have We Observed So Far?

    1. mloda unifies the interfaces for data for various sources, formats and technologies for the definition of the processes and applying the processes on the data. We used the FeatureGroup, the ComputeFramework and mlodaAPI as interfaces.

    2. It integrates with any techologies, e.g. PyArrow and Pandas, enabling flexible tool choices for data processing.

    3. mloda combines data access and computation, reducing complexity and providing a reusable approach to ML workflows. Data Access can be controlled centrally for different sources of data. Here, we showed folders and a database access.

    We will further deepen the advantages of the used approach in the next notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
