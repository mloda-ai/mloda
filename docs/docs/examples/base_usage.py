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
    # mloda demo: How can we make feature engineering shareable?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define dummy data as plugin
    """)
    return


@app.cell
def _():
    import numpy as np
    from mloda.provider import FeatureGroup, DataCreator

    class DummyData(FeatureGroup):
        @classmethod
        def calculate_feature(cls, data, features):
            n_samples = features.get_options_key("n_samples") or 100
            return {
                "age": np.random.randint(18, 80, n_samples),
                "weight": np.random.normal(70, 15, n_samples),
                "state": np.random.choice(["CA", "NY", "TX", "FL"], n_samples),
                "gender": np.random.choice(["M", "F"], n_samples),
            }

        @classmethod
        def input_data(cls):
            return DataCreator({"age", "weight", "state", "gender"})

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Request mlodaAPI to create features
    """)
    return


@app.cell
def _():
    # We load dependencies.
    from mloda.user import mloda, PluginLoader

    PluginLoader.all()
    # Load plugins into namespace so compute frameworks register.
    _result = mloda.run_all(
        ["age", "weight", "state", "gender"], compute_frameworks=["PyArrowTable", "PandasDataFrame"]
    )
    print(_result)
    return (mloda,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Alternative options to consume data

    - Apidata
    - Files
    - DBs
    - Streams
    - ...

    This is not the heart of mloda.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chain features - automatic dependency resolution
    """)
    return


@app.cell
def _(mloda):
    # Load plugin into namespace again
    _result = mloda.run_all(["age__sum_aggr"], compute_frameworks=["PolarsLazyDataFrame"])
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As long as the plugins exists, we can run any datatransformation.

    ### What is behind the "age__sum_aggr" syntax?
    """)
    return


@app.cell
def _(mloda):
    from mloda.user import Feature, Options

    feature = Feature(
        name="CustomConfiguration",
        options=Options(context={"aggregation_type": "sum", "in_features": Feature("age", options={"n_samples": 5})}),
    )
    _result = mloda.run_all([feature], compute_frameworks=["PolarsLazyDataFrame"])
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the chaining essentially works

    ```python
    class FeatureGroup(ABC):

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:

            # In principle, the resolver checks if the feature group depends on another input feature
            # -> then adds it to the chain of features which need to be resolved
            if feature_name contains "input_feature__sum_aggr":
                return input_feature

        # How does mloda knows a feature matches a feature group?
        # Customizable, but some good guesses
        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: Union[FeatureName, str],
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
    ```

    ### Now we have chaining and matching. Why do we do this?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    class FeatureGroup(ABC):

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            \"\"\"
            This function should be used to calculate the feature.
            \"\"\"

            # data is the incoming data from other feature dependencies or data via mloda

            # features is the configuration
    ```

    ### Business knowledge is in the data and in the configuration, but not in the plugin definition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Big idea

    **Separate business logic from transformation logic:**

    - Plugins = generic transformations (shareable across companies)
    - Data + Config = your business knowledge (stays private)

    → Stop rewriting "sum of a column" at every company

    → Build a shared ecosystem of feature engineering plugins
    """)
    return


if __name__ == "__main__":
    app.run()
