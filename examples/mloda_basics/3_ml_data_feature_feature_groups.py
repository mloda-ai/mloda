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
    # Data, Feature, FeatureSets and FeatureGroups in mloda
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    mloda focuses on the processes around data. This means we need to abstract different parts of what is usually summed up in the term "data" into distinct objects.

    These key objects are:

    - Data
    - Feature
    - FeatureGroup
    - FeatureSet

    This notebook will explain the relations shown in the graph below.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%mermaidjs
    # graph LR
    #
    #     User[mloda User] --> | requests | Feature
    #
    #     Feature --> | matches | FeatureGroup
    #
    #     FeatureSet --> | uses | CalculateFunction
    #
    #     subgraph FeatureGroup
    #         FeatureSet
    #         CalculateFunction
    #     end
    #
    #     CalculateFunction --> | accesses | Data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    In mloda, data is considered an object that describes how to access the data. It could be:

    - a dataframe (pandas or polars)
    - an unstructured object (json)
    - a URL
    - an object containing a lazy evaluated function

    As a hard requirement, there must be a way to relate data to a feature. Often, this is done using a name-based approach, other methods could be used as well.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature

    A feature is an object that configures the procedural representation of data, but not the process nor data itself. A feature typically includes configuration options:

    - Name
    - Options (Configurations)
    - Domains
    - Compute Framework
    - Data Type

    ### How do we relate Data and Feature?

    We cannot do this directly. We need to relate the Feature with a FeatureGroup first.

    ## Feature Group

    A FeatureGroup group describe Features, which share data processes and share how the configuration of Features are applied to the data. A FeatureGroup also contains configurations created be the Provider if needed e.g. if a FeatureGroup is only valid for a specific technology.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why do we relate a Feature to a FeatureGroup

    We need to match the Feature with a FeatureGroup, as the FeatureGroup "knows" how to use the access description of the data. Additionally, the FeatureGroup "knows" which other required input Features the Feature needs. mloda will add these required input Features to be resolved as well.

    ```python
    # This could be:
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {OrderAmount, Datetime, ID}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How do we relate a Feature to a FeatureGroup

    To match a Feature with a FeatureGroup, we use the IdentifyFeatureGroupClass functionality of the Engine, which checks the following properties of the FeatureGroup Plugins:

    - Is _filter_feature_group_by_criteria matching?
    - Is domain matching?
    - Is compute framework matching?
    - Are links matching?
    - When multiple feature groups match: Are these just feature groups which have inheritance? If so, use the child.

    As a result, there should be only one feature group that is possible to use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Let us look a bit more into: _filter_feature_group_by_criteria

    ```python
    @classmethod
    def match_feature_group_criteria(
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None
        )
        ...
    ```

    Every feature group must implement this function. However, most will inherit the default behaviour from the FeatureGroup base class.

    The default behaviour covers mostly name based approaches to identify a feature (equal or prefix of a feature name).
    But this can be also a call to a webservice, which knows which data supports or could be any other algorithmic solution.

    As example could be this sqlite database, where we check the table for metainformation.

    ```python
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
    ```

    This means, you are open to customize this logic to match a Feature to a FeatureGroup. But please, do not query the database for every single feature to feature group match lookup. :)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What if we have multiple Features sharing the same FeatureGroup?

    To group features under the same FeatureGroup, we use the FeatureSet object.

    For features to share a FeatureSet, they must also have the same configuration and compute_framework.
    If the configurations differ, mloda will automatically create separate FeatureSets.

    ```python
    def similarity_hash(self) -> int:
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )
        return hash((self.options, compute_frameworks_hashable))
    ```

    Examples:

     - Testing migrations: Feature(A, Polars), Feature(B, Pandas)
     - Sliding time windows: Feature(A, 10 days), Feature(B, 20 days)

    This means inbetween Data to Feature, we have another abstraction, the FeatureSet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FeatureSet

    A FeatureSet is a collection of Features, which share the same configuration and the same FeatureGroup. This FeatureSet is used by the FeatureGroup to apply the operations on the data to receive the requested Feature.

    They are created by mloda. One can access its properties during the calculate_feature function, but are not created by any user.

    ```python
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        ...
    ```

    In that sense, the FeatureSet has informations of

    - the Features itself
    - filters
    - names
    - artifacts
    - and has some convenience functionalities for easier access for the user.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    In this notebook, we explored the key concepts and components in mloda related to data, features, feature sets, and feature groups. We discussed how data is accessed and represented, the definition and role of features, and how features are grouped and managed within feature sets and feature groups. Understanding these components is crucial for building robust and reusable machine learning pipelines. By abstracting and organizing data and features in this manner, mloda ensures consistency, flexibility, and scalability in machine learning workflows.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%mermaidjs
    # graph LR
    #
    #     User[mloda User] --> | requests | Feature
    #
    #     Feature --> | matches | FeatureGroup
    #
    #     FeatureSet --> | uses | CalculateFunction
    #
    #     subgraph FeatureGroup
    #         FeatureSet
    #         CalculateFunction
    #     end
    #
    #     CalculateFunction --> | accesses | Data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In short: we abstracted away processes from data.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%mermaidjs
    # graph LR
    #
    #     All[mloda] --> CalculateFunction
    #     subgraph Process
    #         All
    #         CalculateFunction
    #     end
    #     CalculateFunction --> | accesses | Data
    return


if __name__ == "__main__":
    app.run()
