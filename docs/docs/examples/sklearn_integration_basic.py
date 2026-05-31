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
    # mloda + scikit-learn Integration: Basic Example

    This notebook demonstrates how mloda enhances scikit-learn workflows by providing reusable, manageable feature transformations.

    ## Quick Comparison: Traditional sklearn vs mloda
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: Create Sample Data
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Create realistic sample data with missing values
    np.random.seed(42)
    n_samples = 1000

    data_dict = {
        "age": np.random.randint(18, 80, n_samples),
        "weight": np.random.normal(70, 15, n_samples),
        "state": np.random.choice(["CA", "NY", "TX", "FL"], n_samples),
        "gender": np.random.choice(["M", "F"], n_samples),
    }

    data = pd.DataFrame(data_dict)

    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    data.loc[missing_indices[:50], "age"] = np.nan
    data.loc[missing_indices[50:], "weight"] = np.nan

    print("Sample data with missing values:")
    print(data.head())
    print(f"\nData shape: {data.shape}")
    print(f"Missing values: age={data['age'].isna().sum()}, weight={data['weight'].isna().sum()}")
    return data, data_dict, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Traditional scikit-learn Pipeline
    """)
    return


@app.cell
def _(data, np, pd):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    # numeric_preprocessor = Pipeline(
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            (
                "imputation_constant",
                SimpleImputer(fill_value="missing", strategy="constant"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_preprocessor, ["state", "gender"]),
            ("numerical", numeric_preprocessor, ["age", "weight"]),
        ]
    )

    # Fit and transform
    preprocessor.fit(data)  # Learn imputations, encoder categories, and scaler parameters
    X_transformed = preprocessor.transform(data)  # Apply the transformations

    onehot_feature_names = (
        preprocessor.named_transformers_["categorical"].named_steps["onehot"].get_feature_names_out(["state", "gender"])
    )
    numeric_feature_names = ["age", "weight"]
    all_feature_names = np.concatenate([onehot_feature_names, numeric_feature_names])

    df_transformed = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,  # type: ignore
        columns=all_feature_names,
    )

    print("✅ Transformed dataset with split fit/transform:")
    print(df_transformed.head(2))

    print(f"Traditional pipeline result shape: {X_transformed.shape}")
    print(f"Result: {len(df_transformed.columns)} columns total")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mloda Approach
    """)
    return


@app.cell
def _(data_dict, pd):
    from typing import Optional, Any
    from mloda.user import mloda
    from mloda.provider import FeatureGroup, BaseInputData, DataCreator, FeatureSet
    from mloda.user import PluginLoader
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    # In mloda, we have the concept of feature groups.
    # A feature group is an abstraction between a data framework and processes of a data transformation.
    # In this example, the data framework is clearly pandas.
    # The processes are typically meta information like names, but lifecyle definition, or dependencies or relations to other data.
    class SklearnDataCreator(FeatureGroup):
        # On the basis on the given data_dict earlier defined and its names, we use a DataCreator to inject the data_dict into the feature group abstraction.
        # Very simply spoken: we load the data.

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            data = pd.DataFrame(
                data_dict
            )  # This function is core to mloda. In this spot, the data framework with the actual data representation meets the defined and resolved processes.
            return data  # With this, we have access to the before and after state of a feature.

        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator(
                {"age", "weight", "state", "gender"}
            )  # If this feature would not load data, we would use the data given from the parameter "data".

    features = [
        "age__mean_imputed__standard_scaled",
        "weight__mean_imputed__standard_scaled",
        "state__onehot_encoded",
        "gender__onehot_encoded",
    ]
    PluginLoader().all()
    _result = mloda.run_all(features, compute_frameworks={PandasDataFrame})
    _result, _result2 = (
        _result[0],
        _result[1],
    )  # One way to get this data is via defined input_data. But there are many more and we should not go to deep into this topic for now.
    print("✅ Transformed dataset with split fit/transform:")
    print(_result.head(2))
    print(_result2.head(2))
    # As next, we will use one method of defining what features we want as result from the mloda framework.
    # We now use a trick to register all known feature groups. mloda will only use those which are loaded into the namespace.
    # And then we execute mloda, which will resolve its dependencies of its feature groups and data frame technologies automatically.
    # Remark 1: We have not yet added the functionality to map the value to a column string back. It is planned. https://github.com/mloda-ai/mloda/issues/46
    # Remark 2: If you see the error "ValueError: Multiple feature groups", please restart the notebook. This happens if we load the class SklearnDataCreator twice into the notebook memory.
    #         I have yet to find a solution for this.
    print(
        f"Result: {list(_result.columns)} \n {list(_result2.columns)} columns total"
    )  # Scale imputed age  # Scale imputed weight  # One-hot encode state  # One-hot encode gender
    return (
        Any,
        BaseInputData,
        DataCreator,
        FeatureGroup,
        FeatureSet,
        Optional,
        PandasDataFrame,
        SklearnDataCreator,
        mloda,
    )


@app.cell
def _(PandasDataFrame, mloda):
    # The beauty and strength of mloda is that we can combine feature groups in a very creative way.
    _chained_features = [
        "age__mean_imputed__standard_scaled__max_aggr",
        "weight__mean_imputed__robust_scaled",
        "state__onehot_encoded~0",
    ]
    _result = mloda.run_all(_chained_features, compute_frameworks={PandasDataFrame})  # Do feature pipelines
    print(
        _result[0].head(2), _result[1].head(2), _result[2].head(2)
    )  # Different scaler for weight  # Access specific one-hot column
    return


@app.cell
def _(
    Any,
    BaseInputData,
    DataCreator,
    FeatureGroup,
    FeatureSet,
    Optional,
    PandasDataFrame,
    SklearnDataCreator,
    mloda,
    np,
    pd,
):
    # We can replace feature groups and dataframe plugins in an easy fashion.
    from mloda.user import PluginCollector

    class SecondSklearnDataCreator(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"age", "weight", "state", "gender"})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            print(f"I, {cls.get_class_name()} AM NOW USED.")
            return pd.DataFrame(
                {
                    "age": np.random.randint(25, 65, 500),
                    "weight": np.random.normal(80, 20, 500),
                    "state": np.random.choice(["WA", "OR"], 500),
                    "gender": np.random.choice(["M", "F", "Other"], 500),
                }
            )

    _chained_features = [
        "age__mean_imputed__standard_scaled__max_aggr",
        "weight__mean_imputed__robust_scaled",
        "state__onehot_encoded~0",
    ]
    _result = mloda.run_all(
        _chained_features,
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.disabled_feature_groups(SklearnDataCreator),
    )
    # We deactivated now the other feature group, so that we use SecondSklearnDataCreator.
    print(
        _result[0].head(2), _result[1].head(2), _result[2].head(2)
    )  # Different distribution  # Different states!  # New category!  # Step 2: Scale imputed  # Different scaler for weight  # Access specific one-hot column
    return


if __name__ == "__main__":
    app.run()
