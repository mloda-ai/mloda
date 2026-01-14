"""
Sklearn integration example cells for mloda.

Each function represents a notebook cell that can be copy-pasted.
Functions return data/results that can be used in subsequent cells.
"""

from copy import copy
from typing import Any, Optional
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
import numpy as np
import pandas as pd


# Create a DataCreator feature group for our sample data
class SklearnDataCreator(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"age", "weight", "state", "gender"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data = pd.DataFrame(data_dict)
        return data


np.random.seed(42)
n_samples = 1000

data_dict = {
    "age": np.random.randint(18, 80, n_samples),
    "weight": np.random.normal(70, 15, n_samples),
    "state": np.random.choice(["CA", "NY", "TX", "FL"], n_samples),
    "gender": np.random.choice(["M", "F"], n_samples),
}


def cell1_create_sample_data() -> Any:
    """Create realistic sample data with missing values."""
    # Create realistic sample data with missing values

    data = pd.DataFrame(data_dict)

    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    data.loc[missing_indices[:50], "age"] = np.nan
    data.loc[missing_indices[50:], "weight"] = np.nan

    print("Sample data with missing values:")
    print(data.head())
    print(f"\nData shape: {data.shape}")
    print(f"Missing values: age={data['age'].isna().sum()}, weight={data['weight'].isna().sum()}")

    return data


def cell2_traditional_sklearn_pipeline(data: Any) -> None:
    """Traditional sklearn pipeline - like your example."""
    # Traditional sklearn pipeline - like your example

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

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
    _data = copy(data)
    preprocessor.fit(_data)  # Learn imputations, encoder categories, and scaler parameters
    X_transformed = preprocessor.transform(_data)  # Apply the transformations

    onehot_feature_names = (
        preprocessor.named_transformers_["categorical"].named_steps["onehot"].get_feature_names_out(["state", "gender"])
    )
    numeric_feature_names = ["age", "weight"]
    all_feature_names = np.concatenate([onehot_feature_names, numeric_feature_names])

    # 6. Display transformed data
    df_transformed = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed, columns=all_feature_names
    )

    print("✅ Transformed dataset with split fit/transform:")
    print(df_transformed.head(2))

    print(f"Traditional pipeline result shape: {X_transformed.shape}")
    print(f"Result: {len(df_transformed.columns)} columns total")


def cell4_mloda_approach() -> Any:
    """mloda approach - using DataCreator pattern."""
    from mloda.user import PluginCollector
    from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
    from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
    from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import (
        PandasMissingValueFeatureGroup,
    )
    from mloda.user import mloda
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    # Enable necessary feature groups
    plugin_collector = PluginCollector.enabled_feature_groups(
        {
            SklearnDataCreator,
            PandasEncodingFeatureGroup,
            PandasScalingFeatureGroup,
            PandasMissingValueFeatureGroup,
        }
    )

    # Define features as strings (normal mloda usage)
    features = [
        "age__mean_imputed__standard_scaled",  # Scale imputed age
        "weight__mean_imputed__standard_scaled",  # Scale imputed weight
        "state__onehot_encoded",  # One-hot encode state
        "gender__onehot_encoded",  # One-hot encode gender
    ]

    # Execute with mloda
    result = mloda.run_all(features, compute_frameworks={PandasDataFrame}, plugin_collector=plugin_collector)  # type: ignore
    _result = result[0]
    _result2 = result[1]
    print("✅ Transformed dataset with split fit/transform:")
    print(_result.head(2))
    print(_result2.head(2))

    print(f"Result: {list(_result.columns), list(_result2.columns)} columns total")

    return result, features


def cell5_demonstrate_feature_chaining() -> None:
    """Feature chaining: The real power of mloda."""
    # Feature chaining: The real power of mloda
    from mloda.user import PluginCollector
    from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
    from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
    from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import (
        PandasMissingValueFeatureGroup,
    )
    from mloda.user import mloda
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    # Enable necessary feature groups
    plugin_collector = PluginCollector.enabled_feature_groups(
        {
            SklearnDataCreator,
            PandasEncodingFeatureGroup,
            PandasScalingFeatureGroup,
            PandasMissingValueFeatureGroup,
            PandasAggregatedFeatureGroup,
        }
    )

    chained_features = [
        "age__mean_imputed__standard_scaled__max_aggr",  # Step 2: Scale imputed
        "weight__mean_imputed__robust_scaled",  # Different scaler for weight
        "state__onehot_encoded~0",  # Access specific one-hot column
    ]

    print("Feature chaining example:")
    print("Original: age")
    print("Step 1:   age__mean_imputed")
    print("Step 2:   age__mean_imputed__standard_scaled")
    print("\nmloda automatically resolves dependencies!")

    result = mloda.run_all(chained_features, compute_frameworks={PandasDataFrame}, plugin_collector=plugin_collector)  # type: ignore
    print(
        result[0].head(2),
        result[1].head(2),
        result[2].head(2),
    )


def cell6_reusability_demo() -> None:
    """Demonstrate reusability with new data."""

    class SecondSklearnDataCreator(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"age", "weight", "state", "gender"})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return pd.DataFrame(
                {
                    "age": np.random.randint(25, 65, 500),
                    "weight": np.random.normal(80, 20, 500),  # Different distribution
                    "state": np.random.choice(["WA", "OR"], 500),  # Different states!
                    "gender": np.random.choice(["M", "F", "Other"], 500),  # New category!
                }
            )

    chained_features = [
        "age__mean_imputed__standard_scaled__max_aggr",  # Step 2: Scale imputed
        "weight__mean_imputed__robust_scaled",  # Different scaler for weight
        "state__onehot_encoded~0",  # Access specific one-hot column
    ]

    plugin_collector = PluginCollector.enabled_feature_groups(
        {
            SecondSklearnDataCreator,
            PandasEncodingFeatureGroup,
            PandasScalingFeatureGroup,
            PandasMissingValueFeatureGroup,
            PandasAggregatedFeatureGroup,
        }
    )

    result = mloda.run_all(chained_features, compute_frameworks={PandasDataFrame}, plugin_collector=plugin_collector)  # type: ignore
    print(
        result[0].head(2),
        result[1].head(2),
        result[2].head(2),
    )


def test_sklearn_integration_example() -> None:
    """Test that demonstrates the complete sklearn integration example."""
    # Cell 1: Create data
    print("\nCreate Data\n")
    data = cell1_create_sample_data()

    # Cell 2: Traditional sklearn
    print("\nTraditional sklearn\n")
    cell2_traditional_sklearn_pipeline(data)

    # Cell 4: mloda approach
    print("\nmloda approach\n")
    cell4_mloda_approach()

    # Cell 5: Feature chaining
    print("\nFeature chaining\n")
    cell5_demonstrate_feature_chaining()

    # Cell 6: Reusability
    print("\nReusability\n")
    cell6_reusability_demo()
