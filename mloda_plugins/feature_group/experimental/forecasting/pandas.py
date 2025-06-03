"""
Pandas implementation for forecasting feature groups.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

from datetime import datetime, timedelta

# Check if required packages are available
SKLEARN_AVAILABLE = True
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None  # type: ignore


from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup


class PandasForecastingFeatureGroup(ForecastingFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFrameWork]]:
        """Define the compute framework for this feature group."""
        return {PandasDataframe}

    @classmethod
    def _check_time_filter_feature_exists(cls, data: pd.DataFrame, time_filter_feature: str) -> None:
        """
        Check if the time filter feature exists in the DataFrame.

        Args:
            data: The pandas DataFrame
            time_filter_feature: The name of the time filter feature

        Raises:
            ValueError: If the time filter feature does not exist in the DataFrame
        """
        if time_filter_feature not in data.columns:
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' not found in data. "
                f"Please ensure the DataFrame contains this column."
            )

    @classmethod
    def _check_time_filter_feature_is_datetime(cls, data: pd.DataFrame, time_filter_feature: str) -> None:
        """
        Check if the time filter feature is a datetime column.

        Args:
            data: The pandas DataFrame
            time_filter_feature: The name of the time filter feature

        Raises:
            ValueError: If the time filter feature is not a datetime column
        """
        if not pd.api.types.is_datetime64_any_dtype(data[time_filter_feature]):
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' must be a datetime column. "
                f"Current dtype: {data[time_filter_feature].dtype}"
            )

    @classmethod
    def _check_source_feature_exists(cls, data: pd.DataFrame, mloda_source_feature: str) -> None:
        """
        Check if the source feature exists in the DataFrame.

        Args:
            data: The pandas DataFrame
            mloda_source_feature: The name of the source feature

        Raises:
            ValueError: If the source feature does not exist in the DataFrame
        """
        if mloda_source_feature not in data.columns:
            raise ValueError(f"Source feature '{mloda_source_feature}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: pd.Series) -> pd.DataFrame:
        """
        Add the forecast result to the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to add
            result: The forecast result to add

        Returns:
            The updated DataFrame
        """
        data[feature_name] = result
        return data

    @classmethod
    def _perform_forecasting(
        cls,
        data: pd.DataFrame,
        algorithm: str,
        horizon: int,
        time_unit: str,
        mloda_source_feature: str,
        time_filter_feature: str,
        model_artifact: Optional[Any] = None,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Perform forecasting using scikit-learn models.

        This method:
        1. Checks if a trained model exists in the artifact
        2. If not, prepares the data and trains a new model
        3. Generates forecasts for the specified horizon
        4. Returns the forecasts and the updated artifact

        Args:
            data: The pandas DataFrame
            algorithm: The forecasting algorithm to use
            horizon: The forecast horizon
            time_unit: The time unit for the horizon
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature
            model_artifact: Optional artifact containing a trained model

        Returns:
            A tuple containing (forecast_result, updated_artifact)
        """
        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for forecasting. Please install it with 'pip install scikit-learn'."
            )

        # Cast data to pandas DataFrame
        df = cast(pd.DataFrame, data)

        # Sort data by time
        df = df.sort_values(by=time_filter_feature).copy()

        # Get the last timestamp in the data
        last_timestamp = df[time_filter_feature].max()

        # Generate future timestamps for forecasting
        future_timestamps = cls._generate_future_timestamps(last_timestamp, horizon, time_unit)

        # Create or load the model
        if model_artifact is None:
            # Create feature matrix for training
            X, y = cls._create_features(df, mloda_source_feature, time_filter_feature)

            # Train the model
            model, scaler = cls._train_model(X, y, algorithm)

            # Create the artifact
            artifact = {
                "model": model,
                "scaler": scaler,
                "last_trained_timestamp": last_timestamp,
                "feature_names": X.columns.tolist(),
            }
        else:
            # Load the model from the artifact
            model = model_artifact["model"]
            scaler = model_artifact["scaler"]
            feature_names = model_artifact["feature_names"]

            # Update the artifact with the new last timestamp
            artifact = model_artifact.copy()
            artifact["last_trained_timestamp"] = last_timestamp

        # Create features for future timestamps
        future_features = cls._create_future_features(df, future_timestamps, mloda_source_feature, time_filter_feature)

        # Scale the features if a scaler is available
        if scaler is not None:
            if isinstance(future_features, pd.DataFrame):
                # Ensure the columns match the training data
                future_features = future_features[artifact["feature_names"]]
                future_features_scaled = scaler.transform(future_features)
            else:
                future_features_scaled = scaler.transform(future_features.reshape(1, -1))
        else:
            future_features_scaled = future_features

        # Generate forecasts
        forecasts = model.predict(future_features_scaled)

        # Create a Series with the forecasts
        forecast_series = pd.Series(
            index=future_timestamps,
            data=forecasts,
            name=f"{algorithm}_forecast_{horizon}{time_unit}__{mloda_source_feature}",
        )

        # Combine with the original data's time index
        combined_index = list(df[time_filter_feature]) + future_timestamps
        result = pd.Series(index=combined_index, dtype=float)
        result.loc[df[time_filter_feature]] = df[mloda_source_feature].values
        result.loc[future_timestamps] = forecast_series.values

        return result, artifact

    @classmethod
    def _generate_future_timestamps(cls, last_timestamp: datetime, horizon: int, time_unit: str) -> List[datetime]:
        """
        Generate future timestamps for forecasting.

        Args:
            last_timestamp: The last timestamp in the data
            horizon: The forecast horizon
            time_unit: The time unit for the horizon

        Returns:
            A list of future timestamps
        """
        future_timestamps = []
        for i in range(1, horizon + 1):
            if time_unit == "second":
                future_timestamps.append(last_timestamp + timedelta(seconds=i))
            elif time_unit == "minute":
                future_timestamps.append(last_timestamp + timedelta(minutes=i))
            elif time_unit == "hour":
                future_timestamps.append(last_timestamp + timedelta(hours=i))
            elif time_unit == "day":
                future_timestamps.append(last_timestamp + timedelta(days=i))
            elif time_unit == "week":
                future_timestamps.append(last_timestamp + timedelta(weeks=i))
            elif time_unit == "month":
                # Approximate a month as 30 days
                future_timestamps.append(last_timestamp + timedelta(days=i * 30))
            elif time_unit == "year":
                # Approximate a year as 365 days
                future_timestamps.append(last_timestamp + timedelta(days=i * 365))
        return future_timestamps

    @classmethod
    def _create_features(
        cls, df: pd.DataFrame, mloda_source_feature: str, time_filter_feature: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features for training the forecasting model.

        Args:
            df: The pandas DataFrame
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature

        Returns:
            A tuple containing (feature_matrix, target_vector)
        """
        # Create a copy of the DataFrame
        df_features = df.copy()

        # Extract target variable
        y = df_features[mloda_source_feature]

        # Create time-based features
        df_features = cls._create_time_features(df_features, time_filter_feature)

        # Create lag features (previous values)
        df_features = cls._create_lag_features(df_features, mloda_source_feature, lags=[1, 2, 3, 7])

        # Drop rows with NaN values (from lag features)
        df_features = df_features.dropna()
        y = y.loc[df_features.index]

        # Drop the original source feature and time filter feature
        X = df_features.drop([mloda_source_feature, time_filter_feature], axis=1)

        return X, y

    @classmethod
    def _create_time_features(cls, df: pd.DataFrame, time_filter_feature: str) -> pd.DataFrame:
        """
        Create time-based features from the datetime column.

        Args:
            df: The pandas DataFrame
            time_filter_feature: The name of the time filter feature

        Returns:
            The DataFrame with additional time-based features
        """
        df = df.copy()

        # Extract datetime components
        df["hour"] = df[time_filter_feature].dt.hour
        df["dayofweek"] = df[time_filter_feature].dt.dayofweek
        df["quarter"] = df[time_filter_feature].dt.quarter
        df["month"] = df[time_filter_feature].dt.month
        df["year"] = df[time_filter_feature].dt.year
        df["dayofyear"] = df[time_filter_feature].dt.dayofyear
        df["dayofmonth"] = df[time_filter_feature].dt.day
        df["weekofyear"] = df[time_filter_feature].dt.isocalendar().week

        # Create cyclical features for time components
        # This helps the model understand the cyclical nature of time
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4.0)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4.0)

        # Is weekend feature
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        return df

    @classmethod
    def _create_lag_features(cls, df: pd.DataFrame, feature_name: str, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lag features (previous values) for the specified feature.

        Args:
            df: The pandas DataFrame
            feature_name: The name of the feature to create lags for
            lags: List of lag periods to create

        Returns:
            The DataFrame with additional lag features
        """
        df = df.copy()
        for lag in lags:
            df[f"{feature_name}_lag_{lag}"] = df[feature_name].shift(lag)
        return df

    @classmethod
    def _create_future_features(
        cls,
        df: pd.DataFrame,
        future_timestamps: List[datetime],
        mloda_source_feature: str,
        time_filter_feature: str,
    ) -> pd.DataFrame:
        """
        Create features for future timestamps.

        Args:
            df: The pandas DataFrame with historical data
            future_timestamps: List of future timestamps to create features for
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature

        Returns:
            A DataFrame with features for future timestamps
        """
        # Create a DataFrame with future timestamps
        future_df = pd.DataFrame({time_filter_feature: future_timestamps})

        # Create time-based features
        future_df = cls._create_time_features(future_df, time_filter_feature)

        # Get the most recent values for lag features
        last_values = df[mloda_source_feature].iloc[-3:].tolist()
        last_values.reverse()  # Reverse to get [t-3, t-2, t-1]

        # Pad with the last value if we don't have enough history
        while len(last_values) < 7:
            last_values.append(last_values[-1] if last_values else 0)

        # Create lag features for future timestamps
        for i, lag in enumerate([1, 2, 3, 7]):
            if i < len(last_values):
                future_df[f"{mloda_source_feature}_lag_{lag}"] = last_values[i]
            else:
                future_df[f"{mloda_source_feature}_lag_{lag}"] = last_values[-1]

        # Drop the time filter feature
        future_df = future_df.drop([time_filter_feature], axis=1)

        return future_df

    @classmethod
    def _train_model(cls, X: pd.DataFrame, y: pd.Series, algorithm: str) -> Tuple[Any, Optional[StandardScaler]]:
        """
        Train a forecasting model using the specified algorithm.

        Args:
            X: The feature matrix
            y: The target vector
            algorithm: The forecasting algorithm to use

        Returns:
            A tuple containing (trained_model, scaler)
        """
        # Create a scaler for feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select the model based on the algorithm
        if algorithm == "linear":
            model = LinearRegression()
        elif algorithm == "ridge":
            model = Ridge(alpha=1.0)
        elif algorithm == "lasso":
            model = Lasso(alpha=0.1)
        elif algorithm == "randomforest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif algorithm == "gbr":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif algorithm == "svr":
            model = SVR(kernel="rbf")
        elif algorithm == "knn":
            model = KNeighborsRegressor(n_neighbors=5)
        else:
            raise ValueError(f"Unsupported forecasting algorithm: {algorithm}")

        # Train the model
        model.fit(X_scaled, y)

        return model, scaler
