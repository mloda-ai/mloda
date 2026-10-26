"""A missing point yields NaN for that row; the other rows still compute (#2322)."""
import numpy as np
import pandas as pd

from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistance


def test_nan_point_yields_nan_haversine() -> None:
    df = pd.DataFrame({
        "a": [(52.5, 13.4), None],
        "b": [(48.1, 11.6), (48.1, 11.6)],
    })
    result = PandasGeoDistance._calculate_distance(df, "haversine", "a", "b")
    assert result.iloc[0] > 0
    assert np.isnan(result.iloc[1])


def test_nan_point_yields_nan_euclidean() -> None:
    df = pd.DataFrame({
        "a": [(1.0, 2.0), None],
        "b": [(3.0, 4.0), (3.0, 4.0)],
    })
    result = PandasGeoDistance._calculate_distance(df, "euclidean", "a", "b")
    assert result.iloc[0] > 0
    assert np.isnan(result.iloc[1])


def test_point_containing_nan_yields_nan_haversine() -> None:
    df = pd.DataFrame({
        "a": [(52.5, 13.4)],
        "b": [(float("nan"), 13.4)],
    })
    result = PandasGeoDistance._calculate_distance(df, "haversine", "a", "b")
    assert np.isnan(result.iloc[0])
