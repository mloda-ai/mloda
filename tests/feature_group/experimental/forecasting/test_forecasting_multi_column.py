"""Regression test for issue #1724: multi-column sources must be rejected."""

from __future__ import annotations

import pandas as pd
import pytest

from mloda.provider import FeatureSet
from mloda.user import Feature, Options
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup


class TestForecastingMultiColumnRejection:
    """A source that resolves to more than one column must raise a clear ValueError."""

    def test_multi_column_source_raises(self) -> None:
        data = pd.DataFrame(
            {
                "reference_time": pd.date_range("2024-01-01", periods=10, freq="D"),
                "onehot_encoded__product~0": range(10),
                "onehot_encoded__product~1": range(10, 20),
            }
        )

        feature = Feature(
            name="placeholder",
            options=Options(
                context={
                    PandasForecastingFeatureGroup.ALGORITHM: "linear",
                    PandasForecastingFeatureGroup.HORIZON: 3,
                    PandasForecastingFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.in_features: "onehot_encoded__product",
                }
            ),
        )

        features = FeatureSet([feature])

        with pytest.raises(ValueError, match="exactly one source column"):
            PandasForecastingFeatureGroup.calculate_feature(data, features)
