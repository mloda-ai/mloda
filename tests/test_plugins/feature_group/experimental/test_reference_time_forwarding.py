"""Targeted reference_time forwarding for same-family time-based feature groups
(issue #579, gaps 3 and 4).

``reference_time`` is a GROUP option. When a time-based feature group (time window,
forecasting) builds its SOURCE input feature in the string-parse branch of
``input_features``, that source feature must forward the consumer's
``reference_time`` -- so that a NESTED same-family source (e.g. a time window over
another time window) resolves its own ``reference_time`` correctly -- but it must
forward ONLY ``reference_time``, not arbitrary unrelated consumer group keys.

Before this fix:
- ``TimeWindowFeatureGroup.input_features`` used ``Feature(source_feature, forward_group=True)``
  for the SOURCE feature, which blanket-forwards EVERY consumer group key (leaks
  unrelated keys).
- ``ForecastingFeatureGroup.input_features`` used ``Feature(source_feature)`` (no
  ``forward_group`` at all) for the SOURCE feature, forwarding NOTHING (reference_time
  is lost for nested sources).

The fix must forward reference_time onto the source feature and nothing else.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from mloda.provider import DefaultOptionKeys
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Features
from mloda.user import Options
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
    """Minimal concrete TimeWindowFeatureGroup, mirroring the pattern used in
    tests/test_plugins/feature_group/experimental/test_time_window_feature_group/test_base_time_window_feature_group.py
    """

    @classmethod
    def _check_reference_time_column_exists(cls, data: Any, reference_time_column: str) -> None:
        pass

    @classmethod
    def _check_reference_time_column_is_datetime(cls, data: Any, reference_time_column: str) -> None:
        pass

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        return set()

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        pass

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        return data

    @classmethod
    def _perform_window_operation(
        cls,
        data: Any,
        window_function: str,
        window_size: int,
        time_unit: str,
        in_features: list[str],
        time_filter_feature: str | None = None,
    ) -> Any:
        return None


def _find_source_feature(features: set[Feature], source_name: str) -> Feature:
    for feature in features:
        if str(feature.name) == source_name:
            return feature
    raise AssertionError(f"No feature named '{source_name}' found in {[str(f.name) for f in features]}")


def test_time_window_source_feature_forwards_only_reference_time() -> None:
    """The SOURCE feature of a time window chain forwards reference_time, not unrelated keys."""
    consumer_options = Options(group={DefaultOptionKeys.reference_time: "custom_ts", "unrelated_key": "leak_me"})
    feature_group = ConcreteTimeWindowFeatureGroup()

    result = feature_group.input_features(consumer_options, FeatureName("sales__avg_3_day_window"))
    assert result is not None

    source_feature = _find_source_feature(result, "sales")

    Features([source_feature], child_options=consumer_options, child_uuid=uuid4())

    assert source_feature.options.group.get(DefaultOptionKeys.reference_time) == "custom_ts"
    assert "unrelated_key" not in source_feature.options.group


def test_forecasting_source_feature_forwards_only_reference_time() -> None:
    """The SOURCE feature of a forecasting chain forwards reference_time, not unrelated keys."""
    consumer_options = Options(group={DefaultOptionKeys.reference_time: "custom_ts", "unrelated_key": "leak_me"})
    feature_group = PandasForecastingFeatureGroup()

    result = feature_group.input_features(consumer_options, FeatureName("sales__linear_forecast_7_day"))
    assert result is not None

    source_feature = _find_source_feature(result, "sales")

    Features([source_feature], child_options=consumer_options, child_uuid=uuid4())

    assert source_feature.options.group.get(DefaultOptionKeys.reference_time) == "custom_ts"
    assert "unrelated_key" not in source_feature.options.group
