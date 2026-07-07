"""
Failing tests for the forward-by-default follow-up of issue #579.

The engine now forwards ALL consumer group options (except in_features) to input
features whose forward_group is left at the unspecified None sentinel. Plugins
therefore no longer need to stamp forward_group=True on their children, and they
must not: stamping destroys the None sentinel and with it the distinction
between "unspecified" and an explicit author directive.

Spec under test:

* every child returned from ``input_features`` must be left at the default None
  sentinel (``forward_group is None``); the engine default does the forwarding
* explicit user directives on config-declared children (an allowlist frozenset
  or the False opt-out) are preserved, never overwritten

Covered plugins:

- ForecastingFeatureGroup (including its time_filter child)
- GeoDistanceFeatureGroup
- EncodingFeatureGroup (sklearn)
- SklearnPipelineFeatureGroup (sklearn)
- TimeWindowFeatureGroup (including its time_filter child)
- SourceInputFeatureComposite (str sources and SourceTuple sources)

The concrete Pandas subclasses are instantiated because the base classes are
abstract; ``input_features`` itself is defined on the bases under test.
"""

from __future__ import annotations

from mloda.provider import DefaultOptionKeys
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options

from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import PandasSklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.source_input_feature import (
    SourceInputFeatureComposite,
    SourceTuple,
)
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup


def _by_name(result: set[Feature] | None) -> dict[str, Feature]:
    assert result is not None
    return {str(child.name): child for child in result}


class TestForecastingInputFeatureDefaults:
    """ForecastingFeatureGroup.input_features must leave children at the default None sentinel."""

    def test_string_parsed_children_have_forward_group_none(self) -> None:
        """String path: the source child AND the time_filter child carry forward_group is None."""
        group = PandasForecastingFeatureGroup()
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})

        by_name = _by_name(group.input_features(options, FeatureName("sales__linear_forecast_7day")))

        assert set(by_name) == {"sales", "time_filter"}
        assert by_name["sales"].forward_group is None
        assert by_name["time_filter"].forward_group is None

    def test_config_declared_children_have_forward_group_none(self) -> None:
        """Config path: the unspecified source child stays at None; time_filter is None too."""
        group = PandasForecastingFeatureGroup()
        options = Options(
            group={
                DefaultOptionKeys.in_features: "sales",
                DefaultOptionKeys.reference_time: "time_filter",
            }
        )

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"sales", "time_filter"}
        assert by_name["sales"].forward_group is None
        assert by_name["time_filter"].forward_group is None

    def test_config_declared_child_explicit_allowlist_is_preserved(self) -> None:
        """Config path: an explicit user allowlist on the source child is kept untouched."""
        group = PandasForecastingFeatureGroup()
        source = Feature("sales", forward_group={"keep_me"})
        options = Options(
            group={
                DefaultOptionKeys.in_features: frozenset({source}),
                DefaultOptionKeys.reference_time: "time_filter",
            }
        )

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["sales"].forward_group == frozenset({"keep_me"})
        assert by_name["time_filter"].forward_group is None


class TestGeoDistanceInputFeatureDefaults:
    """GeoDistanceFeatureGroup.input_features must leave both point children at the default None sentinel."""

    def test_string_parsed_children_have_forward_group_none(self) -> None:
        group = PandasGeoDistanceFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("point1&point2__haversine_distance")))

        assert set(by_name) == {"point1", "point2"}
        assert by_name["point1"].forward_group is None
        assert by_name["point2"].forward_group is None

    def test_config_declared_children_have_forward_group_none(self) -> None:
        group = PandasGeoDistanceFeatureGroup()
        options = Options(group={DefaultOptionKeys.in_features: frozenset({Feature("point1"), Feature("point2")})})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"point1", "point2"}
        assert by_name["point1"].forward_group is None
        assert by_name["point2"].forward_group is None

    def test_config_declared_children_explicit_directives_are_preserved(self) -> None:
        """An explicit allowlist and an explicit False opt-out both survive input_features."""
        group = PandasGeoDistanceFeatureGroup()
        allowlisted = Feature("point1", forward_group={"keep_me"})
        opted_out = Feature("point2", forward_group=False)
        options = Options(group={DefaultOptionKeys.in_features: frozenset({allowlisted, opted_out})})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["point1"].forward_group == frozenset({"keep_me"})
        assert by_name["point2"].forward_group is False


class TestSklearnEncodingInputFeatureDefaults:
    """EncodingFeatureGroup.input_features must leave its source child at the default None sentinel."""

    def test_string_parsed_child_has_forward_group_none(self) -> None:
        group = PandasEncodingFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("category__onehot_encoded")))

        assert set(by_name) == {"category"}
        assert by_name["category"].forward_group is None

    def test_string_parsed_multi_column_child_has_forward_group_none(self) -> None:
        """The ~N multi-column pattern resolves to the base child, also left at None."""
        group = PandasEncodingFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("category__onehot_encoded~1")))

        assert set(by_name) == {"category"}
        assert by_name["category"].forward_group is None

    def test_config_declared_child_has_forward_group_none(self) -> None:
        group = PandasEncodingFeatureGroup()
        options = Options(group={DefaultOptionKeys.in_features: "category"})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"category"}
        assert by_name["category"].forward_group is None

    def test_config_declared_child_explicit_allowlist_is_preserved(self) -> None:
        group = PandasEncodingFeatureGroup()
        source = Feature("category", forward_group={"keep_me"})
        options = Options(group={DefaultOptionKeys.in_features: frozenset({source})})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["category"].forward_group == frozenset({"keep_me"})


class TestSklearnPipelineInputFeatureDefaults:
    """SklearnPipelineFeatureGroup.input_features must leave every source child at the default None sentinel."""

    def test_string_parsed_single_child_has_forward_group_none(self) -> None:
        group = PandasSklearnPipelineFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("raw_features__sklearn_pipeline_preprocessing")))

        assert set(by_name) == {"raw_features"}
        assert by_name["raw_features"].forward_group is None

    def test_string_parsed_multiple_children_have_forward_group_none(self) -> None:
        """Comma-separated source features all carry forward_group is None."""
        group = PandasSklearnPipelineFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("income,age__sklearn_pipeline_scaling")))

        assert set(by_name) == {"income", "age"}
        assert by_name["income"].forward_group is None
        assert by_name["age"].forward_group is None

    def test_config_declared_children_have_forward_group_none(self) -> None:
        group = PandasSklearnPipelineFeatureGroup()
        options = Options(group={DefaultOptionKeys.in_features: "income,age"})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"income", "age"}
        assert by_name["income"].forward_group is None
        assert by_name["age"].forward_group is None

    def test_config_declared_children_explicit_directives_are_preserved(self) -> None:
        """An explicit allowlist and an explicit False opt-out both survive input_features."""
        group = PandasSklearnPipelineFeatureGroup()
        allowlisted = Feature("income", forward_group={"keep_me"})
        opted_out = Feature("age", forward_group=False)
        options = Options(group={DefaultOptionKeys.in_features: frozenset({allowlisted, opted_out})})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["income"].forward_group == frozenset({"keep_me"})
        assert by_name["age"].forward_group is False


class TestTimeWindowInputFeatureDefaults:
    """TimeWindowFeatureGroup.input_features must leave children at the default None sentinel."""

    def test_string_parsed_children_have_forward_group_none(self) -> None:
        """String path: the source child AND the time_filter child carry forward_group is None."""
        group = PandasTimeWindowFeatureGroup()
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})

        by_name = _by_name(group.input_features(options, FeatureName("sales__avg_7_day_window")))

        assert set(by_name) == {"sales", "time_filter"}
        assert by_name["sales"].forward_group is None
        assert by_name["time_filter"].forward_group is None

    def test_config_declared_children_have_forward_group_none(self) -> None:
        """Config path: the unspecified source child stays at None; time_filter is None too."""
        group = PandasTimeWindowFeatureGroup()
        options = Options(
            group={
                DefaultOptionKeys.in_features: "sales",
                DefaultOptionKeys.reference_time: "time_filter",
            }
        )

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"sales", "time_filter"}
        assert by_name["sales"].forward_group is None
        assert by_name["time_filter"].forward_group is None

    def test_config_declared_child_explicit_allowlist_is_preserved(self) -> None:
        """Config path: an explicit user allowlist on the source child is kept untouched."""
        group = PandasTimeWindowFeatureGroup()
        source = Feature("sales", forward_group={"keep_me"})
        options = Options(
            group={
                DefaultOptionKeys.in_features: frozenset({source}),
                DefaultOptionKeys.reference_time: "time_filter",
            }
        )

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["sales"].forward_group == frozenset({"keep_me"})
        assert by_name["time_filter"].forward_group is None


class TestSourceInputFeatureCompositeDefaults:
    """SourceInputFeatureComposite.input_features must leave every source at the default None sentinel."""

    def test_str_source_yields_forward_group_none(self) -> None:
        """A plain str source is built as a Feature with forward_group is None."""
        options = Options(group={DefaultOptionKeys.in_features: frozenset({"upstream_source"})})

        by_name = _by_name(SourceInputFeatureComposite.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"upstream_source"}
        assert by_name["upstream_source"].forward_group is None

    def test_source_tuple_feature_carries_forward_group_none(self) -> None:
        """A SourceTuple-built feature also carries forward_group is None."""
        source_tuple = SourceTuple(feature_name="upstream_source")
        options = Options(group={DefaultOptionKeys.in_features: frozenset({source_tuple})})

        by_name = _by_name(SourceInputFeatureComposite.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"upstream_source"}
        assert by_name["upstream_source"].forward_group is None
