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
* an out-of-range config-path in_features count raises the mixin's standard message

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

from typing import Any

import pytest

from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator

from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncoding as PandasEncoding
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.source_input_feature import (
    SourceInputFeatureComposite,
    SourceTuple,
)
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


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
        group = GeoDistanceFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("point1&point2__haversine_distance")))

        assert set(by_name) == {"point1", "point2"}
        assert by_name["point1"].forward_group is None
        assert by_name["point2"].forward_group is None

    @pytest.mark.parametrize("name", ["placeholder", "my&name__custom"], ids=["plain", "amp_dunder"])
    def test_config_declared_children_have_forward_group_none(self, name: str) -> None:
        group = GeoDistanceFeatureGroup()
        options = Options(group={DefaultOptionKeys.in_features: frozenset({Feature("point1"), Feature("point2")})})

        by_name = _by_name(group.input_features(options, FeatureName(name)))

        assert set(by_name) == {"point1", "point2"}
        assert by_name["point1"].forward_group is None
        assert by_name["point2"].forward_group is None

    def test_config_declared_children_explicit_directives_are_preserved(self) -> None:
        """An explicit allowlist and an explicit False opt-out both survive input_features."""
        group = GeoDistanceFeatureGroup()
        allowlisted = Feature("point1", forward_group={"keep_me"})
        opted_out = Feature("point2", forward_group=False)
        options = Options(group={DefaultOptionKeys.in_features: frozenset({allowlisted, opted_out})})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["point1"].forward_group == frozenset({"keep_me"})
        assert by_name["point2"].forward_group is False


class TestSklearnEncodingInputFeatureDefaults:
    """EncodingFeatureGroup.input_features must leave its source child at the default None sentinel."""

    def test_string_parsed_child_has_forward_group_none(self) -> None:
        """Core hands plugins the base name (a ~part request resolves through it), so the child stays at None."""
        group = EncodingFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("category__onehot_encoded")))

        assert set(by_name) == {"category"}
        assert by_name["category"].forward_group is None

    def test_config_declared_child_has_forward_group_none(self) -> None:
        group = EncodingFeatureGroup()
        options = Options(group={DefaultOptionKeys.in_features: "category"})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"category"}
        assert by_name["category"].forward_group is None

    def test_config_declared_child_explicit_allowlist_is_preserved(self) -> None:
        group = EncodingFeatureGroup()
        source = Feature("category", forward_group={"keep_me"})
        options = Options(group={DefaultOptionKeys.in_features: frozenset({source})})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert by_name["category"].forward_group == frozenset({"keep_me"})


class _CategoryCreator(ATestDataCreator):
    """Supplies the `category` source column."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {"category": ["a", "b", "c", "a", "b"]}


class TestSklearnEncodingPartRequestEndToEnd:
    """Converted from the string-parsed `~1` unit test: a part request reaches input_features by its base name."""

    def test_string_parsed_multi_column_child_has_forward_group_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("sklearn")
        seen: list[dict[str, Feature]] = []
        original = EncodingFeatureGroup.input_features

        def spy(self: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            result = original(self, options, feature_name)
            seen.append(_by_name(result))
            return result

        monkeypatch.setattr(EncodingFeatureGroup, "input_features", spy)
        collector = PluginCollector.enabled_feature_groups({_CategoryCreator, EncodingFeatureGroup})

        mloda(["category__onehot_encoded~1"], [PandasDataFrame], plugin_collector=collector).run()

        assert seen
        for by_name in seen:
            assert set(by_name) == {"category"}
            assert by_name["category"].forward_group is None


class TestSklearnPipelineInputFeatureDefaults:
    """SklearnPipelineFeatureGroup.input_features must leave every source child at the default None sentinel."""

    def test_string_parsed_single_child_has_forward_group_none(self) -> None:
        group = SklearnPipelineFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("raw_features__sklearn_pipeline_preprocessing")))

        assert set(by_name) == {"raw_features"}
        assert by_name["raw_features"].forward_group is None

    def test_string_parsed_multiple_children_have_forward_group_none(self) -> None:
        """Comma-separated source features all carry forward_group is None."""
        group = SklearnPipelineFeatureGroup()

        by_name = _by_name(group.input_features(Options(), FeatureName("income,age__sklearn_pipeline_scaling")))

        assert set(by_name) == {"income", "age"}
        assert by_name["income"].forward_group is None
        assert by_name["age"].forward_group is None

    def test_config_declared_children_have_forward_group_none(self) -> None:
        group = SklearnPipelineFeatureGroup()
        options = Options(group={DefaultOptionKeys.in_features: "income,age"})

        by_name = _by_name(group.input_features(options, FeatureName("placeholder")))

        assert set(by_name) == {"income", "age"}
        assert by_name["income"].forward_group is None
        assert by_name["age"].forward_group is None

    def test_config_declared_children_explicit_directives_are_preserved(self) -> None:
        """An explicit allowlist and an explicit False opt-out both survive input_features."""
        group = SklearnPipelineFeatureGroup()
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
        group = TimeWindowFeatureGroup()
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})

        by_name = _by_name(group.input_features(options, FeatureName("sales__avg_7_day_window")))

        assert set(by_name) == {"sales", "time_filter"}
        assert by_name["sales"].forward_group is None
        assert by_name["time_filter"].forward_group is None

    def test_config_declared_children_have_forward_group_none(self) -> None:
        """Config path: the unspecified source child stays at None; time_filter is None too."""
        group = TimeWindowFeatureGroup()
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
        group = TimeWindowFeatureGroup()
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


class TestConfigPathCountErrorWording:
    """Config-path count errors use the mixin's standard message."""

    @pytest.mark.parametrize(
        "group_cls, operands",
        [
            pytest.param(GeoDistanceFeatureGroup, ["a"], id="geo_distance-below_min"),
            pytest.param(GeoDistanceFeatureGroup, ["a", "b", "c"], id="geo_distance-above_max"),
            pytest.param(TimeWindowFeatureGroup, ["a", "b"], id="time_window-above_max"),
            pytest.param(PandasForecastingFeatureGroup, ["a", "b"], id="forecasting-above_max"),
            pytest.param(EncodingFeatureGroup, ["a", "b"], id="encoding-above_max"),
        ],
    )
    def test_error_message_is_standard_reason(
        self, group_cls: type[FeatureChainParserMixin], operands: list[str]
    ) -> None:
        options = Options(group={DefaultOptionKeys.in_features: operands})

        with pytest.raises(ValueError) as exc_info:
            group_cls().input_features(options, FeatureName("placeholder"))
        assert str(exc_info.value) == group_cls.in_feature_count_reason("placeholder", len(operands))
