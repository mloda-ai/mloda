"""Tests that all mlodaAPI entry points use @classmethod so subclasses dispatch correctly."""

import inspect
import types
from typing import Any, Dict, List, Optional, Set, Union
from unittest.mock import patch

from mloda.core.api.request import mlodaAPI
from mloda.user import Feature, PluginCollector, mloda
from mloda.provider import ApiInputDataFeature, FeatureGroup, FeatureSet
from mloda.user import Index, Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class ClassMethodFeature(FeatureGroup):
    """A simple feature for classmethod dispatch tests."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature(name="cm_id", index=Index(("cm_id",))),
            Feature(name="cm_value", index=Index(("cm_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["ClassMethodFeature"] = data["cm_id"].astype(str) + "_" + data["cm_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


_enabled = PluginCollector.enabled_feature_groups(
    {
        ApiInputDataFeature,
        ClassMethodFeature,
    }
)

_api_data: Dict[str, Dict[str, Any]] = {
    "ClassMethodInput": {
        "cm_id": [1, 2],
        "cm_value": ["a", "b"],
    }
}

_features: List[Union[Feature, str]] = [Feature(name="ClassMethodFeature")]


class TestEntryPointsAreClassMethods:
    """All three public entry points must be @classmethod."""

    def test_run_all_is_classmethod(self) -> None:
        assert isinstance(inspect.getattr_static(mlodaAPI, "run_all"), classmethod)

    def test_stream_all_is_classmethod(self) -> None:
        assert isinstance(inspect.getattr_static(mlodaAPI, "stream_all"), classmethod)

    def test_prepare_is_classmethod(self) -> None:
        assert isinstance(inspect.getattr_static(mlodaAPI, "prepare"), classmethod)


class TestSubclassDispatch:
    """Subclass entry points must construct the subclass, not mlodaAPI."""

    def test_prepare_returns_subclass_instance(self) -> None:
        class CustomAPI(mlodaAPI):
            pass

        session = CustomAPI.prepare(
            _features,
            compute_frameworks={PandasDataFrame},
            api_data=_api_data,
            plugin_collector=_enabled,
        )

        assert isinstance(session, CustomAPI)

    def test_run_all_uses_cls_prepare(self) -> None:
        """run_all must delegate to cls.prepare, not mlodaAPI.prepare."""

        class CustomAPI(mlodaAPI):
            pass

        with patch.object(CustomAPI, "prepare", wraps=CustomAPI.prepare) as mock_prepare:
            CustomAPI.run_all(
                _features,
                compute_frameworks={PandasDataFrame},
                api_data=_api_data,
                plugin_collector=_enabled,
            )
            mock_prepare.assert_called_once()

    def test_stream_all_uses_cls_prepare(self) -> None:
        """stream_all must delegate to cls.prepare, not mlodaAPI.prepare."""

        class CustomAPI(mlodaAPI):
            pass

        with patch.object(CustomAPI, "prepare", wraps=CustomAPI.prepare) as mock_prepare:
            list(
                CustomAPI.stream_all(
                    _features,
                    compute_frameworks={PandasDataFrame},
                    api_data=_api_data,
                    plugin_collector=_enabled,
                )
            )
            mock_prepare.assert_called_once()


class TestBaseClassBehaviorUnchanged:
    """Existing mlodaAPI usage must remain unaffected."""

    def test_run_all_still_works_on_base_class(self) -> None:
        result = mlodaAPI.run_all(
            _features,
            compute_frameworks={PandasDataFrame},
            api_data=_api_data,
            plugin_collector=_enabled,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_stream_all_still_returns_generator(self) -> None:
        result = mlodaAPI.stream_all(
            _features,
            compute_frameworks={PandasDataFrame},
            api_data=_api_data,
            plugin_collector=_enabled,
        )
        assert isinstance(result, types.GeneratorType)

    def test_mloda_alias_run_all_works(self) -> None:
        result = mloda.run_all(
            _features,
            compute_frameworks={PandasDataFrame},
            api_data=_api_data,
            plugin_collector=_enabled,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_stream_all_importable_from_user_still_works(self) -> None:
        from mloda.user import stream_all

        result = stream_all(
            _features,
            compute_frameworks={PandasDataFrame},
            api_data=_api_data,
            plugin_collector=_enabled,
        )
        assert isinstance(result, types.GeneratorType)
        results = list(result)
        assert len(results) == 1
