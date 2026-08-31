"""End-to-end regression: HookContext.input_features must reflect the engine's planning-time
input_features() resolution, carried onto the execution plan's FeatureSet, rather than a
runtime re-call. The dependent feature group's input_features() must run exactly once per
mloda.run_all(...) call, not once during planning and again at runtime to populate HookContext.
"""

from typing import Any, Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

CIF_ROOT_FEATURE = "cif_root_feature_name"
CIF_DEPENDENT_FEATURE = "cif_dependent_feature_name"


class _CifRootFeatureGroup(FeatureGroup):
    """Root feature group producing cif_root_feature_name."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({CIF_ROOT_FEATURE})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {CIF_ROOT_FEATURE: [1, 2, 3]}


class _CifDependentFeatureGroup(FeatureGroup):
    """Depends on cif_root_feature_name; counts input_features() calls."""

    input_features_calls = 0

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        # A name-only match, bypassing the default is_root()/input_features() matching probe:
        # this test counts input_features() calls, so resolution itself must not call it too.
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == CIF_DEPENDENT_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Any]]:
        type(self).input_features_calls += 1
        return {CIF_ROOT_FEATURE}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {CIF_DEPENDENT_FEATURE: [value * 2 for value in data[CIF_ROOT_FEATURE]]}


class _CifCapturingExtender(Extender):
    """Captures every FEATURE_GROUP_CALCULATE_FEATURE HookContext observed during the run."""

    def __init__(self) -> None:
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result


class TestExecutionPlanCarriesDeclaredInputFeatureNames:
    """The engine's planning-time input_features() resolution must reach HookContext without a runtime recompute."""

    def test_hook_context_input_features_matches_engine_resolution_and_calls_once(self) -> None:
        _CifDependentFeatureGroup.input_features_calls = 0
        collector = PluginCollector.enabled_feature_groups({_CifRootFeatureGroup, _CifDependentFeatureGroup})
        extender = _CifCapturingExtender()

        mloda.run_all(
            [Feature(CIF_DEPENDENT_FEATURE)],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            function_extender={extender},
        )

        dependent_contexts = [
            context for context in extender.captured if CIF_DEPENDENT_FEATURE in context.feature_names
        ]
        assert len(dependent_contexts) == 1
        assert dependent_contexts[0].input_features == frozenset({CIF_ROOT_FEATURE})
        assert _CifDependentFeatureGroup.input_features_calls == 1
