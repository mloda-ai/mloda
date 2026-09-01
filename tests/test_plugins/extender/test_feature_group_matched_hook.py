"""Failing tests for int-051 Phase 2: wiring FEATURE_GROUP_MATCHED into Engine's matching path.

Covers mlodaAPI.prepare/run_all threading function_extender into Engine, Engine.get_function_extender,
HookContext population (run_id/carrier/worker_index/plan_*), and deny-before-match / deny-with-fallback.
"""

import logging
from typing import Any, Optional
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.core.engine import Engine
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Features, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import BaseTestComputeFramework1
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1

_MARKER = "fgmatch051"


class _MatchHookRootFeatureGroup(FeatureGroup):
    """Root feature group: single resolved feature for the basic FEATURE_GROUP_MATCHED assertions."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{_MARKER}_root_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_root_col": [1, 2, 3]}


class _MatchHookColOneFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{_MARKER}_col_one"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_col_one": [1, 2, 3]}


class _MatchHookColTwoFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{_MARKER}_col_two"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_col_two": [4, 5, 6]}


class _MatchVetoFeatureGroupA(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{_MARKER}_veto_col_a"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_veto_col_a": [1, 2, 3]}


class _MatchVetoFeatureGroupB(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{_MARKER}_veto_col_b"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_veto_col_b": [4, 5, 6]}


class _MatchDepthRootFeatureGroup(FeatureGroup):
    """Root of a two-level input_features() chain, for the plan_depth assertions."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{_MARKER}_depth_root_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_depth_root_col": [1, 2, 3]}


class _MatchDepthDerivedFeatureGroup(FeatureGroup):
    """Requested top-level; declares the root feature group's column as its one input feature."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(f"{_MARKER}_depth_root_col")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[f"{_MARKER}_depth_derived_col"] = data[f"{_MARKER}_depth_root_col"]
        return data


class _MatchContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_MATCHED}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


class _MatchListCapturingExtender(Extender):
    """Appends every captured HookContext for FEATURE_GROUP_MATCHED, in call order."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_MATCHED}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result


def _require_int(value: Optional[int]) -> int:
    assert value is not None
    return value


def _feature_name_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    """Best-effort: find the Feature being resolved among the wrapped resolution's arguments."""
    for value in (*args, *kwargs.values()):
        name = getattr(value, "name", None)
        if name is not None:
            return str(name)
    return None


class _MatchVetoExtender(Extender):
    """raise_on_error selects deny-before-match (True, default) vs deny-with-fallback (False)."""

    def __init__(self, veto_feature_name: str, raise_on_error: bool = True) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error
        self.name = "match_veto"
        self._veto_feature_name = veto_feature_name

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_MATCHED}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if _feature_name_from_args(args, kwargs) == self._veto_feature_name:
            raise RuntimeError(f"denied match for {self._veto_feature_name}")
        return func(*args, **kwargs)


class TestFeatureGroupMatchedHookFiresOnResolve:
    def test_hook_fires_once_with_correct_feature_group_class(self) -> None:
        extender = _MatchContextCapturingExtender()

        mloda.prepare(
            [Feature(f"{_MARKER}_root_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({_MatchHookRootFeatureGroup}),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert extender.captured is not None
        assert extender.captured.hook == ExtenderHook.FEATURE_GROUP_MATCHED
        assert extender.captured.feature_group_class == (
            f"{_MatchHookRootFeatureGroup.__module__}.{_MatchHookRootFeatureGroup.__qualname__}"
        )
        assert extender.captured.feature_names == (f"{_MARKER}_root_col",)


class TestRunIdConsistentAcrossMatches:
    def test_run_id_is_non_empty_and_equal_to_session_run_id_across_matches(self) -> None:
        extender = _MatchListCapturingExtender()

        session = mloda.prepare(
            [Feature(f"{_MARKER}_col_one"), Feature(f"{_MARKER}_col_two")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {_MatchHookColOneFeatureGroup, _MatchHookColTwoFeatureGroup}
            ),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert len(extender.captured) == 2
        assert isinstance(session.run_id, str)
        assert session.run_id
        run_ids = {context.run_id for context in extender.captured}
        assert run_ids == {session.run_id}


class TestCarrierAndWorkerIndexNoneDuringMatch:
    def test_carrier_and_worker_index_are_none_on_every_match_context(self) -> None:
        extender = _MatchListCapturingExtender()

        mloda.prepare(
            [Feature(f"{_MARKER}_col_one"), Feature(f"{_MARKER}_col_two")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {_MatchHookColOneFeatureGroup, _MatchHookColTwoFeatureGroup}
            ),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert len(extender.captured) == 2
        assert all(context.carrier is None for context in extender.captured)
        assert all(context.worker_index is None for context in extender.captured)


class TestNoExtenderRegisteredBaselineRegressionGuard:
    """BASELINE (already passes today): matching with no function_extender registered is unaffected.

    Not a new-behavior assertion; guards the "activate only when needed" short-circuit once
    FEATURE_GROUP_MATCHED is wired in, so a future refactor can't silently break it.
    """

    def test_prepare_without_function_extender_resolves_successfully(self) -> None:
        session = mloda.prepare(
            [Feature(f"{_MARKER}_root_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({_MatchHookRootFeatureGroup}),
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert session.resolution_report()[0].feature_name == f"{_MARKER}_root_col"


class TestDenyBeforeMatch:
    """A raise_on_error=True (default) extender that raises inside __call__ instead of delegating
    prevents the match/resolution from completing for the targeted feature only."""

    def test_veto_raises_and_propagates_for_the_targeted_feature(self) -> None:
        veto_name = f"{_MARKER}_veto_col_a"
        extender = _MatchVetoExtender(veto_name)

        with pytest.raises(RuntimeError, match="denied match"):
            mloda.run_all(
                [Feature(veto_name)],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {_MatchVetoFeatureGroupA, _MatchVetoFeatureGroupB}
                ),
                parallelization_modes={ParallelizationMode.SYNC},
                function_extender={extender},
            )

    def test_veto_does_not_affect_an_unrelated_sibling_feature_requested_alone(self) -> None:
        veto_name = f"{_MARKER}_veto_col_a"
        sibling_name = f"{_MARKER}_veto_col_b"
        extender = _MatchVetoExtender(veto_name)

        result = mloda.run_all(
            [Feature(sibling_name)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({_MatchVetoFeatureGroupA, _MatchVetoFeatureGroupB}),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert len(result) == 1


class TestDenyWithFallback:
    """A raise_on_error=False extender that raises inside __call__ still lets resolution succeed
    (falls back to the wrapped resolution, per _invoke_extender's warning-only semantics)."""

    def test_warning_only_veto_logs_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        veto_name = f"{_MARKER}_veto_col_a"
        extender = _MatchVetoExtender(veto_name, raise_on_error=False)

        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [Feature(veto_name)],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {_MatchVetoFeatureGroupA, _MatchVetoFeatureGroupB}
                ),
                parallelization_modes={ParallelizationMode.SYNC},
                function_extender={extender},
            )

        assert len(result) == 1
        assert any(record.levelno == logging.WARNING and "denied match" in record.message for record in caplog.records)


class TestPlanCountsAndDepthOnMatchContext:
    def test_plan_fields_are_ints_and_depth_reflects_the_input_features_chain(self) -> None:
        extender = _MatchListCapturingExtender()
        derived_name = _MatchDepthDerivedFeatureGroup.get_class_name()

        mloda.prepare(
            [Feature(derived_name)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {_MatchDepthRootFeatureGroup, _MatchDepthDerivedFeatureGroup}
            ),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert len(extender.captured) == 2
        for context in extender.captured:
            assert isinstance(context.plan_feature_count, int)
            assert context.plan_feature_count >= 0
            assert isinstance(context.plan_node_count, int)
            assert context.plan_node_count >= 0
            assert isinstance(context.plan_depth, int)
            assert context.plan_depth >= 0

        by_class = {context.feature_group_class: context for context in extender.captured}
        derived_key = f"{_MatchDepthDerivedFeatureGroup.__module__}.{_MatchDepthDerivedFeatureGroup.__qualname__}"
        root_key = f"{_MatchDepthRootFeatureGroup.__module__}.{_MatchDepthRootFeatureGroup.__qualname__}"
        assert by_class[derived_key].plan_depth == 0
        assert by_class[root_key].plan_depth == 1

        feature_counts = [_require_int(context.plan_feature_count) for context in extender.captured]
        assert feature_counts[0] < feature_counts[1]


class TestEngineFunctionExtenderAndRunIdConstruction:
    """Engine accepts function_extender/run_id kwargs and get_function_extender delegates to
    the Phase-1 free function (mloda.core.abstract_plugins.function_extender.get_function_extender)."""

    def test_engine_accepts_kwargs_and_get_function_extender_delegates(self) -> None:
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_derived_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_derived_accessible_plugins.return_value = {
                BaseTestFeatureGroup1: [BaseTestComputeFramework1],
            }
            extender = _MatchContextCapturingExtender()
            features = Features(["BaseTestFeature1"])
            compute_framework: set[type[ComputeFramework]] = {BaseTestComputeFramework1}

            engine = Engine(
                features,
                compute_framework,
                None,
                function_extender={extender},
                run_id="fgmatch051-direct-run-id",
            )

            assert engine.get_function_extender(ExtenderHook.FEATURE_GROUP_MATCHED) is extender
            assert engine.get_function_extender(ExtenderHook.JOIN) is None
