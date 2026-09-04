"""Tests wiring ExtenderHook.JOIN into JoinStep._merge_data: HookContext population, the
no-extender baseline, and deny-before-merge / deny-with-fallback. For a keyed join type,
join_keys pairs each left column with its corresponding right column as ``"left=right"``
strings; this is independent of ``swap_merge_sides`` since it reads off the immutable
``Link``, not whichever cfw is "self" at merge time. APPEND/UNION links merge without using
keys, so their join_keys is None.
"""

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_MARKER = "joinhook051"


class _JoinHookLeftFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_left_id", f"{_MARKER}_left_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_left_id": [1, 2, 3], f"{_MARKER}_left_value": ["a", "b", "c"]}


class _JoinHookRightFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_right_id", f"{_MARKER}_right_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_right_id": [1, 2, 3], f"{_MARKER}_right_value": [10, 20, 30]}


class _JoinHookConsumerFeatureGroup(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name=f"{_MARKER}_left_value"),
            Feature(name=f"{_MARKER}_right_value"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data[f"{_MARKER}_left_value"]}


def _join_hook_link() -> Link:
    return Link.inner(
        JoinSpec(_JoinHookLeftFeatureGroup, Index((f"{_MARKER}_left_id",))),
        JoinSpec(_JoinHookRightFeatureGroup, Index((f"{_MARKER}_right_id",))),
    )


_ENABLED = PluginCollector.enabled_feature_groups(
    {_JoinHookLeftFeatureGroup, _JoinHookRightFeatureGroup, _JoinHookConsumerFeatureGroup}
)


class _JoinListCapturingExtender(Extender):
    """Appends every captured HookContext for JOIN, in call order."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.JOIN}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result


class _JoinVetoExtender(Extender):
    """raise_on_error selects deny-before-merge (True, default) vs deny-with-fallback (False)."""

    def __init__(self, raise_on_error: bool = True) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error
        self.name = "join_veto"

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.JOIN}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("denied join")


class TestJoinHookFiresWithCorrectContext:
    def test_join_hook_captures_type_keys_run_id_carrier_worker_index_and_cfw_name(self, flight_server: Any) -> None:
        extender = _JoinListCapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}

        session = mloda.prepare(
            [Feature(name=_JoinHookConsumerFeatureGroup.get_class_name())],
            links={_join_hook_link()},
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        result = session.run(
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
            function_extender={extender},
            carrier=carrier,
        )

        assert len(result) == 1
        assert len(extender.captured) == 1
        context = extender.captured[0]
        assert context.hook == ExtenderHook.JOIN
        assert context.join_type == "inner"
        assert context.join_keys == (f"{_MARKER}_left_id={_MARKER}_right_id",)
        assert context.compute_framework_name == "PythonDictFramework"
        assert context.run_id == session.run_id
        assert context.carrier == carrier
        assert context.worker_index is None


class TestNoJoinExtenderRegisteredBaselineRegressionGuard:
    """Baseline guard: a join with no JOIN extender registered is unaffected."""

    def test_inner_join_completes_and_produces_expected_merged_values(self, flight_server: Any) -> None:
        result = mloda.run_all(
            [Feature(name=_JoinHookConsumerFeatureGroup.get_class_name())],
            links={_join_hook_link()},
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert len(result) == 1
        column = result[0][_JoinHookConsumerFeatureGroup.get_class_name()]
        assert set(column) == {"a", "b", "c"}


class TestDenyBeforeMerge:
    """A raise_on_error=True (default) JOIN extender that raises instead of delegating prevents the run from completing."""

    def test_veto_raises_and_propagates(self, flight_server: Any) -> None:
        extender = _JoinVetoExtender()

        with pytest.raises(RuntimeError, match="denied join"):
            mloda.run_all(
                [Feature(name=_JoinHookConsumerFeatureGroup.get_class_name())],
                links={_join_hook_link()},
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                flight_server=flight_server,
                parallelization_modes={ParallelizationMode.SYNC},
                function_extender={extender},
            )


class TestDenyWithFallback:
    """A raise_on_error=False JOIN extender that raises still lets the join complete, with a warning logged."""

    def test_warning_only_veto_logs_and_falls_back(self, flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
        extender = _JoinVetoExtender(raise_on_error=False)

        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [Feature(name=_JoinHookConsumerFeatureGroup.get_class_name())],
                links={_join_hook_link()},
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                flight_server=flight_server,
                parallelization_modes={ParallelizationMode.SYNC},
                function_extender={extender},
            )

        assert len(result) == 1
        column = result[0][_JoinHookConsumerFeatureGroup.get_class_name()]
        assert set(column) == {"a", "b", "c"}
        assert any(record.levelno == logging.WARNING and "denied join" in record.message for record in caplog.records)


# === Star topology (hub + two spokes): JOIN fires once per merge ===


class _JoinHookStarHubFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_star_row_id", f"{_MARKER}_star_hub_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_star_row_id": [1, 2, 3], f"{_MARKER}_star_hub_value": ["h1", "h2", "h3"]}


class _JoinHookStarSpokeAFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_star_row_id", f"{_MARKER}_star_spoke_a_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_star_row_id": [1, 2, 3], f"{_MARKER}_star_spoke_a_value": [10, 20, 30]}


class _JoinHookStarSpokeBFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_star_row_id", f"{_MARKER}_star_spoke_b_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_star_row_id": [1, 2, 3], f"{_MARKER}_star_spoke_b_value": ["x", "y", "z"]}


class _JoinHookStarConsumerFeatureGroup(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name=f"{_MARKER}_star_hub_value"),
            Feature(name=f"{_MARKER}_star_spoke_a_value"),
            Feature(name=f"{_MARKER}_star_spoke_b_value"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data[f"{_MARKER}_star_hub_value"]}


class TestJoinHookFiresOncePerMergeInStarTopology:
    def test_three_way_star_join_fires_the_hook_twice(self, flight_server: Any) -> None:
        extender = _JoinListCapturingExtender()
        links = Link.star(
            _JoinHookStarHubFeatureGroup,
            _JoinHookStarSpokeAFeatureGroup,
            _JoinHookStarSpokeBFeatureGroup,
            index_column=f"{_MARKER}_star_row_id",
        )
        enabled = PluginCollector.enabled_feature_groups(
            {
                _JoinHookStarHubFeatureGroup,
                _JoinHookStarSpokeAFeatureGroup,
                _JoinHookStarSpokeBFeatureGroup,
                _JoinHookStarConsumerFeatureGroup,
            }
        )

        result = mloda.run_all(
            [Feature(name=_JoinHookStarConsumerFeatureGroup.get_class_name())],
            links=links,
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=enabled,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert len(result) == 1
        assert len(extender.captured) == 2
        assert all(context.hook == ExtenderHook.JOIN for context in extender.captured)


# === APPEND join type: join_keys is None, since APPEND does not use keys to merge ===


class _JoinHookAppendLeftFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_append_left_id", f"{_MARKER}_append_left_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_append_left_id": [1, 2], f"{_MARKER}_append_left_value": ["a", "b"]}


class _JoinHookAppendRightFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({f"{_MARKER}_append_right_id", f"{_MARKER}_append_right_value"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_append_right_id": [3, 4], f"{_MARKER}_append_right_value": ["c", "d"]}


class _JoinHookAppendConsumerFeatureGroup(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        link = Link.append(
            JoinSpec(_JoinHookAppendLeftFeatureGroup, Index((f"{_MARKER}_append_left_id",))),
            JoinSpec(_JoinHookAppendRightFeatureGroup, Index((f"{_MARKER}_append_right_id",))),
        )
        return {
            Feature(name=f"{_MARKER}_append_left_value", link=link, index=Index((f"{_MARKER}_append_left_id",))),
            Feature(name=f"{_MARKER}_append_right_value", index=Index((f"{_MARKER}_append_right_id",))),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data[f"{_MARKER}_append_left_value"]}


class TestJoinKeysIsNoneForAppendJoinType:
    """Fix: APPEND/UNION links don't use join keys to merge, so join_keys must be None,
    not a misleading key tuple."""

    def test_append_link_reports_join_keys_as_none(self, flight_server: Any) -> None:
        extender = _JoinListCapturingExtender()
        enabled = PluginCollector.enabled_feature_groups(
            {_JoinHookAppendLeftFeatureGroup, _JoinHookAppendRightFeatureGroup, _JoinHookAppendConsumerFeatureGroup}
        )

        result = mloda.run_all(
            [Feature(name=_JoinHookAppendConsumerFeatureGroup.get_class_name())],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=enabled,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
        )

        assert len(result) == 1
        assert len(extender.captured) == 1
        assert extender.captured[0].join_type == "append"
        assert extender.captured[0].join_keys is None
