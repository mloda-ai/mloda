"""Tests wiring ExtenderHook.JOIN into JoinStep._merge_data: HookContext population, the
no-extender baseline, and deny-before-merge / deny-with-fallback. For a keyed join type,
join_keys pairs each left column with its corresponding right column as ``"left=right"``
strings; this is independent of ``swap_merge_sides`` since it reads off the immutable
``Link``, not whichever cfw is "self" at merge time. APPEND/UNION links merge without using
keys, so their join_keys is None. The last sections pin ordering of joins that share a destination frame.
"""

import logging
import threading
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.core.core.step.join_step import JoinStep
from mloda.provider import ComputeFramework, GeneratedFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_MARKER = "joinhook051"


class _JoinHookLeftFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_left_id", f"{_MARKER}_left_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_left_id": [1, 2, 3], f"{_MARKER}_left_value": ["a", "b", "c"]}


class _JoinHookRightFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_right_id", f"{_MARKER}_right_value"}

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
            function_extender={extender},
        )

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            result = session.run(
                parallelization_modes={ParallelizationMode.SYNC},
                flight_server=flight_server,
                carrier=carrier,
            )

        assert len(result) == 1
        assert len(extender.captured) == 1
        context = extender.captured[0]
        assert context.hook == ExtenderHook.JOIN
        assert context.join_type == "inner"
        assert context.join_keys == (f"{_MARKER}_left_id={_MARKER}_right_id",)
        assert context.compute_framework_name == "PythonDictFramework"
        assert context.run_id is not None
        assert context.plan_id == session.plan_id
        assert context.structure_hash is not None
        assert context.structure_hash == session.plan_context.structure_hash
        assert context.run_id != session.plan_id
        assert context.carrier == carrier
        assert context.worker_index is None
        assert context.tenant_id == "acme"
        assert context.project_id == "proj1"
        assert context.principal == "hash123"
        assert context.asof_config is None
        assert context.feature_group_class is None
        assert context.feature_group_version is None
        assert context.join_left_feature_group == _fg_path(_JoinHookLeftFeatureGroup)
        assert context.join_right_feature_group == _fg_path(_JoinHookRightFeatureGroup)


def _fg_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _build_direct_join_step(link: Link | None = None, swap_merge_sides: bool = False) -> JoinStep:
    """A JoinStep usable for direct _merge_data calls, bypassing DAG execution entirely."""
    return JoinStep(
        link=link if link is not None else _join_hook_link(),
        swap_merge_sides=swap_merge_sides,
        destination_framework=PythonDictFramework,
        source_framework=PythonDictFramework,
        required_uuids=set(),
        destination_framework_uuids=set(),
        source_framework_uuids=set(),
    )


class TestJoinHookCarrierIsNotAliasedAcrossTwoMergesOnSameComputeFramework:
    """Two JOIN HookContexts built from merges into the SAME destination ComputeFramework
    instance (mirroring a star join's hub, which merges twice) must not share the carrier
    dict object read off cfw.run_context.carrier."""

    def test_two_direct_merge_calls_get_distinct_carrier_objects(self) -> None:
        extender = _JoinListCapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        step = _build_direct_join_step()
        cfw = PythonDictFramework(function_extender={extender})
        cfw.run_context = RunContext(carrier=carrier)
        cfw.data = {f"{_MARKER}_left_id": [1, 2, 3], f"{_MARKER}_left_value": ["a", "b", "c"]}
        from_cfw_data = {f"{_MARKER}_right_id": [1, 2, 3], f"{_MARKER}_right_value": [10, 20, 30]}
        second_from_cfw_data = {f"{_MARKER}_right_id": [1, 2, 3], f"{_MARKER}_other_value": [7, 8, 9]}

        step._merge_data(cfw, from_cfw_data)
        step._merge_data(cfw, second_from_cfw_data)

        assert len(extender.captured) == 2
        first_context, second_context = extender.captured
        assert first_context.carrier == second_context.carrier == carrier
        assert first_context.carrier is not second_context.carrier
        assert first_context.carrier is not cfw.run_context.carrier
        assert second_context.carrier is not cfw.run_context.carrier

    def test_mutating_one_carrier_does_not_leak_into_the_other_or_run_context(self) -> None:
        extender = _JoinListCapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        step = _build_direct_join_step()
        cfw = PythonDictFramework(function_extender={extender})
        cfw.run_context = RunContext(carrier=carrier)
        cfw.data = {f"{_MARKER}_left_id": [1, 2, 3], f"{_MARKER}_left_value": ["a", "b", "c"]}
        from_cfw_data = {f"{_MARKER}_right_id": [1, 2, 3], f"{_MARKER}_right_value": [10, 20, 30]}
        second_from_cfw_data = {f"{_MARKER}_right_id": [1, 2, 3], f"{_MARKER}_other_value": [7, 8, 9]}

        step._merge_data(cfw, from_cfw_data)
        step._merge_data(cfw, second_from_cfw_data)

        assert len(extender.captured) == 2
        first_context, second_context = extender.captured
        assert first_context.carrier is not None
        with pytest.raises(TypeError):
            first_context.carrier["mutated"] = "yes"

        assert second_context.carrier is not None
        assert "mutated" not in second_context.carrier
        assert cfw.run_context.carrier is not None
        assert "mutated" not in cfw.run_context.carrier


class TestJoinHookFeatureGroupOrientationFollowsLink:
    @pytest.mark.parametrize("swap_merge_sides", [False, True])
    def test_declared_orientation_holds_regardless_of_swap(self, swap_merge_sides: bool) -> None:
        extender = _JoinListCapturingExtender()
        step = _build_direct_join_step(swap_merge_sides=swap_merge_sides)
        cfw = PythonDictFramework(function_extender={extender})
        cfw.run_context = RunContext()
        left = {f"{_MARKER}_left_id": [1, 2, 3], f"{_MARKER}_left_value": ["a", "b", "c"]}
        right = {f"{_MARKER}_right_id": [1, 2, 3], f"{_MARKER}_right_value": [10, 20, 30]}
        cfw.data, from_cfw_data = (right, left) if swap_merge_sides else (left, right)

        step._merge_data(cfw, from_cfw_data)

        assert len(extender.captured) == 1
        context = extender.captured[0]
        assert context.join_left_feature_group == _fg_path(_JoinHookLeftFeatureGroup)
        assert context.join_right_feature_group == _fg_path(_JoinHookRightFeatureGroup)


class TestJoinHookCarriesAsofConfig:
    def test_asof_link_exposes_asof_config_and_type_on_context(self) -> None:
        extender = _JoinListCapturingExtender()
        link = Link.asof(
            JoinSpec(_JoinHookLeftFeatureGroup, Index((f"{_MARKER}_left_id",))),
            JoinSpec(_JoinHookRightFeatureGroup, Index((f"{_MARKER}_right_id",))),
            left_time_column=f"{_MARKER}_left_ts",
            right_time_column=f"{_MARKER}_right_ts",
            direction="forward",
            allow_exact_matches=False,
        )
        step = _build_direct_join_step(link)
        cfw = PythonDictFramework(function_extender={extender})
        cfw.run_context = RunContext()
        cfw.data = {
            f"{_MARKER}_left_id": [1, 2, 3],
            f"{_MARKER}_left_ts": [10, 20, 30],
            f"{_MARKER}_left_value": ["a", "b", "c"],
        }
        from_cfw_data = {
            f"{_MARKER}_right_id": [1, 2, 3],
            f"{_MARKER}_right_ts": [11, 21, 31],
            f"{_MARKER}_right_value": [10, 20, 30],
        }

        step._merge_data(cfw, from_cfw_data)

        assert len(extender.captured) == 1
        context = extender.captured[0]
        assert context.join_type == "asof"
        assert link.asof_config is not None
        assert context.asof_config == link.asof_config


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


class _JoinHookStarHubFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_star_row_id", f"{_MARKER}_star_hub_value", f"{_MARKER}_star_hub_other"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            f"{_MARKER}_star_row_id": [1, 2, 3],
            f"{_MARKER}_star_hub_value": ["h1", "h2", "h3"],
            f"{_MARKER}_star_hub_other": ["g1", "g2", "g3"],
        }


class _JoinHookStarSpokeAFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_star_row_id", f"{_MARKER}_star_spoke_a_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_star_row_id": [1, 2, 3], f"{_MARKER}_star_spoke_a_value": [10, 20, 30]}


class _JoinHookStarSpokeBFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_star_row_id", f"{_MARKER}_star_spoke_b_value"}

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


class _JoinHookAppendLeftFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_append_left_id", f"{_MARKER}_append_left_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_MARKER}_append_left_id": [1, 2], f"{_MARKER}_append_left_value": ["a", "b"]}


class _JoinHookAppendRightFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_MARKER}_append_right_id", f"{_MARKER}_append_right_value"}

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


# === Join ordering: joins of different links writing one destination frame are ordered ===


def _star_plugins(*consumers: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(
        {_JoinHookStarHubFeatureGroup, _JoinHookStarSpokeAFeatureGroup, _JoinHookStarSpokeBFeatureGroup, *consumers}
    )


class _JoinOverlapExtender(Extender):
    """Records the highest number of JOIN merges in flight at once."""

    def __init__(self) -> None:
        self.priority = 100
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.JOIN}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return func(*args, **kwargs)
        finally:
            with self.lock:
                self.active -= 1


class _JoinHookStarRowsConsumerFeatureGroup(FeatureGroup):
    """Packs the hub and both spoke values of each row into one string."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        index = Index((f"{_MARKER}_star_row_id",))
        return {
            Feature(name=f"{_MARKER}_star_hub_value", index=index),
            Feature(name=f"{_MARKER}_star_spoke_a_value", index=index),
            Feature(name=f"{_MARKER}_star_spoke_b_value", index=index),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        hub, a, b = (data[f"{_MARKER}_star_{name}_value"] for name in ("hub", "spoke_a", "spoke_b"))
        return {cls.get_class_name(): [f"{x}|{y}|{z}" for x, y, z in zip(hub, a, b)]}


class _JoinHookStarPairConsumerFeatureGroup(FeatureGroup):
    """Reads one hub column and one spoke column per row."""

    hub_column: ClassVar[str]
    spoke_column: ClassVar[str]

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        index = Index((f"{_MARKER}_star_row_id",))
        return {Feature(name=self.hub_column, index=index), Feature(name=self.spoke_column, index=index)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [f"{x}|{y}" for x, y in zip(data[cls.hub_column], data[cls.spoke_column])]}


class _JoinHookStarHubValueWithSpokeAFeatureGroup(_JoinHookStarPairConsumerFeatureGroup):
    hub_column = f"{_MARKER}_star_hub_value"
    spoke_column = f"{_MARKER}_star_spoke_a_value"


class _JoinHookStarHubOtherWithSpokeBFeatureGroup(_JoinHookStarPairConsumerFeatureGroup):
    hub_column = f"{_MARKER}_star_hub_other"
    spoke_column = f"{_MARKER}_star_spoke_b_value"


def _ordering_edges(session: Any) -> int:
    """Count join-to-join waits; fails unless the plan holds exactly two JoinSteps."""
    assert session.engine is not None
    joins = [step for step in session.engine.execution_planner if isinstance(step, JoinStep)]
    assert len(joins) == 2
    return sum(1 for a in joins for b in joins if a is not b and a.uuid in b.required_uuids)


def _prepare_star(links: set[Link], *consumers: type[FeatureGroup], extender: Extender | None = None) -> Any:
    return mloda.prepare(
        [Feature(name=consumer.get_class_name()) for consumer in consumers],
        links=links,
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_star_plugins(*consumers),
        parallelization_modes={ParallelizationMode.THREADING},
        function_extender={extender} if extender else None,
    )


class TestJoinsOfDifferentLinksSharingADestinationAreOrdered:
    def test_two_right_links_sharing_the_right_frame_are_ordered_and_never_merge_concurrently(self) -> None:
        index = Index((f"{_MARKER}_star_row_id",))
        links = {
            Link.right(JoinSpec(spoke, index), JoinSpec(_JoinHookStarHubFeatureGroup, index))
            for spoke in (_JoinHookStarSpokeAFeatureGroup, _JoinHookStarSpokeBFeatureGroup)
        }
        extender = _JoinOverlapExtender()
        consumer = _JoinHookStarRowsConsumerFeatureGroup

        session = _prepare_star(links, consumer, extender=extender)
        result = session.run(parallelization_modes={ParallelizationMode.THREADING})

        assert _ordering_edges(session) == 1
        assert extender.calls == 2
        assert extender.max_active == 1
        rows = [row for frame in result for row in frame.get(consumer.get_class_name(), [])]
        assert sorted(rows) == ["h1|10|x", "h2|20|y", "h3|30|z"]

    @pytest.mark.parametrize("join_type", ["append", "union"])
    def test_two_row_stacking_links_sharing_the_left_frame_are_ordered_and_keep_every_row(self, join_type: str) -> None:
        index = Index((f"{_MARKER}_star_row_id",))
        make = getattr(Link, join_type)
        links = {
            make(JoinSpec(_JoinHookStarHubFeatureGroup, index), JoinSpec(spoke, index))
            for spoke in (_JoinHookStarSpokeAFeatureGroup, _JoinHookStarSpokeBFeatureGroup)
        }
        extender = _JoinOverlapExtender()
        consumer = _JoinHookStarRowsConsumerFeatureGroup

        session = _prepare_star(links, consumer, extender=extender)
        result = session.run(parallelization_modes={ParallelizationMode.THREADING})

        assert _ordering_edges(session) == 1
        assert extender.calls == 2
        assert extender.max_active == 1
        rows = [row for frame in result for row in frame.get(consumer.get_class_name(), [])]
        assert len(rows) == 9
        for value in ("h1", "h3", "10", "30", "x", "z"):
            assert any(value in row.split("|") for row in rows), f"rows of {value} were lost: {rows}"

    def test_two_links_joining_different_hub_columns_into_one_hub_frame_are_ordered(self) -> None:
        index = Index((f"{_MARKER}_star_row_id",))
        links = {
            Link.inner(JoinSpec(_JoinHookStarHubFeatureGroup, index), JoinSpec(spoke, index))
            for spoke in (_JoinHookStarSpokeAFeatureGroup, _JoinHookStarSpokeBFeatureGroup)
        }
        extender = _JoinOverlapExtender()
        first = _JoinHookStarHubValueWithSpokeAFeatureGroup
        second = _JoinHookStarHubOtherWithSpokeBFeatureGroup

        session = _prepare_star(links, first, second, extender=extender)
        result = session.run(parallelization_modes={ParallelizationMode.THREADING})

        assert _ordering_edges(session) == 1
        assert extender.calls == 2
        assert extender.max_active == 1
        for consumer, expected in ((first, ["h1|10", "h2|20", "h3|30"]), (second, ["g1|x", "g2|y", "g3|z"])):
            rows = [row for frame in result for row in frame.get(consumer.get_class_name(), [])]
            assert sorted(rows) == expected

    def test_an_append_chain_plans_and_keeps_every_row(self) -> None:
        index = Index((f"{_MARKER}_star_row_id",))
        links = {
            Link.append(
                JoinSpec(_JoinHookStarHubFeatureGroup, index), JoinSpec(_JoinHookStarSpokeAFeatureGroup, index)
            ),
            Link.append(
                JoinSpec(_JoinHookStarSpokeAFeatureGroup, index), JoinSpec(_JoinHookStarSpokeBFeatureGroup, index)
            ),
        }
        consumer = _JoinHookStarRowsConsumerFeatureGroup

        session = _prepare_star(links, consumer)
        result = session.run(parallelization_modes={ParallelizationMode.THREADING})

        rows = [row for frame in result for row in frame.get(consumer.get_class_name(), [])]
        assert len(rows) == 9


# === A source derived from an already joined group must not turn the ordering into a step cycle ===

_ORDER_KEY = f"{_MARKER}_order_key"


class _JoinOrderRootFeatureGroup(GeneratedFeatureGroup):
    keys: ClassVar[list[int]] = [1, 2]
    value_prefix: ClassVar[str] = ""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {_ORDER_KEY: list(cls.keys), cls.get_class_name(): [f"{cls.value_prefix}{k}" for k in cls.keys]}


class _JoinOrderOrdersFeatureGroup(_JoinOrderRootFeatureGroup):
    value_prefix = "o"


class _JoinOrderCustomersFeatureGroup(_JoinOrderRootFeatureGroup):
    value_prefix = "c"


class _JoinOrderScoreFeatureGroup(FeatureGroup):
    """Derived from the Orders and Customers join; returns the input frame plus its own column."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name=_JoinOrderOrdersFeatureGroup.get_class_name()),
            Feature(name=_JoinOrderCustomersFeatureGroup.get_class_name()),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        orders = data[_JoinOrderOrdersFeatureGroup.get_class_name()]
        customers = data[_JoinOrderCustomersFeatureGroup.get_class_name()]
        data = dict(data)
        data[cls.get_class_name()] = [f"{o}|{c}" for o, c in zip(orders, customers)]
        return data


class _JoinOrderReportFeatureGroup(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name=_JoinOrderOrdersFeatureGroup.get_class_name()),
            Feature(name=_JoinOrderScoreFeatureGroup.get_class_name()),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        orders = data[_JoinOrderOrdersFeatureGroup.get_class_name()]
        scores = data[_JoinOrderScoreFeatureGroup.get_class_name()]
        return {cls.get_class_name(): [f"{o}:{s}" for o, s in zip(orders, scores)]}


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
def test_a_source_derived_from_an_already_joined_group_plans_without_a_step_cycle(mode: ParallelizationMode) -> None:
    index = Index((_ORDER_KEY,))
    links = {
        Link.inner(JoinSpec(_JoinOrderOrdersFeatureGroup, index), JoinSpec(_JoinOrderCustomersFeatureGroup, index)),
        Link.inner(JoinSpec(_JoinOrderOrdersFeatureGroup, index), JoinSpec(_JoinOrderScoreFeatureGroup, index)),
    }
    enabled = PluginCollector.enabled_feature_groups(
        {
            _JoinOrderOrdersFeatureGroup,
            _JoinOrderCustomersFeatureGroup,
            _JoinOrderScoreFeatureGroup,
            _JoinOrderReportFeatureGroup,
        }
    )

    result = mloda.run_all(
        [Feature(name=_JoinOrderReportFeatureGroup.get_class_name())],
        links=links,
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=enabled,
        parallelization_modes={mode},
    )

    rows = [row for frame in result for row in frame.get(_JoinOrderReportFeatureGroup.get_class_name(), [])]
    assert sorted(rows) == ["o1:o1|c1", "o2:o2|c2"]
