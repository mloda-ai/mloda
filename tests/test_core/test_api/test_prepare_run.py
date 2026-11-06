"""
Tests for the prepare() / run() public API on mlodaAPI.

These tests define the contract for a two-phase execution model:
  1. prepare() - classmethod that builds the execution plan and returns a reusable mlodaAPI instance
  2. run() - instance method that executes with fresh api_data, reusing the cached plan
"""

import dataclasses
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from unittest.mock import patch
from typing import Any

import numpy as np
import pytest

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.api.plan_lock import _lock_text, plan_structure_hash
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import mloda, mlodaAPI, Feature, ParallelizationMode, PluginCollector
from mloda.provider import (
    ApiDataFG,
    ComputeFramework,
    FeatureGroup,
    FeatureSet,
    DataCreatorFeatureGroup,
    PropertySpec,
    property_spec,
)
from tests.test_core.test_abstract_plugins.name_bound_fixtures import make_name_bound_fg, make_root_fg
from mloda.steward import RunSurface
from mloda.user import Options, FeatureName, Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_plugins.api.test_simplified_api_data import (
    ApiNeutralBoundedFG,
    ApiNeutralGenericFG,
    ApiNeutralPandasOnlyFG,
)


CTX_DEFAULT = "prskip_ctx_val"
CTX_KEY = "prskip_ctx_default"


def _make_boundary_probe_fg(feature_name: str) -> type[FeatureGroup]:
    """Root FeatureGroup whose calculate_feature reads feature.options directly (no options_with_defaults)."""

    class PrSkipBoundaryProbeFeatureGroup(DataCreatorFeatureGroup):
        PROPERTY_MAPPING = {CTX_KEY: PropertySpec("A context concrete default.", context=True, default=CTX_DEFAULT)}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {feature_name}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(f.name): [{"ctx": f.options.get(CTX_KEY)}] for f in features.features}

    return PrSkipBoundaryProbeFeatureGroup


def _single_row(frame: Any, column: str) -> Any:
    values = list(frame[column]) if isinstance(frame, dict) else [row[column] for row in frame]
    assert len(values) == 1, f"expected exactly one row for {column}, got {values!r}"
    return values[0]


class PrepareRunApiFeature(FeatureGroup):
    """A simple feature that consumes api data for prepare/run tests."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="api_id", index=Index(("api_id",))),
            Feature(name="api_value", index=Index(("api_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PrepareRunApiFeature"] = data["api_id"].astype(str) + "_" + data["api_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_enabled = PluginCollector.enabled_feature_groups(
    {
        ApiDataFG,
        PrepareRunApiFeature,
    }
)


class TestPrepareReturnsInstance:
    """Test 1: prepare() returns an mlodaAPI instance."""

    def test_prepare_returns_mloda_api_instance(self) -> None:
        """Calling mlodaAPI.prepare() should return an mlodaAPI instance with a cached execution plan."""
        api_data = {
            "PrepareExample": {
                "api_id": [1, 2, 3],
                "api_value": ["a", "b", "c"],
            }
        }

        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
            plugin_collector=_enabled,
        )

        assert isinstance(session, mlodaAPI)


class TestRunReturnsResults:
    """Test 2: run() returns a list of results."""

    def test_run_returns_results(self) -> None:
        """After prepare(), calling run() should execute the plan and return a list of results."""
        api_data = {
            "PrepareExample": {
                "api_id": [1, 2, 3],
                "api_value": ["a", "b", "c"],
            }
        }

        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
            plugin_collector=_enabled,
        )

        result = session.run(api_data=api_data)

        assert isinstance(result, list)
        assert len(result) == 1
        df = result[0]
        assert "PrepareRunApiFeature" in df.columns
        assert len(df) == 3
        assert df["PrepareRunApiFeature"].tolist() == ["1_a", "2_b", "3_c"]


class TestRunMatchesRunAllOutput:
    """Test 3: run() output matches run_all() output."""

    def test_run_matches_run_all_output(self) -> None:
        """The prepare()+run() path must produce the same results as run_all()."""
        api_data = {
            "PrepareExample": {
                "api_id": [10, 20],
                "api_value": ["x", "y"],
            }
        }

        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]

        run_all_result = mloda.run_all(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
            plugin_collector=_enabled,
        )

        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
            plugin_collector=_enabled,
        )
        prepare_run_result = session.run(api_data=api_data)

        assert len(run_all_result) == len(prepare_run_result)

        df_run_all = run_all_result[0]
        df_prepare_run = prepare_run_result[0]

        assert df_run_all["PrepareRunApiFeature"].tolist() == df_prepare_run["PrepareRunApiFeature"].tolist()


class TestMultipleSequentialRuns:
    """Test 4: Multiple sequential runs with different api_data produce correct, independent results."""

    def test_multiple_sequential_runs_with_different_api_data(self) -> None:
        """prepare() once, then run() twice with different api_data.
        Each run should produce results matching only its own input data.
        """
        initial_api_data = {
            "PrepareExample": {
                "api_id": [1],
                "api_value": ["initial"],
            }
        }

        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=initial_api_data,
            plugin_collector=_enabled,
        )

        first_api_data = {
            "PrepareExample": {
                "api_id": [1, 2],
                "api_value": ["a", "b"],
            }
        }
        first_result = session.run(api_data=first_api_data)

        assert len(first_result) == 1
        df_first = first_result[0]
        assert df_first["PrepareRunApiFeature"].tolist() == ["1_a", "2_b"]

        second_api_data = {
            "PrepareExample": {
                "api_id": [10, 20, 30],
                "api_value": ["x", "y", "z"],
            }
        }
        second_result = session.run(api_data=second_api_data)

        assert len(second_result) == 1
        df_second = second_result[0]
        assert df_second["PrepareRunApiFeature"].tolist() == ["10_x", "20_y", "30_z"]


class _RefuseSecondRunStartExtender(Extender):
    """Raises from on_run_start on the second run only."""

    raise_on_error = True

    def __init__(self) -> None:
        self.starts = 0

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        self.starts += 1
        if self.starts == 2:
            raise RuntimeError("refuse second run")


def _rerun(session: Any, path: str, api_data: dict[str, Any]) -> list[Any]:
    if path == "batch":
        return list(session.run(api_data=api_data))
    return list(session.stream_run(api_data=api_data))


class TestFailedRerunDoesNotExposeStaleResults:
    """A failed re-run raises, and the next successful run returns only its own fresh data."""

    _first = {"PrepareExample": {"api_id": [1, 2], "api_value": ["a", "b"]}}
    _fresh = {"PrepareExample": {"api_id": [7, 8, 9], "api_value": ["p", "q", "r"]}}

    def _prepare(self, function_extender: Any = None) -> Any:
        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]
        return mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data={"PrepareExample": {"api_id": [1], "api_value": ["initial"]}},
            plugin_collector=_enabled,
            function_extender=function_extender,
        )

    @staticmethod
    def _assert_fresh(frames: list[Any]) -> None:
        assert len(frames) == 1
        assert frames[0]["PrepareRunApiFeature"].tolist() == ["7_p", "8_q", "9_r"]

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_rerun_failing_in_computation_then_fresh_run(self, path: str) -> None:
        session = self._prepare()
        first = _rerun(session, path, self._first)
        assert first[0]["PrepareRunApiFeature"].tolist() == ["1_a", "2_b"]

        bad = {"PrepareExample": {"api_id": [1, 2], "api_value": [3, 4]}}
        with pytest.raises(TypeError) as raised:
            _rerun(session, path, bad)
        assert getattr(raised.value, "__notes__", None) is None

        self._assert_fresh(_rerun(session, path, self._fresh))
        assert not hasattr(session, "runner")

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_rerun_refused_at_start_then_fresh_run(self, path: str) -> None:
        session = self._prepare(function_extender={_RefuseSecondRunStartExtender()})
        _rerun(session, path, self._first)

        with pytest.raises(RuntimeError, match="refuse second run"):
            _rerun(session, path, self._first)

        self._assert_fresh(_rerun(session, path, self._fresh))
        assert not hasattr(session, "runner")


class TestSessionKeepsNoPerRunState:
    """The prepared session holds no orchestrator and RunResult carries none."""

    def test_removed_session_api_and_result_holds_no_orchestrator(self) -> None:
        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]
        data = {"PrepareExample": {"api_id": [1], "api_value": ["a"]}}
        session = mloda.prepare(
            features, compute_frameworks=[PandasDataFrame], api_data=data, plugin_collector=_enabled
        )

        session._connection_handles[0] = "handle"

        result = session.run(api_data=data)

        for name in ("runner", "get_result", "get_artifacts"):
            assert not hasattr(session, name)
        assert not hasattr(result, "runner")
        assert not any(isinstance(value, ExecutionOrchestrator) for value in vars(result).values())
        assert not any(isinstance(value, ExecutionOrchestrator) for value in vars(session).values())
        assert result._connection_handles is not session._connection_handles


class TestConcurrentRunsOnOneSession:
    """One prepared session serves concurrent runs without sharing per-run state."""

    @staticmethod
    def _data(i: int) -> dict[str, Any]:
        return {"PrepareExample": {"api_id": [i, i + 100], "api_value": [f"v{i}", f"w{i}"]}}

    @staticmethod
    def _plan_state(session: Any) -> list[tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[Any]]]:
        """Per step: vars, FeatureSet vars, Feature vars, and (group, context) copies of every Options."""
        state = []
        for step in session.engine.execution_planner:
            fs = getattr(step, "features", None)
            features = list(fs.features) if fs is not None else []
            options = [(f.options.group, f.options.context) for f in features]
            state.append(
                (
                    step,
                    dict(vars(step)),
                    dict(vars(fs)) if fs is not None else {},
                    [dict(vars(f)) for f in features],
                    [(f.options, dict(g), dict(c)) for f, (g, c) in zip(features, options)],
                )
            )
        return state

    @pytest.mark.parametrize("path", ["batch", "stream"])
    @pytest.mark.parametrize(
        "modes", [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}], ids=["sync", "threading"]
    )
    def test_threads_get_their_own_results_and_session_is_unchanged(
        self, path: str, modes: set[ParallelizationMode]
    ) -> None:
        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]
        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=self._data(0),
            plugin_collector=_enabled,
            parallelization_modes=modes,
        )
        expected = {
            i: [f["PrepareRunApiFeature"].tolist() for f in _rerun(session, path, self._data(i))] for i in range(8)
        }

        session_before = dict(vars(session))
        engine_before = dict(vars(session.engine))
        plan_before = session.resolved_plan()
        plan_state_before = self._plan_state(session)
        for step, step_vars, *_ in plan_state_before:
            assert "step_is_done" not in step_vars and "location" not in step_vars
        barrier = threading.Barrier(8, timeout=5)

        def worker(i: int) -> list[Any]:
            barrier.wait()
            return [f["PrepareRunApiFeature"].tolist() for f in _rerun(session, path, self._data(i))]

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(8)]
            got = {i: future.result(timeout=8) for i, future in enumerate(futures)}

        assert got == expected
        for before, after in ((session_before, vars(session)), (engine_before, vars(session.engine))):
            assert after.keys() == before.keys()
            assert all(after[key] is before[key] for key in before)
        assert session.resolved_plan() == plan_before
        plan_state_after = self._plan_state(session)
        assert len(plan_state_after) == len(plan_state_before)
        for step_before, step_after in zip(plan_state_before, plan_state_after):
            assert step_before[0] is step_after[0]
            for vars_before, vars_after in ((step_before[1], step_after[1]), (step_before[2], step_after[2])):
                assert vars_after.keys() == vars_before.keys()
                assert all(vars_after[key] is vars_before[key] for key in vars_before)
            for feature_before, feature_after in zip(step_before[3], step_after[3]):
                assert feature_after.keys() == feature_before.keys()
                assert all(feature_after[key] is feature_before[key] for key in feature_before)
            for (_, group, context), (_, group_before, context_before) in zip(step_after[4], step_before[4]):
                assert group == group_before
                assert context == context_before


class _StartCountingExtender(Extender):
    """Counts on_run_start calls and keeps the RunContexts it sees."""

    def __init__(self) -> None:
        self.starts = 0
        self.runs: list[Any] = []

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        self.starts += 1
        self.runs.append(run)


class TestRunRefusesApiDataThatDiffersFromPrepare:
    """run()/stream_run() refuse api_data whose sets differ from prepare() before the run starts."""

    _prepared = {"PrepareExample": {"api_id": [1], "api_value": ["initial"]}}

    def _prepare(self, extender: Any) -> Any:
        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]
        return mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=self._prepared,
            plugin_collector=_enabled,
            function_extender={extender},
        )

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_missing_set_key_is_refused(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)

        with pytest.raises(ValueError, match="PrepareExample"):
            _rerun(session, path, {"OtherKey": {"api_id": [1], "api_value": ["a"]}})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_different_column_names_are_refused(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)

        with pytest.raises(ValueError, match=r"(?s)PrepareExample.*api_other"):
            _rerun(session, path, {"PrepareExample": {"api_id": [1], "api_other": ["a"]}})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_api_data_for_a_plan_prepared_without_it_is_refused(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = mloda.prepare(
            [Feature(f"{_PFEXT_MARKER}_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_pfext_enabled,
            function_extender={extender},
        )

        with pytest.raises(ValueError, match="PrepareExample"):
            _rerun(session, path, {"PrepareExample": {"api_id": [1]}})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_column_order_alone_is_accepted(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)

        _rerun(session, path, {"PrepareExample": {"api_value": ["a"], "api_id": [1]}})

        assert extender.starts == 1

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_empty_run_data_is_refused_when_the_plan_has_sets(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)

        with pytest.raises(ValueError, match="PrepareExample"):
            _rerun(session, path, {})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_non_mapping_set_value_is_refused_without_leaking_it(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)
        secret = "prepare_run_secret_value_q7"

        with pytest.raises(ValueError, match="PrepareExample") as raised:
            _rerun(session, path, {"PrepareExample": secret})

        assert secret not in str(raised.value)
        assert "str" in str(raised.value)
        assert extender.starts == 0


class TestRunRefusesShapeThatDiffersFromDefaultPrepare:
    """A default-prepared session records dict-of-list sets and refuses other shapes before the run starts."""

    _lists = {"NeutralSet": {"api_neutral_id": [1], "api_neutral_val": ["initial"]}}

    def _prepare(self, extender: Any, compute_frameworks: Any = None, group: Any = ApiNeutralBoundedFG) -> Any:
        return mloda.prepare(
            ["api_neutral_id"],
            compute_frameworks=compute_frameworks,
            api_data=self._lists,
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            function_extender={extender},
        )

    @pytest.mark.parametrize("group", [ApiNeutralBoundedFG, ApiNeutralGenericFG])
    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_ndarray_column_is_refused_naming_set_and_column(self, path: str, group: Any) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender, group=group)

        with pytest.raises(ValueError, match=r"(?s)NeutralSet.*api_neutral_id"):
            _rerun(session, path, {"NeutralSet": {"api_neutral_id": np.array([1]), "api_neutral_val": ["a"]}})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_non_dict_mapping_is_refused_naming_the_set(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)

        data = MappingProxyType({"api_neutral_id": [1], "api_neutral_val": ["a"]})
        with pytest.raises(ValueError, match="NeutralSet"):
            _rerun(session, path, {"NeutralSet": data})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_dict_subclass_with_lists_is_accepted(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender)

        _rerun(session, path, {"NeutralSet": OrderedDict(api_neutral_id=[1], api_neutral_val=["a"])})

        assert extender.starts == 1

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_step_on_a_framework_not_taking_the_dict_accepts_an_ndarray_column(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender, group=ApiNeutralPandasOnlyFG)

        _rerun(session, path, {"NeutralSet": {"api_neutral_id": np.array([1]), "api_neutral_val": ["a"]}})

        assert extender.starts == 1

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_explicit_dict_framework_session_refuses_an_ndarray_column(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender, compute_frameworks=[PythonDictFramework])

        with pytest.raises(ValueError, match=r"(?s)NeutralSet.*api_neutral_id"):
            _rerun(session, path, {"NeutralSet": {"api_neutral_id": np.array([1]), "api_neutral_val": ["a"]}})

        assert extender.starts == 0

    @pytest.mark.parametrize("path", ["batch", "stream"])
    def test_explicit_list_session_accepts_a_tuple_column(self, path: str) -> None:
        extender = _StartCountingExtender()
        session = self._prepare(extender, compute_frameworks=[PandasDataFrame])

        _rerun(session, path, {"NeutralSet": {"api_neutral_id": (1,), "api_neutral_val": ("a",)}})

        assert extender.starts == 1


def _run_twice_under_spies(name_bound: bool) -> dict[str, Any]:
    """Prepare, then run twice under spies; throwaway groups never outlive this frame."""
    seen: dict[str, Any] = {}
    if name_bound:
        fg = make_name_bound_fg(
            "PrSkip",
            __name__,
            "prskipop",
            ("prskip_first", "prskip_second"),
            defaults={"prskip_first": "a", "prskip_second": "a"},
            seen=seen,
            input_column="prskip_src",
        )
        source = make_root_fg("PrSkipRoot", __name__, "prskip_src", [1, 2])
        name = "prskip_src__prskipop_q_b"
        expected: tuple[Any, ...] = ("q", "b")
    else:
        fg = _make_boundary_probe_fg("prskip_default_feature")
        source = fg
        name = "prskip_default_feature"
        expected = (CTX_DEFAULT,)
    session = mloda.prepare(
        [Feature(name)],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({fg, source}),
    )
    seen.clear()
    with (
        patch.object(FeatureGroup, "options_with_defaults", wraps=FeatureGroup.options_with_defaults) as defaults,
        patch.object(FeatureGroup, "name_bindings", wraps=FeatureGroup.name_bindings) as bindings,
        patch.object(Feature, "__hash__", autospec=True, side_effect=Feature.__hash__) as hashes,
    ):
        first = session.run()
        second = session.run()
    if not name_bound:
        seen[name] = (_single_row(second[0], name)["ctx"],)
    return {
        "defaults": defaults.call_count,
        "bindings": bindings.call_count,
        "hashes": hashes.call_count,
        "seen": seen.get(name),
        "expected": expected,
        "result_counts": (len(first), len(second)),
    }


def _replace_options_between_runs() -> list[Any]:
    """Prepare with copy_features=False, run, swap the plan feature's options for a fresh Options, run again."""
    fg = _make_boundary_probe_fg("prskip_replaced_feature")
    feature = Feature("prskip_replaced_feature")
    session = mloda.prepare(
        [feature],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({fg}),
        copy_features=False,
    )
    session.run()
    feature.options = Options()
    return list(session.run())


class TestStepStateDoesNotLeakBetweenRuns:
    """Test 5: Internal step state does not leak between runs."""

    def test_step_state_does_not_leak_between_runs(self) -> None:
        """prepare() once, run() twice with the same data.
        Both runs must succeed and produce identical results.
        If per-run step state leaked from the first run,
        the second run would fail or produce wrong results.
        """
        api_data = {
            "PrepareExample": {
                "api_id": [5, 6],
                "api_value": ["p", "q"],
            }
        }

        features: list[Feature | str] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
            plugin_collector=_enabled,
        )

        first_result = session.run(api_data=api_data)
        second_result = session.run(api_data=api_data)

        assert len(first_result) == 1
        assert len(second_result) == 1

        df_first = first_result[0]
        df_second = second_result[0]

        assert df_first["PrepareRunApiFeature"].tolist() == ["5_p", "6_q"]
        assert df_second["PrepareRunApiFeature"].tolist() == ["5_p", "6_q"]
        assert df_first["PrepareRunApiFeature"].tolist() == df_second["PrepareRunApiFeature"].tolist()

    @pytest.mark.parametrize("name_bound", [False, True], ids=["declared_defaults", "name_bound_chain"])
    def test_runs_skip_default_materialization_recompute(self, name_bound: bool) -> None:
        """After prepare(), run() neither recomputes defaults/name bindings nor hashes features."""
        observed = _run_twice_under_spies(name_bound)

        assert observed["defaults"] == 0
        assert observed["bindings"] == 0
        assert observed["hashes"] == 0
        assert observed["seen"] == observed["expected"]
        assert observed["result_counts"] == (1, 1)

    def test_run_after_options_replacement_still_materializes_defaults(self) -> None:
        """With copy_features=False, replacing a plan feature's options after prepare() must not lose defaults."""
        results = _replace_options_between_runs()
        assert _single_row(results[0], "prskip_replaced_feature")["ctx"] == CTX_DEFAULT

    def test_runs_build_no_transformer_registry(self) -> None:
        """After prepare(), runs converting dict api_data to pandas reuse the plan's one registry."""
        api_data = {"PrepareExample": {"api_id": [5, 6], "api_value": ["p", "q"]}}
        session = mloda.prepare(
            [Feature(name="PrepareRunApiFeature")],
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
            plugin_collector=_enabled,
        )

        original_init = ComputeFrameworkTransformer.__init__
        constructed: list[int] = []

        def _counting_init(self: ComputeFrameworkTransformer) -> None:
            constructed.append(1)
            original_init(self)

        with patch.object(ComputeFrameworkTransformer, "__init__", _counting_init):
            session.run(api_data=api_data)
            session.run(api_data=api_data)

        assert constructed == []


_PFEXT_MARKER = "prepfallback051"


class _PrepareRunExtenderFeatureGroup(DataCreatorFeatureGroup):
    """Simple root feature group for pinning function_extender fallback between prepare() and run()."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {f"{_PFEXT_MARKER}_col"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {f"{_PFEXT_MARKER}_col": [1, 2, 3]}


class _CalculateHookRecordingExtender(Extender):
    """Counts FEATURE_GROUP_CALCULATE_FEATURE invocations it wraps."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.call_count = 0

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return func(*args, **kwargs)


_pfext_enabled = PluginCollector.enabled_feature_groups({_PrepareRunExtenderFeatureGroup})


class TestRunUsesSessionFunctionExtender:
    """function_extender is session-level: set once at prepare(), reused by every run()."""

    def test_run_fires_the_extender_passed_to_prepare(self) -> None:
        recorder = _CalculateHookRecordingExtender()

        session = mloda.prepare(
            [Feature(f"{_PFEXT_MARKER}_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_pfext_enabled,
            function_extender={recorder},
        )
        session.run()

        assert recorder.call_count == 1


class TestRunAndStreamRunRejectFunctionExtender:
    """run()/stream_run() no longer accept function_extender; it is session-level, set only via prepare()."""

    @pytest.mark.parametrize("name", ["run", "stream_run"])
    def test_function_extender_kwarg_raises_type_error(self, name: str) -> None:
        recorder = _CalculateHookRecordingExtender()

        session = mloda.prepare(
            [Feature(f"{_PFEXT_MARKER}_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_pfext_enabled,
        )

        with pytest.raises(TypeError, match="unexpected keyword argument 'function_extender'"):
            getattr(session, name)(**{"function_extender": {recorder}})


class PrepareRunMutatingOptionsFG(FeatureGroup):
    """Writes feature.options in calculate_feature when the api data asks for it."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="api_id", index=Index(("api_id",))),
            Feature(name="api_value", index=Index(("api_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        trigger = data["api_value"].tolist()[0]
        for feature in features.features:
            if trigger == "set":
                feature.options.add_to_context("late_ctx_key", 1)
            elif trigger == "group":
                feature.options.group["late_group_key"] = 1
        data["PrepareRunMutatingOptionsFG"] = data["api_id"]
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_mutating_enabled = PluginCollector.enabled_feature_groups({ApiDataFG, PrepareRunMutatingOptionsFG})


class TestOptionWriteInCalculateFeatureIsRefused:
    """A plugin writing feature.options in calculate_feature gets a FrozenPlanError naming the key."""

    @staticmethod
    def _data(trigger: str) -> dict[str, Any]:
        return {"PrepareExample": {"api_id": [1, 2], "api_value": [trigger, trigger]}}

    @pytest.mark.parametrize(
        "modes",
        [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}, {ParallelizationMode.MULTIPROCESSING}],
        ids=["sync", "threading", "multiprocessing"],
    )
    @pytest.mark.parametrize("trigger,key", [("set", "late_ctx_key"), ("group", "late_group_key")])
    def test_write_raises_then_session_runs_cleanly(
        self, modes: set[ParallelizationMode], trigger: str, key: str
    ) -> None:
        from mloda.provider import FrozenPlanError

        features: list[Feature | str] = [Feature(name="PrepareRunMutatingOptionsFG")]
        session = mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            api_data=self._data("ok"),
            plugin_collector=_mutating_enabled,
            parallelization_modes=modes,
        )

        with pytest.raises(Exception) as raised:
            session.run(api_data=self._data(trigger))

        message = str(raised.value)
        chain: list[BaseException] = []
        err: BaseException | None = raised.value
        while err is not None and err not in chain:
            chain.append(err)
            err = err.__cause__ or err.__context__
        if ParallelizationMode.MULTIPROCESSING in modes:
            assert any(isinstance(e, FrozenPlanError) for e in chain) or "FrozenPlanError" in message
        else:
            assert any(isinstance(e, FrozenPlanError) for e in chain)
        for fragment in (key, "before prepare", "run(api_data=", "run(artifacts="):
            assert fragment in message

        result = session.run(api_data=self._data("ok"))
        assert result[0]["PrepareRunMutatingOptionsFG"].tolist() == [1, 2]


def _plan_snapshot(session: Any) -> list[tuple[Any, ...]]:
    """Per planned Feature: name, hash, group and context options (copied) and its identity."""
    return [
        (str(f.name), hash(f), dict(f.options.group), dict(f.options.context), id(f))
        for step in session.engine.execution_planner
        for f in getattr(getattr(step, "features", None), "features", ())
    ]


def _plan_features(session: Any) -> list[Feature]:
    return [
        f for step in session.engine.execution_planner for f in getattr(getattr(step, "features", None), "features", ())
    ]


_CACHED_ID = Feature(name="api_id", index=Index(("api_id",)))
_CACHED_VALUE = Feature(name="api_value", index=Index(("api_id",)))


class PrepareRunCachedInputsFG(FeatureGroup):
    """input_features returns class-level cached Feature objects and one nested in an option value."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        nested = options.get("nested_source")
        return {_CACHED_ID, nested if nested is not None else _CACHED_VALUE}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PrepareRunCachedInputsFG"] = data["api_id"]
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_cached_enabled = PluginCollector.enabled_feature_groups({ApiDataFG, PrepareRunCachedInputsFG})
_API = {"PrepareExample": {"api_id": [1, 2], "api_value": ["a", "b"]}}


class TestPreparedPlanIsFrozen:
    """After prepare, every FeatureGroupStep's FeatureSet, Features and Options raise on write."""

    def test_feature_set_features_and_options_reject_writes(self) -> None:
        from mloda.core.abstract_plugins.components.plan_freeze import FrozenPlanError
        from mloda.core.core.step.feature_group_step import FeatureGroupStep

        session = mloda.prepare(
            [Feature(name="PrepareRunApiFeature", options={"k": 1})],
            compute_frameworks=[PandasDataFrame],
            api_data=_API,
            plugin_collector=_enabled,
        )

        assert session.engine is not None
        steps = [s for s in session.engine.execution_planner if isinstance(s, FeatureGroupStep)]
        assert steps
        for step in steps:
            with pytest.raises(FrozenPlanError):
                step.features.prepared = False
            with pytest.raises(FrozenPlanError):
                step.features.features.add(Feature(name="late"))
            for feature in step.features.features:
                with pytest.raises(FrozenPlanError):
                    feature.domain = None
                with pytest.raises(FrozenPlanError):
                    feature.options.add_to_context("late", 1)
                with pytest.raises(FrozenPlanError):
                    feature.options.group["late"] = 1


class TestSecondPrepareLeavesFirstPlanUnchanged:
    """Intake copies, so later prepares and caller edits never write into an earlier plan."""

    def test_reused_user_features_with_copy_features_false(self) -> None:
        user_feature = Feature(name="PrepareRunApiFeature", options={"k": 1})
        kwargs: dict[str, Any] = {
            "compute_frameworks": [PandasDataFrame],
            "api_data": _API,
            "plugin_collector": _enabled,
            "copy_features": False,
        }
        first = mloda.prepare([user_feature], **kwargs)
        before = _plan_snapshot(first)

        user_feature.options.add_to_context("later", 2)
        second = mloda.prepare([user_feature], **kwargs)

        assert _plan_snapshot(first) == before
        assert not any(f is user_feature for f in _plan_features(first) + _plan_features(second))
        assert first.run(api_data=_API)[0]["PrepareRunApiFeature"].tolist() == ["1_a", "2_b"]

    def test_cached_and_nested_input_features_are_not_shared_between_plans(self) -> None:
        nested = Feature(name="api_value", index=Index(("api_id",)))
        kwargs: dict[str, Any] = {
            "compute_frameworks": [PandasDataFrame],
            "api_data": _API,
            "plugin_collector": _cached_enabled,
        }
        first = mloda.prepare([Feature(name="PrepareRunCachedInputsFG", options={"nested_source": nested})], **kwargs)
        before = _plan_snapshot(first)
        hashes = (hash(_CACHED_ID), hash(nested))

        second = mloda.prepare([Feature(name="PrepareRunCachedInputsFG", options={"tag": 2})], **kwargs)

        assert _plan_snapshot(first) == before
        assert (hash(_CACHED_ID), hash(nested)) == hashes
        shared = {id(_CACHED_ID), id(_CACHED_VALUE), id(nested)}
        assert not shared & {id(f) for f in _plan_features(first) + _plan_features(second)}
        assert first.run(api_data=_API)[0]["PrepareRunCachedInputsFG"].tolist() == [1, 2]


class _TaggedOptions(Options):
    """Options subclass carrying an extra attribute the plan must keep."""

    def __init__(self, *args: Any, tag: str = "unset", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tag = tag


class PrepareRunSubclassOptionsFG(FeatureGroup):
    """input_features returns a Feature whose Options is a subclass with an extra attribute."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name="api_id", options=_TaggedOptions({"s": 1}, tag="kept"), index=Index(("api_id",)))}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PrepareRunSubclassOptionsFG"] = data["api_id"]
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class TestIntakeCopiesKeepOptionsSubclass:
    """The plan's copies of a Feature keep its Options subclass and the subclass attributes."""

    def test_provider_input_feature_keeps_options_subclass(self) -> None:
        session = mloda.prepare(
            [Feature(name="PrepareRunSubclassOptionsFG")],
            compute_frameworks=[PandasDataFrame],
            api_data=_API,
            plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG, PrepareRunSubclassOptionsFG}),
        )
        planned = [f for f in _plan_features(session) if str(f.name) == "api_id"]
        assert planned
        for feature in planned:
            assert type(feature.options) is _TaggedOptions
            assert feature.options.tag == "kept"

    def test_requested_feature_keeps_options_subclass_with_copy_features_false(self) -> None:
        requested = Feature(name="PrepareRunApiFeature", options=_TaggedOptions({"k": 1}, tag="kept"))
        session = mloda.prepare(
            [requested],
            compute_frameworks=[PandasDataFrame],
            api_data=_API,
            plugin_collector=_enabled,
            copy_features=False,
        )
        planned = [f for f in _plan_features(session) if str(f.name) == "PrepareRunApiFeature"]
        assert planned
        for feature in planned:
            assert feature is not requested
            assert type(feature.options) is _TaggedOptions
            assert feature.options.tag == "kept"


_RTP_LEVELS = ("low", "mid", "high")


class RtpLevelFG(DataCreatorFeatureGroup):
    """Root group whose runtime parameter 'rtp_level' is read through runtime_parameter."""

    PROPERTY_MAPPING = {
        "rtp_level": property_spec("A level.", runtime=True, strict=True, allowed_values=_RTP_LEVELS, default="low")
    }
    calls = 0

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rtp_level_col", "rtp_level_col_b"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        return {str(f.name): [features.runtime_parameter("rtp_level", f)] for f in features.features}


class RtpTightFG(DataCreatorFeatureGroup):
    """Declares the same runtime key as RtpLevelFG with a narrower value space."""

    PROPERTY_MAPPING = {
        "rtp_level": property_spec("A level.", runtime=True, strict=True, allowed_values=("low", "mid"), default="low")
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rtp_tight_col"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rtp_tight_col": [features.runtime_parameter("rtp_level")]}


_RTP_EDGE_KEYS = ("rtp_coll", "rtp_none_no", "rtp_none_ok", "rtp_zero")


class RtpEdgeFG(DataCreatorFeatureGroup):
    """Runtime keys for collection, explicit-None and falsy run values."""

    PROPERTY_MAPPING = {
        "rtp_coll": property_spec("Coll.", runtime=True, strict=True, allowed_values=("a", "b"), default="a"),
        "rtp_none_no": property_spec("No None.", runtime=True, default="base"),
        "rtp_none_ok": property_spec("None ok.", runtime=True, default="base", allow_explicit_none=True),
        "rtp_zero": property_spec("Falsy.", runtime=True, default="base"),
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rtp_edge_col"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rtp_edge_col": [{k: features.runtime_parameter(k) for k in _RTP_EDGE_KEYS}]}


def _edge_session(*names: str) -> Any:
    return mloda.prepare(
        list(names) or ["rtp_edge_col"],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=PluginCollector.enabled_feature_groups({RtpLevelFG, RtpTightFG, RtpEdgeFG}),
    )


@pytest.fixture(autouse=True)
def _reset_rtp_calls() -> None:
    RtpLevelFG.calls = 0


def _rtp_enabled() -> PluginCollector:
    return PluginCollector.enabled_feature_groups({RtpLevelFG, RtpTightFG})


def _rtp_session(*features: Feature | str, extender: Extender | None = None) -> Any:
    return mloda.prepare(
        list(features) or ["rtp_level_col"],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_rtp_enabled(),
        function_extender={extender} if extender else None,
    )


def _rtp_values(result: Any) -> dict[str, Any]:
    return {name: values[0] for frame in result for name, values in frame.items()}


def _rtp_feature(name: str, level: str) -> Feature:
    return Feature(name, options=Options(group={}, context={"rtp_level": level}))


class TestRuntimeParametersPerRun:
    """run(parameters=...) replaces a declared runtime option for that run only."""

    def test_default_is_used_without_parameters(self) -> None:
        assert _rtp_values(_rtp_session().run()) == {"rtp_level_col": "low"}

    def test_run_override_is_seen_by_the_step_and_the_next_run_sees_the_baseline(self) -> None:
        session = _rtp_session()

        assert _rtp_values(session.run(parameters={"rtp_level": "high"})) == {"rtp_level_col": "high"}
        assert _rtp_values(session.run()) == {"rtp_level_col": "low"}

    def test_stream_run_accepts_parameters(self) -> None:
        session = _rtp_session()

        assert _rtp_values(list(session.stream_run(parameters={"rtp_level": "mid"}))) == {"rtp_level_col": "mid"}

    def test_prepare_time_context_value_is_the_baseline(self) -> None:
        session = _rtp_session(_rtp_feature("rtp_level_col", "mid"))

        assert _rtp_values(session.run()) == {"rtp_level_col": "mid"}
        assert _rtp_values(session.run(parameters={"rtp_level": "high"})) == {"rtp_level_col": "high"}

    def test_per_feature_baseline_through_runtime_parameter(self) -> None:
        session = _rtp_session(_rtp_feature("rtp_level_col", "mid"), _rtp_feature("rtp_level_col_b", "high"))

        assert _rtp_values(session.run()) == {"rtp_level_col": "mid", "rtp_level_col_b": "high"}
        assert _rtp_values(session.run(parameters={"rtp_level": "low"})) == {
            "rtp_level_col": "low",
            "rtp_level_col_b": "low",
        }

    def test_one_value_applies_to_every_group_declaring_the_key(self) -> None:
        session = _rtp_session("rtp_level_col", "rtp_tight_col")

        assert _rtp_values(session.run(parameters={"rtp_level": "mid"})) == {
            "rtp_level_col": "mid",
            "rtp_tight_col": "mid",
        }

    def test_runs_with_different_values_keep_plan_identity_and_lock(self) -> None:
        session = _rtp_session()
        before = session.resolved_plan()
        lock_before, hash_before = _lock_text(before), plan_structure_hash(before)

        first = session.run(parameters={"rtp_level": "high"})
        second = session.run(parameters={"rtp_level": "mid"})

        after = session.resolved_plan()
        assert first.plan_id == second.plan_id == session.plan_id
        assert _lock_text(after) == lock_before
        assert plan_structure_hash(after) == hash_before
        assert "high" not in lock_before
        assert "high" not in _lock_text(after)
        (step,) = [s for s in session.resolved_plan() if s.step_kind == "compute"]
        assert step.feature_set_options is not None
        assert "rtp_level" not in step.feature_set_options.group
        baseline = _rtp_values(session.run())
        assert baseline == {"rtp_level_col": "low"}

    def test_a_list_run_value_is_checked_per_element_like_prepare(self) -> None:
        session = _edge_session()

        assert _rtp_values(session.run(parameters={"rtp_coll": ["a", "b"]}))["rtp_edge_col"]["rtp_coll"] == ["a", "b"]
        with pytest.raises(ValueError, match="rtp_coll"):
            session.run(parameters={"rtp_coll": ["a", "x"]})

    def test_none_is_refused_unless_the_spec_allows_it(self) -> None:
        session = _edge_session()

        with pytest.raises(ValueError, match="rtp_none_no"):
            session.run(parameters={"rtp_none_no": None})
        seen = _rtp_values(session.run(parameters={"rtp_none_ok": None}))["rtp_edge_col"]
        assert seen["rtp_none_ok"] is None

    @pytest.mark.parametrize("falsy", [0, "", False], ids=["zero", "empty", "false"])
    def test_falsy_run_values_replace_a_non_falsy_baseline(self, falsy: Any) -> None:
        seen = _rtp_values(_edge_session().run(parameters={"rtp_zero": falsy}))["rtp_edge_col"]

        assert seen["rtp_zero"] == falsy
        assert type(seen["rtp_zero"]) is type(falsy)

    def test_a_key_declared_only_outside_the_plan_is_unknown(self) -> None:
        session = _edge_session("rtp_level_col")

        with pytest.raises(ValueError, match="rtp_coll"):
            session.run(parameters={"rtp_coll": "a"})


class TestRuntimeParametersAreValidatedBeforeTheRun:
    """A bad parameters mapping is refused at the call, before on_run_start and before any step."""

    @staticmethod
    def _call(session: Any, how: str, parameters: Any) -> None:
        if how == "run":
            session.run(parameters=parameters)
        else:
            session.stream_run(parameters=parameters)

    @pytest.mark.parametrize("how", ["run", "stream_run"])
    @pytest.mark.parametrize(
        ("parameters", "match", "also_names"),
        [
            ({"rtp_nope": "x"}, "rtp_nope", ("rtp_level",)),
            ({"rtp_level": "extreme"}, "rtp_level", ("RtpLevelFG",)),
            ({1: "x"}, ".", ()),
            (["rtp_level"], ".", ()),
        ],
        ids=["unknown-key", "strict-value", "non-str-key", "non-mapping"],
    )
    def test_refused_at_the_call_before_on_run_start_and_any_step(
        self, how: str, parameters: Any, match: str, also_names: tuple[str, ...]
    ) -> None:
        extender = _StartCountingExtender()
        session = _rtp_session(extender=extender)
        calls_before = RtpLevelFG.calls

        with pytest.raises((ValueError, TypeError), match=match) as caught:
            self._call(session, how, parameters)

        assert all(name in str(caught.value) for name in also_names)
        assert extender.starts == 0
        assert RtpLevelFG.calls == calls_before

    def test_a_value_must_pass_every_declaring_group(self) -> None:
        session = _rtp_session("rtp_level_col", "rtp_tight_col")

        with pytest.raises(ValueError, match="RtpTightFG"):
            session.run(parameters={"rtp_level": "high"})

    def test_a_refused_value_is_not_printed_raw_for_a_secret_shaped_key(self) -> None:
        class RtpSecretFG(DataCreatorFeatureGroup):
            PROPERTY_MAPPING = {
                "rtp_api_token": property_spec(
                    "Token.", runtime=True, strict=True, allowed_values=("ok",), default="ok"
                )
            }

            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"rtp_secret_col"}

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PythonDictFramework}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"rtp_secret_col": [features.runtime_parameter("rtp_api_token")]}

        session = mloda.prepare(
            ["rtp_secret_col"],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({RtpSecretFG}),
        )
        raw = "rtp-raw-secret-value-1234"  # nosec B105

        with pytest.raises(ValueError, match="rtp_api_token") as caught:
            session.run(parameters={"rtp_api_token": raw})

        assert raw not in str(caught.value)

    def test_a_refused_credential_shaped_value_is_not_printed_under_a_plain_key(self) -> None:
        session = _rtp_session()

        with pytest.raises(ValueError, match="rtp_level") as caught:
            session.run(parameters={"rtp_level": "postgres://u:pw@h/db"})

        assert "pw" not in str(caught.value)
        assert "RtpLevelFG" in str(caught.value)

    def test_group_placement_of_a_runtime_key_is_refused_at_prepare(self) -> None:
        feature = Feature("rtp_level_col", options={"rtp_level": "mid"})

        with pytest.raises(ValueError, match="rtp_level"):
            _rtp_session(feature)

    def test_runtime_parameter_of_an_undeclared_key_raises(self) -> None:
        with pytest.raises(ValueError, match="rtp_level"):
            FeatureSet().runtime_parameter("rtp_level")

    def test_a_fresh_feature_set_declares_no_runtime_parameters(self) -> None:
        assert FeatureSet().runtime_parameters == ()


class RtpRecordFG(DataCreatorFeatureGroup):
    """Declares plain, secret-shaped and nested runtime parameters for run-record tests."""

    PROPERTY_MAPPING = {
        "rtp_plain": property_spec("Plain.", runtime=True, default="base"),
        "api_token": property_spec("Token.", runtime=True, default="base-token"),  # nosec B106
        "rtp_config": property_spec("Nested.", runtime=True, default={"password": "base-pw", "plain": 1}),
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rtp_record_col"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rtp_record_col": [1]}


def _record_session(extender: Extender, *, api: bool = False, rtp: bool = False) -> Any:
    groups: set[type[FeatureGroup]] = {ApiDataFG, PrepareRunApiFeature, RtpRecordFG}
    features: list[Feature | str] = []
    if api:
        features.append("PrepareRunApiFeature")
    if rtp:
        features.append("rtp_record_col")
    return mloda.prepare(
        features,
        compute_frameworks=[PandasDataFrame, PythonDictFramework, "PyArrowTable"],
        api_data=_API if api else None,
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
        function_extender={extender},
    )


class TestRunSurface:
    """session.run_surface lists what run() accepts."""

    def test_api_data_columns_per_set_are_sorted(self) -> None:
        session = _record_session(_StartCountingExtender(), api=True)

        surface = session.run_surface
        assert isinstance(surface, RunSurface)
        assert surface.api_data == {"PrepareExample": ("api_id", "api_value")}

    def test_parameters_map_each_key_to_the_sorted_declaring_groups(self) -> None:
        session = _rtp_session("rtp_level_col", "rtp_tight_col")

        assert session.run_surface.parameters == {"rtp_level": ("RtpLevelFG", "RtpTightFG")}

    def test_a_plan_without_inputs_has_an_empty_surface(self) -> None:
        surface = _record_session(_StartCountingExtender()).run_surface

        assert (dict(surface.api_data), surface.artifacts, dict(surface.parameters), surface.credentials) == (
            {},
            frozenset(),
            {},
            (),
        )

    def test_parameters_accepted_by_the_surface_run(self) -> None:
        session = _rtp_session()

        assert _rtp_values(session.run(parameters={k: "mid" for k in session.run_surface.parameters})) == {
            "rtp_level_col": "mid"
        }

    def test_mappings_are_read_only_and_the_surface_is_frozen(self) -> None:
        surface = _record_session(_StartCountingExtender(), api=True, rtp=True).run_surface

        with pytest.raises(TypeError):
            surface.api_data["x"] = ("c",)
        with pytest.raises(TypeError):
            surface.parameters["x"] = ("G",)
        with pytest.raises(dataclasses.FrozenInstanceError):
            surface.credentials = ("x",)


class TestRunInputsOnRunStart:
    """on_run_start sees the names and masked values this run was given."""

    def test_api_data_names_come_from_the_run_argument(self) -> None:
        recorder = _StartCountingExtender()
        session = _record_session(recorder, api=True)

        session.run(api_data=_API)

        assert recorder.runs[-1].run_inputs.api_data == ("PrepareExample",)

    def test_api_data_names_fall_back_to_the_prepared_sets(self) -> None:
        recorder = _StartCountingExtender()
        session = _record_session(recorder, api=True)

        session.run()

        assert recorder.runs[-1].run_inputs.api_data == ("PrepareExample",)

    @pytest.mark.parametrize(
        ("field", "empty"),
        [("api_data", ()), ("parameters", {})],
        ids=["api-data", "parameters"],
    )
    def test_a_run_without_that_input_records_none(self, field: str, empty: Any) -> None:
        recorder = _StartCountingExtender()
        _record_session(recorder, rtp=True).run()

        assert getattr(recorder.runs[-1].run_inputs, field) == empty

    def test_secret_shaped_parameter_values_are_masked_in_the_stored_record(self) -> None:
        recorder = _StartCountingExtender()
        session = _record_session(recorder, rtp=True)
        raw = "rtp-raw-token-1234"  # nosec B105

        session.run(parameters={"api_token": raw, "rtp_plain": "mine", "rtp_config": {"password": raw, "plain": 2}})

        record = recorder.runs[-1].run_inputs
        assert record.parameters["api_token"] == "***"
        assert record.parameters["rtp_plain"] == "mine"
        assert record.parameters["rtp_config"]["password"] == "***"
        assert record.parameters["rtp_config"]["plain"] == 2
        with pytest.raises(TypeError):
            record.parameters["x"] = 1

    def test_credential_shaped_value_under_a_plain_key_is_masked_in_the_stored_record(self) -> None:
        recorder = _StartCountingExtender()
        session = _record_session(recorder, rtp=True)

        session.run(parameters={"rtp_plain": "postgres://u:pw@h/db"})

        assert "pw" not in str(recorder.runs[-1].run_inputs.parameters["rtp_plain"])

    def test_scalars_stay_raw_and_non_scalars_are_stored_as_text(self) -> None:
        recorder = _StartCountingExtender()
        session = _record_session(recorder, rtp=True)
        caller_list = ["x", "y"]

        session.run(parameters={"rtp_plain": caller_list})
        listed = recorder.runs[-1].run_inputs.parameters["rtp_plain"]
        caller_list.append("z")
        assert isinstance(listed, str)
        assert "z" not in listed

        session.run(parameters={"rtp_plain": 7})
        assert recorder.runs[-1].run_inputs.parameters["rtp_plain"] == 7
        session.run(parameters={"rtp_plain": "mine"})
        assert recorder.runs[-1].run_inputs.parameters["rtp_plain"] == "mine"
