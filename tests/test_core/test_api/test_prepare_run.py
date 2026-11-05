"""
Tests for the prepare() / run() public API on mlodaAPI.

These tests define the contract for a two-phase execution model:
  1. prepare() - classmethod that builds the execution plan and returns a reusable mlodaAPI instance
  2. run() - instance method that executes with fresh api_data, reusing the cached plan
"""

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
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import mloda, mlodaAPI, Feature, ParallelizationMode, PluginCollector
from mloda.provider import ApiDataFG, ComputeFramework, FeatureGroup, FeatureSet, DataCreatorFeatureGroup, PropertySpec
from tests.test_core.test_abstract_plugins.name_bound_fixtures import make_name_bound_fg, make_root_fg
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
    """Counts on_run_start calls."""

    def __init__(self) -> None:
        self.starts = 0

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        self.starts += 1


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
