"""E2E tests for run_id/carrier plumbing through the real mlodaAPI SYNC execution path,
down to HookContext."""

from typing import Any

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.uuid7_assertions import assert_valid_uuid7

_CARRIER = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}


class _RunIdCarrierFeatureGroupOne(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_id_carrier_e2e_col_one"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"run_id_carrier_e2e_col_one": [1, 2, 3]}


class _RunIdCarrierFeatureGroupTwo(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_id_carrier_e2e_col_two"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"run_id_carrier_e2e_col_two": [4, 5, 6]}


_ENABLED = PluginCollector.enabled_feature_groups({_RunIdCarrierFeatureGroupOne, _RunIdCarrierFeatureGroupTwo})


def _noop_child_bootstrap() -> None:
    """Proves child_bootstrap threads through without error; SYNC mode never invokes it."""


class _ContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


class _MultiCaptureExtender(Extender):
    """Appends every captured HookContext for FEATURE_GROUP_CALCULATE_FEATURE, in call order."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result


class TestCarrierParameterAcceptedAtEveryEntryPoint:
    """carrier follows exactly the call path function_extender already takes."""

    def test_run_all_accepts_carrier_kwarg(self) -> None:
        result = mloda.run_all(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
            carrier=_CARRIER,
        )

        assert len(result) == 1

    def test_stream_all_accepts_carrier_kwarg(self) -> None:
        result = list(
            mloda.stream_all(
                [Feature(name="run_id_carrier_e2e_col_one")],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                parallelization_modes={ParallelizationMode.SYNC},
                carrier=_CARRIER,
            )
        )

        assert len(result) == 1

    def test_session_run_accepts_carrier_kwarg(self) -> None:
        session = mloda.prepare(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        result = session.run(parallelization_modes={ParallelizationMode.SYNC}, carrier=_CARRIER)

        assert len(result) == 1

    def test_session_stream_run_accepts_carrier_kwarg(self) -> None:
        session = mloda.prepare(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        result = list(session.stream_run(parallelization_modes={ParallelizationMode.SYNC}, carrier=_CARRIER))

        assert len(result) == 1


class TestChildBootstrapParameterAcceptedAtEveryEntryPoint:
    """child_bootstrap follows exactly the call path function_extender/carrier already take."""

    def test_run_all_accepts_child_bootstrap_kwarg(self) -> None:
        result = mloda.run_all(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
            child_bootstrap=_noop_child_bootstrap,
        )

        assert len(result) == 1

    def test_stream_all_accepts_child_bootstrap_kwarg(self) -> None:
        result = list(
            mloda.stream_all(
                [Feature(name="run_id_carrier_e2e_col_one")],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                parallelization_modes={ParallelizationMode.SYNC},
                child_bootstrap=_noop_child_bootstrap,
            )
        )

        assert len(result) == 1

    def test_session_run_accepts_child_bootstrap_kwarg(self) -> None:
        session = mloda.prepare(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        result = session.run(parallelization_modes={ParallelizationMode.SYNC}, child_bootstrap=_noop_child_bootstrap)

        assert len(result) == 1

    def test_session_stream_run_accepts_child_bootstrap_kwarg(self) -> None:
        session = mloda.prepare(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        result = list(
            session.stream_run(parallelization_modes={ParallelizationMode.SYNC}, child_bootstrap=_noop_child_bootstrap)
        )

        assert len(result) == 1


class TestRunIdAndCarrierSurfaceOnHookContext:
    """The session's minted run_id and the exact carrier dict both reach the captured HookContext."""

    def test_captured_run_id_matches_session_and_carrier_matches_exact_dict_passed(self) -> None:
        extender = _ContextCapturingExtender()
        session = mloda.prepare(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        session.run(
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
            carrier=_CARRIER,
        )

        assert_valid_uuid7(session.run_id)
        assert extender.captured is not None
        assert extender.captured.run_id == session.run_id
        assert extender.captured.carrier == _CARRIER


class TestCarrierIsCopiedOnIngestNotAliased:
    """HookContext.carrier must be a copy, so a hook mutating it can't leak into the caller's dict."""

    def test_captured_carrier_is_not_the_same_object_as_the_dict_passed_in(self) -> None:
        extender = _ContextCapturingExtender()
        caller_carrier = dict(_CARRIER)
        session = mloda.prepare(
            [Feature(name="run_id_carrier_e2e_col_one")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        session.run(
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={extender},
            carrier=caller_carrier,
        )

        assert extender.captured is not None
        assert extender.captured.carrier == caller_carrier
        assert extender.captured.carrier is not caller_carrier

        # Mutating the ComputeFramework-owned copy must not leak back into the caller's dict.
        assert extender.captured.carrier is not None
        extender.captured.carrier["mutated"] = "yes"
        assert "mutated" not in caller_carrier
        assert caller_carrier == _CARRIER


class TestTwoFeatureGroupsShareSameRunId:
    """Two feature groups computed within the SAME run_all()/run() call get the SAME run_id."""

    def test_both_feature_groups_capture_the_same_run_id(self) -> None:
        extender = _MultiCaptureExtender()
        session = mloda.prepare(
            [
                Feature(name="run_id_carrier_e2e_col_one"),
                Feature(name="run_id_carrier_e2e_col_two"),
            ],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        session.run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        assert len(extender.captured) == 2
        run_ids = {context.run_id for context in extender.captured}
        assert run_ids == {session.run_id}
        assert_valid_uuid7(session.run_id)
