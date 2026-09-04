"""E2E tests for set_verified_context()/tenant_id/project_id/principal plumbing through the
real mlodaAPI SYNC execution path, down to HookContext. Also proves Options can never override
the seam's values."""

from typing import Any, Optional

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.verified_context import set_verified_context
from mloda.core.api.request import mlodaAPI
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _VerifiedContextFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"verified_context_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"verified_context_e2e_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_VerifiedContextFeatureGroup})


class _ContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None
        self.captured_features: FeatureSet | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        self.captured_features = args[1]
        return result


def _prepare_session(options: Optional[dict[str, Any]] = None) -> mlodaAPI:
    return mloda.prepare(
        [Feature(name="verified_context_e2e_col", options=options)],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED,
        parallelization_modes={ParallelizationMode.SYNC},
    )


class TestVerifiedContextSurfacesOnHookContext:
    def test_tenant_project_principal_surface_when_scope_wraps_session_run(self) -> None:
        extender = _ContextCapturingExtender()
        session = _prepare_session()

        with set_verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            session.run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        assert extender.captured is not None
        assert extender.captured.tenant_id == "acme"
        assert extender.captured.project_id == "proj1"
        assert extender.captured.principal == "hash123"


class TestVerifiedContextAbsentWithoutScope:
    def test_tenant_project_principal_stay_none_with_no_active_scope(self) -> None:
        extender = _ContextCapturingExtender()
        session = _prepare_session()

        session.run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        assert extender.captured is not None
        assert extender.captured.tenant_id is None
        assert extender.captured.project_id is None
        assert extender.captured.principal is None


class TestStreamRunReadsVerifiedContextAtCreationNotAtIteration:
    """Bug: stream_run/stream_all are generator functions, so their whole body (including the
    current_verified_context() read) only executes at first iteration, not at the stream_run()/
    stream_all() call. The captured tenant must reflect the scope active when the stream was
    CREATED, not whatever scope (or lack of one) is active when it is later consumed."""

    def test_tenant_reflects_the_scope_active_at_creation_not_at_later_consumption(self) -> None:
        extender = _ContextCapturingExtender()
        session = _prepare_session()

        with set_verified_context(tenant_id="acme", project_id="proj-a", principal="hash-a"):
            stream = session.stream_run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        list(stream)

        assert extender.captured is not None
        assert extender.captured.tenant_id == "acme"
        assert extender.captured.project_id == "proj-a"
        assert extender.captured.principal == "hash-a"

    def test_tenant_does_not_leak_from_a_different_scope_opened_after_stream_creation(self) -> None:
        """Sharper regression: draining a stream created under one tenant's scope inside a
        DIFFERENT tenant's later scope (e.g. a server reusing a worker across requests) must not
        stamp the run with the wrong tenant's identity."""
        extender = _ContextCapturingExtender()
        session = _prepare_session()

        with set_verified_context(tenant_id="acme", project_id="proj-a", principal="hash-a"):
            stream = session.stream_run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        with set_verified_context(tenant_id="tenant-b", project_id="proj-b", principal="hash-b"):
            list(stream)

        assert extender.captured is not None
        assert extender.captured.tenant_id == "acme"
        assert extender.captured.project_id == "proj-a"
        assert extender.captured.principal == "hash-a"


class TestOptionsCannotOverrideVerifiedContext:
    """Options values that look like the seam's keys must never reach HookContext's fields."""

    _SPOOFED_OPTIONS = {
        "tenant_id": "spoofed-tenant",
        "project_id": "spoofed-project",
        "user.hash": "spoofed-principal",
        "principal": "spoofed-principal-alt",
    }

    def _assert_spoofed_options_present_on_the_feature(self, extender: _ContextCapturingExtender) -> None:
        """Proves the spoofed options were genuinely on the feature at hook-capture time, so a
        regression that silently dropped `options` entirely could not pass this test vacuously."""
        assert extender.captured_features is not None
        feature = next(iter(extender.captured_features.features))
        assert feature.options.get("tenant_id") == "spoofed-tenant"
        assert feature.options.get("project_id") == "spoofed-project"
        assert feature.options.get("user.hash") == "spoofed-principal"
        assert feature.options.get("principal") == "spoofed-principal-alt"

    def test_spoofed_options_are_ignored_when_a_verified_scope_is_active(self) -> None:
        extender = _ContextCapturingExtender()
        session = _prepare_session(self._SPOOFED_OPTIONS)

        with set_verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            session.run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        assert extender.captured is not None
        assert extender.captured.tenant_id == "acme"
        assert extender.captured.project_id == "proj1"
        assert extender.captured.principal == "hash123"
        self._assert_spoofed_options_present_on_the_feature(extender)

    def test_spoofed_options_are_ignored_when_no_verified_scope_is_active(self) -> None:
        extender = _ContextCapturingExtender()
        session = _prepare_session(self._SPOOFED_OPTIONS)

        session.run(parallelization_modes={ParallelizationMode.SYNC}, function_extender={extender})

        assert extender.captured is not None
        assert extender.captured.tenant_id is None
        assert extender.captured.project_id is None
        assert extender.captured.principal is None
        self._assert_spoofed_options_present_on_the_feature(extender)
