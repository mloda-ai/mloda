"""Tests for HookContext's declared-input resolution and related degrade-silently behaviors.

Covers root-feature-group warnings, plain-str/batched input_features declarations,
once-per-step resolution, warning-only extenders on validate hooks, row_count's
type-only __len__ gate, instrument's rows_out reset, and feature_group_version.
"""

import gc
import logging
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext, instrument
from mloda.provider import FeatureGroup
from mloda.user import Feature, FeatureName, Options, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_UTILS_LOGGER = "mloda.core.abstract_plugins.components.utils"


class _ContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, hook: ExtenderHook, priority: int = 100) -> None:
        self._hook = hook
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {self._hook}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


class _AllHooksContextCapturingExtender(Extender):
    """Wraps all three hooks, appending the active HookContext for every call, in order."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result


class _RaisesBeforeDelegatingExtender(Extender):
    """raise_on_error=False; raises before ever calling func."""

    def __init__(self, hook: ExtenderHook, priority: int = 100) -> None:
        self._hook = hook
        self.priority = priority
        self.raise_on_error = False

    def wraps(self) -> set[ExtenderHook]:
        return {self._hook}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("extender exploded before delegating")


def _build_feature_set() -> FeatureSet:
    return FeatureSet([Feature("my_feature")])


def _build_framework(extenders: set[Extender]) -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender=extenders)


class _RootFeatureGroup(FeatureGroup):
    """Root feature group: input_features left unimplemented."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"col": [1, 2, 3]}


class TestRootFeatureGroupLogsNoWarning:
    """A root FeatureGroup must not emit a 'Degraded field' WARNING for input_features."""

    def test_root_feature_group_produces_no_degraded_field_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        feature_set = _build_feature_set()
        extender = _AllHooksContextCapturingExtender()
        cfw = _build_framework({extender})
        cfw.data = {"col": [1, 2, 3]}

        with caplog.at_level(logging.WARNING, logger=_UTILS_LOGGER):
            cfw.run_validate_input_features(_RootFeatureGroup, feature_set)
            cfw.set_column_names()
            cfw.run_calculate_feature(_RootFeatureGroup, feature_set)
            cfw.run_validate_output_features(_RootFeatureGroup, feature_set)

        assert not any("Degraded field" in record.message for record in caplog.records)
        assert len(extender.captured) == 3
        assert all(context.input_features is None for context in extender.captured)


class TestPlainStringDeclaredInputs:
    """input_features may declare plain str names instead of Feature objects."""

    def test_plain_str_set_becomes_frozenset_of_names(self, caplog: pytest.LogCaptureFixture) -> None:
        class _PlainStrInputsFeatureGroup(FeatureGroup):
            """input_features declares plain str names, not Feature objects."""

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Any]]:
                return {"base_amount", "currency"}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"col": [1, 2, 3]}

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            with caplog.at_level(logging.WARNING, logger=_UTILS_LOGGER):
                cfw.run_calculate_feature(_PlainStrInputsFeatureGroup, feature_set)

            assert extender.captured is not None
            assert extender.captured.input_features == frozenset({"base_amount", "currency"})
            assert not any("Degraded field" in record.message for record in caplog.records)
        finally:
            del _PlainStrInputsFeatureGroup
            gc.collect()


class TestBatchedFeatureSetDeclaredInputsUnion:
    """A batched FeatureSet's declared inputs union across all requested feature names."""

    def test_declared_inputs_union_across_batch(self) -> None:
        class _BatchedInputFeatureGroup(FeatureGroup):
            """input_features returns one Feature per requested feature_name."""

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
                return {Feature(f"src_{str(feature_name)}")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"col": [1, 2, 3]}

        try:
            feature_set = FeatureSet([Feature("a"), Feature("b")])
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            cfw.run_calculate_feature(_BatchedInputFeatureGroup, feature_set)

            assert extender.captured is not None
            assert extender.captured.input_features == frozenset({"src_a", "src_b"})
        finally:
            del _BatchedInputFeatureGroup
            gc.collect()


class TestDeclaredInputsResolvedOncePerStep:
    """input_features and __init__ resolve exactly once per step, not once per hook site."""

    def test_input_features_and_init_called_once_across_the_step(self) -> None:
        class _CountingFeatureGroup(FeatureGroup):
            """Class-level counters track __init__ and input_features invocations."""

            init_calls = 0
            input_features_calls = 0

            def __init__(self) -> None:
                type(self).init_calls += 1

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Any]]:
                type(self).input_features_calls += 1
                return {"x"}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"col": [1, 2, 3]}

        try:
            extender = _AllHooksContextCapturingExtender()
            cfw = _build_framework({extender})
            cfw.data = {"col": [1, 2, 3]}
            cfw.set_column_names()
            feature_set = FeatureSet([Feature("my_feature")])

            cfw.run_validate_input_features(_CountingFeatureGroup, feature_set)
            cfw.run_calculate_feature(_CountingFeatureGroup, feature_set)
            cfw.run_validate_output_features(_CountingFeatureGroup, feature_set)

            assert _CountingFeatureGroup.input_features_calls == 1
            assert _CountingFeatureGroup.init_calls == 1
            assert len(extender.captured) == 3
            for context in extender.captured:
                assert context.input_features == frozenset({"x"})
        finally:
            del _CountingFeatureGroup
            gc.collect()


class _ValidateInputTrackingFeatureGroup(FeatureGroup):
    """Module-level to avoid registry-leak tracebacks; tracks whether validate_input_features ran."""

    executed = False

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        cls.executed = True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"col": [1, 2, 3]}


class _ValidateOutputTrackingFeatureGroup(FeatureGroup):
    """Module-level to avoid registry-leak tracebacks; tracks whether validate_output_features ran."""

    executed = False

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        cls.executed = True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"col": [1, 2, 3]}


class TestWarningOnlyExtenderOnValidateHooksDoesNotBreakRun:
    """A raise_on_error=False extender on a validate hook must not break the run."""

    def test_validate_input_extender_raising_before_delegating_still_runs_validation(self) -> None:
        _ValidateInputTrackingFeatureGroup.executed = False
        feature_set = _build_feature_set()
        extender = _RaisesBeforeDelegatingExtender(ExtenderHook.VALIDATE_INPUT_FEATURE)
        cfw = _build_framework({extender})
        cfw.data = {"col": [1, 2, 3]}

        cfw.run_validate_input_features(_ValidateInputTrackingFeatureGroup, feature_set)

        assert _ValidateInputTrackingFeatureGroup.executed is True

    def test_validate_output_extender_raising_before_delegating_still_runs_validation(self) -> None:
        _ValidateOutputTrackingFeatureGroup.executed = False
        feature_set = _build_feature_set()
        extender = _RaisesBeforeDelegatingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        cfw = _build_framework({extender})
        cfw.data = {"col": [1, 2, 3]}
        cfw.set_column_names()

        cfw.run_validate_output_features(_ValidateOutputTrackingFeatureGroup, feature_set)

        assert _ValidateOutputTrackingFeatureGroup.executed is True


class _NoLenGetattrRecordingDouble:
    """No __len__; a custom __getattr__ that records every name it's asked for and always raises."""

    def __init__(self) -> None:
        self.requested_names: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.requested_names.append(name)
        raise AttributeError(name)


class _NoLenGetattrFabricatingDouble:
    """No __len__; a custom __getattr__ that fabricates a callable for any requested name."""

    def __getattr__(self, name: str) -> Any:
        return lambda: 99


class TestRowCountNeverTriggersInstanceGetattr:
    """HookContext.row_count must gate on the TYPE's __len__, never an instance __getattr__ fallback."""

    def test_recording_getattr_double_returns_none_and_is_never_queried(self) -> None:
        double = _NoLenGetattrRecordingDouble()

        result = HookContext.row_count(double)

        assert result is None
        assert double.requested_names == []

    def test_fabricating_getattr_double_still_returns_none(self) -> None:
        double = _NoLenGetattrFabricatingDouble()

        result = HookContext.row_count(double)

        assert result is None


def _make_context(**overrides: Any) -> HookContext:
    required: dict[str, Any] = {
        "hook": ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        "feature_group_class": "tests.something.FakeFeatureGroup",
        "feature_group_version": "v1",
        "plugin_version": None,
        "feature_names": ("my_feature",),
        "input_features": None,
        "compute_framework_name": "FakeFramework",
    }
    required.update(overrides)
    return HookContext(**required)


class TestInstrumentResetsRowsOutOnEntry:
    """instrument must reset rows_out to None on entry, not leave a stale value from a prior call."""

    def test_rows_out_reset_to_none_on_entry_before_raising(self) -> None:
        context = _make_context()
        context.rows_out = 7

        def raise_value_error() -> None:
            raise ValueError("boom")

        wrapped = instrument(context, raise_value_error)

        with pytest.raises(ValueError, match="boom"):
            wrapped()

        assert context.rows_out is None
        assert context.status == "error"


class TestFeatureGroupVersionDegradesSilently:
    """A version() that raises must degrade to 'unavailable' without a WARNING log."""

    def test_version_raising_degrades_without_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        class _VersionRaisesFeatureGroup(FeatureGroup):
            """version() classmethod raises."""

            @classmethod
            def version(cls) -> str:
                raise RuntimeError("boom")

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"col": [1, 2, 3]}

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            with caplog.at_level(logging.WARNING, logger=_UTILS_LOGGER):
                cfw.run_calculate_feature(_VersionRaisesFeatureGroup, feature_set)

            assert extender.captured is not None
            assert extender.captured.feature_group_version == "unavailable"
            assert not any("Degraded field 'feature_group_version'" in record.message for record in caplog.records)
        finally:
            del _VersionRaisesFeatureGroup
            gc.collect()
