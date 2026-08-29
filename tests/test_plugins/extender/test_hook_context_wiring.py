"""Tests wiring HookContext through ComputeFramework's three Extender hook call sites.

Exercises run_calculate_feature, run_validate_input_features, and
run_validate_output_features on a concrete PythonDictFramework instance.
"""

import functools
import gc
from typing import Any

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import FeatureGroup
from mloda.user import Feature, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _CalcFeatureGroup(FeatureGroup):
    """Root feature group (input_features unimplemented) returning a sized list."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return [{"value": i} for i in range(3)]


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


class _OldStyleExtender(Extender):
    """Mirrors today's real extenders: ignores HookContext entirely."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _build_feature_set() -> FeatureSet:
    return FeatureSet([Feature("my_feature")])


def _build_framework(extenders: set[Extender]) -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender=extenders)


class TestCalculateFeatureHookContext:
    """FEATURE_GROUP_CALCULATE_FEATURE hook surfaces a populated HookContext."""

    def test_captures_call_facts_and_post_call_instrumentation(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        cfw = _build_framework({extender})

        result = cfw.run_calculate_feature(_CalcFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        assert captured.feature_group_class == f"{_CalcFeatureGroup.__module__}.{_CalcFeatureGroup.__qualname__}"
        assert captured.feature_group_version == _CalcFeatureGroup.version()
        assert captured.feature_names == feature_set.get_all_names()
        assert captured.compute_framework_name == "PythonDictFramework"
        assert captured.status == "success"
        assert captured.duration_seconds is not None
        assert captured.duration_seconds >= 0
        assert captured.rows_out == len(result)


class TestValidateInputFeatureHookContext:
    """VALIDATE_INPUT_FEATURE hook surfaces a populated HookContext."""

    def test_captures_hook_and_success_status(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender(ExtenderHook.VALIDATE_INPUT_FEATURE)
        cfw = _build_framework({extender})
        # run_validate_input_features returns immediately without touching extenders when data is None.
        cfw.data = {"col": [1, 2, 3]}

        cfw.run_validate_input_features(_CalcFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.hook == ExtenderHook.VALIDATE_INPUT_FEATURE
        assert captured.status == "success"


class TestValidateOutputFeatureHookContext:
    """VALIDATE_OUTPUT_FEATURE hook surfaces a populated HookContext."""

    def test_captures_hook_and_success_status(self) -> None:
        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        cfw = _build_framework({extender})
        # Preconditions: data isn't None; either no feature is initial_requested_data (as here)
        # or column_names is populated; features carry no data_type so DataTypeValidator no-ops.
        cfw.data = {"col": [1, 2, 3]}
        cfw.set_column_names()

        cfw.run_validate_output_features(_CalcFeatureGroup, feature_set)

        captured = extender.captured
        assert captured is not None
        assert captured.hook == ExtenderHook.VALIDATE_OUTPUT_FEATURE
        assert captured.status == "success"


class TestBackwardCompatibility:
    """Adding the context seam must not change what an unmodified old-style extender receives."""

    def test_old_style_extender_behavior_is_unchanged(self) -> None:
        feature_set = _build_feature_set()

        cfw_with_extender = _build_framework({_OldStyleExtender()})
        result_with_extender = cfw_with_extender.run_calculate_feature(_CalcFeatureGroup, feature_set)

        cfw_without_extender = _build_framework(set())
        result_without_extender = cfw_without_extender.run_calculate_feature(_CalcFeatureGroup, feature_set)

        assert result_with_extender == result_without_extender


class TestHookContextScoping:
    """HookContext.current() is scoped to the hook call, not leaked before/after."""

    def test_current_is_none_before_and_after_the_call(self) -> None:
        assert HookContext.current() is None

        feature_set = _build_feature_set()
        extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        cfw = _build_framework({extender})

        cfw.run_calculate_feature(_CalcFeatureGroup, feature_set)

        assert HookContext.current() is None


class TestCompositeExtenderHookContext:
    """Multiple extenders on the same hook observe the identical HookContext instance."""

    def test_both_extenders_capture_the_same_context_instance(self) -> None:
        feature_set = _build_feature_set()
        extender_low = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, priority=10)
        extender_high = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, priority=20)
        cfw = _build_framework({extender_low, extender_high})

        result = cfw.run_calculate_feature(_CalcFeatureGroup, feature_set)

        assert extender_low.captured is not None
        assert extender_high.captured is not None
        assert extender_low.captured is extender_high.captured
        assert extender_low.captured.status == "success"
        assert extender_low.captured.duration_seconds is not None
        assert extender_low.captured.rows_out == len(result)


class TestDeclaredInputFeaturesBestEffort:
    """Declared input features surface as None for the common root-feature case."""

    def test_root_feature_group_input_features_is_none(self) -> None:
        feature_set = _build_feature_set()
        assert feature_set.options is not None
        assert _CalcFeatureGroup().is_root(feature_set.options, feature_set.get_name_of_one_feature()) is True

        extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        cfw = _build_framework({extender})

        cfw.run_calculate_feature(_CalcFeatureGroup, feature_set)

        assert extender.captured is not None
        assert extender.captured.input_features is None


class TestObservabilityFailureDoesNotBreakCalculation:
    """An observability read failing must never fail run_calculate_feature itself."""

    def test_ctor_requiring_arg_degrades_input_features_to_none(self) -> None:
        class _CtorRequiresArgFeatureGroup(FeatureGroup):
            """FeatureGroup whose __init__ requires a positional argument, so zero-arg feature_group() raises."""

            def __init__(self, required: int) -> None:
                self.required = required

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return [{"value": i} for i in range(3)]

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            result = cfw.run_calculate_feature(_CtorRequiresArgFeatureGroup, feature_set)

            assert result == [{"value": i} for i in range(3)]
            assert extender.captured is not None
            assert extender.captured.input_features is None
        finally:
            del _CtorRequiresArgFeatureGroup
            gc.collect()

    def test_version_raising_degrades_to_unavailable_fallback(self) -> None:
        class _VersionRaisesFeatureGroup(FeatureGroup):
            """FeatureGroup whose version() classmethod raises."""

            @classmethod
            def version(cls) -> str:
                raise RuntimeError("boom")

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return [{"value": i} for i in range(3)]

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            result = cfw.run_calculate_feature(_VersionRaisesFeatureGroup, feature_set)

            assert result == [{"value": i} for i in range(3)]
            assert extender.captured is not None
            assert extender.captured.feature_group_version == "unavailable"
        finally:
            del _VersionRaisesFeatureGroup
            gc.collect()

    def test_version_returning_non_str_normalizes_to_unavailable_fallback(self) -> None:
        class _VersionReturnsIntFeatureGroup(FeatureGroup):
            """FeatureGroup whose version() classmethod returns a non-str value."""

            @classmethod
            def version(cls) -> str:
                return 123  # type: ignore[return-value]

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return [{"value": i} for i in range(3)]

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            result = cfw.run_calculate_feature(_VersionReturnsIntFeatureGroup, feature_set)

            assert result == [{"value": i} for i in range(3)]
            assert extender.captured is not None
            assert extender.captured.feature_group_version == "unavailable"
            assert isinstance(extender.captured.feature_group_version, str)
        finally:
            del _VersionReturnsIntFeatureGroup
            gc.collect()

    def test_rows_in_len_raising_degrades_to_none(self) -> None:
        class _LenRaisesDouble:
            """Stand-in whose __len__ raises, simulating an observability row-count read gone wrong."""

            def __len__(self) -> int:
                raise RuntimeError("len boom")

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})
            cfw.data = _LenRaisesDouble()

            result = cfw.run_calculate_feature(_CalcFeatureGroup, feature_set)

            assert result == [{"value": i} for i in range(3)]
            assert extender.captured is not None
            assert extender.captured.rows_in is None
        finally:
            del _LenRaisesDouble
            gc.collect()

    def test_rows_out_len_raising_degrades_to_none(self) -> None:
        class _LenRaisesDouble:
            """Stand-in whose __len__ raises, simulating an observability row-count read gone wrong."""

            def __len__(self) -> int:
                raise RuntimeError("len boom")

        class _ReturnsLenRaisesFeatureGroup(FeatureGroup):
            """Returns a _LenRaisesDouble: the rows_out read must not break a successful calculation."""

            double_factory = _LenRaisesDouble

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return cls.double_factory()

        try:
            feature_set = _build_feature_set()
            extender = _ContextCapturingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
            cfw = _build_framework({extender})

            result = cfw.run_calculate_feature(_ReturnsLenRaisesFeatureGroup, feature_set)

            assert isinstance(result, _LenRaisesDouble)
            assert extender.captured is not None
            assert extender.captured.rows_out is None
        finally:
            del _ReturnsLenRaisesFeatureGroup
            del _LenRaisesDouble
            gc.collect()


class _FeatureGroupNameCapturingExtender(Extender):
    """Calls Extender.feature_group_name(func) on the func it's handed and records the result."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.seen_name: str | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.seen_name = Extender.feature_group_name(func)
        return func(*args, **kwargs)


class TestInstrumentPreservesSelfForNameResolution:
    """instrument's wrapper must carry __self__ so feature_group_name resolves correctly."""

    def test_feature_group_name_resolves_through_extra_decorator(self) -> None:
        def _plain_function_no_self(*args: Any, **kwargs: Any) -> Any:
            """Function with no __self__: the wraps() target of _extra_decorator below."""
            return None

        def _extra_decorator(func: Any, _target: Any = _plain_function_no_self) -> Any:
            """Simulates a plugin's own decorator stacked on @classmethod, using functools.wraps on a
            __self__-less function so unwrap walks past the real bound classmethod."""

            @functools.wraps(_target)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        class _DoublyDecoratedFeatureGroup(FeatureGroup):
            """calculate_feature is wrapped in an extra decorator whose own __wrapped__ chain leads elsewhere."""

            @classmethod
            @_extra_decorator
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return [{"value": i} for i in range(3)]

        try:
            feature_set = _build_feature_set()
            extender = _FeatureGroupNameCapturingExtender()
            cfw = _build_framework({extender})

            cfw.run_calculate_feature(_DoublyDecoratedFeatureGroup, feature_set)

            assert extender.seen_name == "_DoublyDecoratedFeatureGroup"
        finally:
            del _DoublyDecoratedFeatureGroup
            del _extra_decorator
            del _plain_function_no_self
            gc.collect()
