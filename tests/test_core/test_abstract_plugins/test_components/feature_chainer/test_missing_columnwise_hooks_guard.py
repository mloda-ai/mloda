"""A framework-bound class that skips a required column-wise hook is warned at class-definition time.

The three hooks are raising defaults on FeatureChainParserMixin, so a downstream author only learned
about a skipped hook from a NotImplementedError raised after the upstream feature groups had already
computed. ``warn_missing_columnwise_hooks`` moves that signal to the class body.

It fires only for a class that (a) inherits a non-empty REQUIRED_COLUMNWISE_HOOKS, (b) declares
``compute_framework_rule`` in its OWN __dict__, which is the static marker of a framework-bound
implementation, and (c) still resolves at least one required hook to the raising default. The guard
never CALLS compute_framework_rule: a class-definition-time call would run author code too early.

All fixture names carry a "c898" suffix so they cannot collide in the global plugin registry.

The guard is a DIAGNOSTIC, so three properties hold whatever an author declares: class creation never
fails because of it, only real hook names are ever reported, and a class that owns no hook at all owes
every hook it declared.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards import (
    warn_missing_columnwise_hooks,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    COLUMN_DISCOVERY_HOOKS,
    COLUMNWISE_HOOKS,
    FeatureChainParserMixin,
    declared_columnwise_hooks,
    missing_columnwise_hooks,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)

AUTHOR_GUARDS_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"

ADD_HOOK = "_add_result_to_data"
CHECK_HOOK = "_check_source_features_exist"
DISCOVERY_HOOK = "_get_available_columns"

# A name no hook has: an unrecognized entry in a declaration is always author error.
UNKNOWN_HOOK = "_not_a_columnwise_hook_c898"

# What a plain-string declaration leaks when it is iterated: one "hook" per distinct character.
CHARACTER_JUNK = ", ".join(sorted(set(ADD_HOOK)))


class _DeclaringBaseC898(FeatureChainParserMixin):
    """Stands in for a family base: it declares the requirement and implements none of it."""

    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS


class _RaisingDeclarationC898:
    """A REQUIRED_COLUMNWISE_HOOKS descriptor whose read raises, the second shape of a broken declaration."""

    def __get__(self, obj: Any, owner: type) -> Any:
        raise RuntimeError("attribute lookup fails")


def _guard_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The guard's warnings, told apart from the sibling author guards by the hook names they carry."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == AUTHOR_GUARDS_LOGGER
        and record.levelno == logging.WARNING
        and any(hook in record.getMessage() for hook in (DISCOVERY_HOOK, CHECK_HOOK, ADD_HOOK))
    ]


def _all_guard_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every warning the author-guards module emitted, including ones that name no real hook."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == AUTHOR_GUARDS_LOGGER and record.levelno == logging.WARNING
    ]


def _warnings_naming(caplog: pytest.LogCaptureFixture, token: str) -> list[str]:
    """The guard's warnings carrying a token, so a message is pinned by content rather than by wording."""
    return [message for message in _all_guard_warnings(caplog) if token in message]


class TestMissingHookWarning:
    """The firing case and the shape of the message."""

    def test_framework_bound_subclass_implementing_nothing_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """One warning naming the class and every hook it left on the raising default."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _MissingEverythingC898(_DeclaringBaseC898):
                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

            assert _MissingEverythingC898.REQUIRED_COLUMNWISE_HOOKS == COLUMN_DISCOVERY_HOOKS

        warnings = _guard_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one definition-time warning, got {warnings}"
        message = warnings[0]
        assert "_MissingEverythingC898" in message, f"the warning does not name the class: {message}"
        for hook in sorted(COLUMN_DISCOVERY_HOOKS):
            assert hook in message, f"the warning omits the missing hook {hook}: {message}"

    def test_partial_implementation_warns_only_about_the_missing_hooks(self, caplog: pytest.LogCaptureFixture) -> None:
        """A class that implements one required hook is warned about the other two, not about that one."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _PartialC898(_DeclaringBaseC898):
                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

                @classmethod
                def _get_available_columns(cls, data: Any) -> set[str]:
                    return set(data.columns)

            assert _PartialC898.REQUIRED_COLUMNWISE_HOOKS == COLUMN_DISCOVERY_HOOKS

        warnings = _guard_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one definition-time warning, got {warnings}"
        message = warnings[0]
        assert CHECK_HOOK in message, f"the warning omits {CHECK_HOOK}: {message}"
        assert ADD_HOOK in message, f"the warning omits {ADD_HOOK}: {message}"
        assert DISCOVERY_HOOK not in message, f"the warning names the implemented {DISCOVERY_HOOK}: {message}"


class TestGuardStaysSilent:
    """The three shapes that must not be warned, so the diagnostic keeps its signal."""

    def test_complete_implementation_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """A framework-bound class implementing every required hook is nothing to warn about."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _CompleteC898(_DeclaringBaseC898):
                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

                @classmethod
                def _get_available_columns(cls, data: Any) -> set[str]:
                    return set(data.columns)

                @classmethod
                def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
                    return None

                @classmethod
                def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
                    return data

            assert _CompleteC898.REQUIRED_COLUMNWISE_HOOKS == COLUMN_DISCOVERY_HOOKS

        assert _guard_warnings(caplog) == []

    def test_declaring_base_without_a_framework_rule_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """A family base declares the requirement for its children and implements nothing: that is correct."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _AnotherBaseC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS

            assert "compute_framework_rule" not in _AnotherBaseC898.__dict__

        assert _guard_warnings(caplog) == []

    def test_framework_bound_class_without_a_requirement_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """A parse-only feature group binds a framework but needs no hook, so it must never be warned."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _NoRequirementC898(FeatureChainParserMixin):
                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

            assert _NoRequirementC898.REQUIRED_COLUMNWISE_HOOKS == frozenset()

        assert _guard_warnings(caplog) == []

    def test_subclass_of_a_complete_implementation_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """An INHERITED real implementation satisfies the requirement: only the raising default counts as missing.

        The subclass re-pins compute_framework_rule so the framework-bound marker holds and the test
        really exercises hook resolution rather than passing on the marker check.
        """
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _PandasAggregatedChildC898(PandasAggregatedFeatureGroup):
                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

            try:
                assert _PandasAggregatedChildC898.REQUIRED_COLUMNWISE_HOOKS == COLUMN_DISCOVERY_HOOKS
                assert _guard_warnings(caplog) == []
            finally:
                # Plugin discovery walks the live __subclasses__() registry, so a leaked child would
                # outrank its parent in later resolutions on this worker.
                del _PandasAggregatedChildC898
                gc.collect()


class TestGuardIsCallableStandalone:
    """The guard is a plain function on the owner class, callable outside class definition."""

    def test_direct_call_on_an_incomplete_class_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Calling it on an already defined incomplete class reproduces the same warning."""

        class _StandaloneC898(_DeclaringBaseC898):
            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):
            # Defining the class above already warned once; only the direct call is under test here.
            caplog.clear()
            warn_missing_columnwise_hooks(_StandaloneC898)

        warnings = _guard_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one warning from the direct call, got {warnings}"
        assert "_StandaloneC898" in warnings[0]

    def test_guard_does_not_call_the_framework_rule(self, caplog: pytest.LogCaptureFixture) -> None:
        """compute_framework_rule is a static marker here: a raising one must still be diagnosed, not executed."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _RaisingRuleC898(_DeclaringBaseC898):
                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    raise RuntimeError("compute_framework_rule must not be called at class definition")

            assert "compute_framework_rule" in _RaisingRuleC898.__dict__

        warnings = _guard_warnings(caplog)
        assert len(warnings) == 1, f"expected the marker to be read statically, got {warnings}"
        assert "_RaisingRuleC898" in warnings[0]


class TestMalformedDeclarationNeverAbortsClassCreation:
    """A diagnostic that can kill the class statement is not a diagnostic.

    The guard runs from ``__init_subclass__``, so any raise it lets through becomes the author's
    class-definition error. Its own docstring, and the docs page, promise a warning instead.
    """

    def test_non_iterable_declaration_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        """REQUIRED_COLUMNWISE_HOOKS = 5 is author error, but the class must still come into existence."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _NonIterableDeclC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = 5  # type: ignore[assignment]

                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

        assert declared_columnwise_hooks(_NonIterableDeclC898) == frozenset()
        assert missing_columnwise_hooks(_NonIterableDeclC898) == []

    def test_raising_declaration_read_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        """A descriptor whose __get__ raises is contained the same way: no hooks, no crash."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _RaisingDeclC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = _RaisingDeclarationC898()

                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

        assert declared_columnwise_hooks(_RaisingDeclC898) == frozenset()
        assert missing_columnwise_hooks(_RaisingDeclC898) == []


class TestOnlyRealHookNamesAreReported:
    """A name that is not a column-wise hook is never reported, and never warned about as if it were one."""

    def test_string_declaration_leaks_no_characters(self, caplog: pytest.LogCaptureFixture) -> None:
        """A forgotten pair of braces makes the declaration a string, which is iterable one character at a time."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _StringDeclC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = ADD_HOOK  # type: ignore[assignment]

                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

            declared = declared_columnwise_hooks(_StringDeclC898)
            missing = set(missing_columnwise_hooks(_StringDeclC898))

        assert declared <= COLUMN_DISCOVERY_HOOKS, f"declared reports non-hooks: {sorted(declared)}"
        assert missing <= COLUMN_DISCOVERY_HOOKS, f"missing reports non-hooks: {sorted(missing)}"
        junk = [message for message in _all_guard_warnings(caplog) if CHARACTER_JUNK in message]
        assert junk == [], f"the warning names the characters of the declaration: {junk}"

    def test_unknown_name_is_dropped_from_the_report(self, caplog: pytest.LogCaptureFixture) -> None:
        """The real hook of a half-wrong declaration is still reported; the unrecognized name is not."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _UnknownNameC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = frozenset({ADD_HOOK, UNKNOWN_HOOK})

            declared = declared_columnwise_hooks(_UnknownNameC898)
            missing = missing_columnwise_hooks(_UnknownNameC898)

        assert declared == frozenset({ADD_HOOK}), f"declared keeps the unknown name: {sorted(declared)}"
        assert missing == [ADD_HOOK], f"missing keeps the unknown name: {missing}"

    def test_unknown_name_is_surfaced_to_the_author(self, caplog: pytest.LogCaptureFixture) -> None:
        """Dropped, but not swallowed: an unrecognized hook name is always author error, so it is named."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _UnknownNameWarnedC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = frozenset({ADD_HOOK, UNKNOWN_HOOK})

            assert UNKNOWN_HOOK in _UnknownNameWarnedC898.REQUIRED_COLUMNWISE_HOOKS

        named = _warnings_naming(caplog, UNKNOWN_HOOK)
        assert len(named) == 1, f"expected exactly one warning naming the unrecognized hook, got {named}"
        assert "_UnknownNameWarnedC898" in named[0], f"the warning does not name the class: {named[0]}"


class TestAbsentHookCountsAsMissing:
    """A class that owns no hook at all owes every hook it declared, rather than owing nothing."""

    def test_class_without_the_mixin_reports_its_declaration_missing(self) -> None:
        """Absent is not implemented: a resolved hook of None must never read as 'not the raising default'."""

        class _NoHooksAtAllC898:
            """Declares the requirement without inheriting the mixin, so it owns none of the three hooks."""

            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

        assert declared_columnwise_hooks(_NoHooksAtAllC898) == COLUMNWISE_HOOKS
        assert missing_columnwise_hooks(_NoHooksAtAllC898) == sorted(COLUMNWISE_HOOKS)


class TestHookShapeDecidesImplementedness:
    """The hooks are invoked as ``cls._hook(...)``, so a shape that call cannot reach is not an implementation."""

    def test_plain_function_hook_counts_as_missing(self) -> None:
        """A hook written without @classmethod takes cls as its data argument and fails at runtime."""

        class _PlainFunctionHookC898(FeatureChainParserMixin):
            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
                return None

            def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:  # type: ignore[override]
                """Missing @classmethod: cls._add_result_to_data(data, name, result) shifts every argument."""
                return data

        assert missing_columnwise_hooks(_PlainFunctionHookC898) == [ADD_HOOK]

    def test_classmethod_hook_counts_as_implemented(self) -> None:
        """The shape every shipped implementation uses stays implemented."""

        class _ClassmethodHookC898(FeatureChainParserMixin):
            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
                return None

            @classmethod
            def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
                return data

        assert missing_columnwise_hooks(_ClassmethodHookC898) == []

    def test_staticmethod_hook_counts_as_implemented(self) -> None:
        """A staticmethod takes the call unchanged, so it is a legitimate implementation."""

        class _StaticmethodHookC898(FeatureChainParserMixin):
            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

            @staticmethod
            def _check_source_features_exist(data: Any, feature_names: list[str]) -> None:
                return None

            @staticmethod
            def _add_result_to_data(data: Any, feature_name: str, result: Any) -> Any:
                return data

        assert missing_columnwise_hooks(_StaticmethodHookC898) == []

    def test_guard_warns_about_the_plain_function_hook_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """The author-visible half of the same rule: the unreachable hook is named, the reachable one is not."""
        with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):

            class _PlainFunctionBoundC898(FeatureChainParserMixin):
                REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

                @classmethod
                def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                    return {PandasDataFrame}

                @classmethod
                def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
                    return None

                def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:  # type: ignore[override]
                    """Missing @classmethod, so the runtime call shifts every argument."""
                    return data

        warnings = _guard_warnings(caplog)
        assert len(warnings) == 1, f"expected exactly one definition-time warning, got {warnings}"
        assert ADD_HOOK in warnings[0], f"the warning omits the unreachable {ADD_HOOK}: {warnings[0]}"
        assert CHECK_HOOK not in warnings[0], f"the warning names the implemented {CHECK_HOOK}: {warnings[0]}"


class TestDynamicFeatureGroupFactoryIsNotWarned:
    """The factory injects a delegating compute_framework_rule into every class it builds.

    That injection is the guard's framework-bound marker, so a dynamic feature group over a family base
    is warned exactly as a hand-written framework implementation would be, although the caller supplied
    no compute-framework property and the injected rule only calls super().
    """

    def test_created_class_over_a_family_base_emits_no_hook_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        class_name = "DynAggregatedC898"
        created = None
        try:
            with caplog.at_level(logging.WARNING, logger=AUTHOR_GUARDS_LOGGER):
                created = DynamicFeatureGroupCreator.create(
                    properties={}, class_name=class_name, feature_group_cls=AggregatedFeatureGroup
                )
            assert created.__name__ == class_name
            assert _guard_warnings(caplog) == [], "no compute framework was pinned, so there is nothing to warn about"
        finally:
            # Plugin discovery walks the live __subclasses__() registry, and the factory caches by name,
            # so both the cache entry and the class itself have to go.
            DynamicFeatureGroupCreator._created_classes.pop(class_name, None)
            del created
            gc.collect()
