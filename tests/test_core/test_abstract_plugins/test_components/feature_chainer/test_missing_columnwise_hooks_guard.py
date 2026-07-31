"""A framework-bound class that skips a required column-wise hook is warned at class-definition time.

The three hooks are raising defaults on FeatureChainParserMixin, so a downstream author only learned
about a skipped hook from a NotImplementedError raised after the upstream feature groups had already
computed. ``warn_missing_columnwise_hooks`` moves that signal to the class body.

It fires only for a class that (a) inherits a non-empty REQUIRED_COLUMNWISE_HOOKS, (b) declares
``compute_framework_rule`` in its OWN __dict__, which is the static marker of a framework-bound
implementation, and (c) still resolves at least one required hook to the raising default. The guard
never CALLS compute_framework_rule: a class-definition-time call would run author code too early.

All fixture names carry a "c898" suffix so they cannot collide in the global plugin registry.
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
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

AUTHOR_GUARDS_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"

ADD_HOOK = "_add_result_to_data"
CHECK_HOOK = "_check_source_features_exist"
DISCOVERY_HOOK = "_get_available_columns"


class _DeclaringBaseC898(FeatureChainParserMixin):
    """Stands in for a family base: it declares the requirement and implements none of it."""

    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS


def _guard_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The guard's warnings, told apart from the sibling author guards by the hook names they carry."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == AUTHOR_GUARDS_LOGGER
        and record.levelno == logging.WARNING
        and any(hook in record.getMessage() for hook in (DISCOVERY_HOOK, CHECK_HOOK, ADD_HOOK))
    ]


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
