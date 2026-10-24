"""Tests for the MATCH-TIME compute-framework capability hook (issue #482).

These tests pin down the contract for a sanctioned way for a FeatureGroup to
declare an operation unsupported on a given compute framework, evaluated during
feature -> feature-group resolution.

Contract under test (to be implemented by the Green agent):

1. ``FeatureGroup.supports_compute_framework(feature_name, options, compute_framework) -> bool``
   classmethod, default ``True``. Arg order mirrors ``match_feature_group_criteria``.
2. ``IdentifyFeatureGroupClass._filter_loop`` narrows the candidate compute-framework
   set per feature using the hook (route-around when an unsupported framework is not
   the only candidate).
3. When the only candidate is an unsupported framework (e.g. user pin), a DEDICATED,
   distinguishable ``ValueError`` is raised that names the unsupported and supported
   framework(s) and skips the fuzzy "Did you mean" suggestion path.

All tests are written to FAIL until the feature exists.
"""

import importlib
from typing import Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import accessible_implementations
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.steward import get_feature_group_docs
from mloda.user import PluginCollector, mloda
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise, identify_winner


CAPABILITY_FEATURE = "capability_test_feature"


class CapabilityFwA(ComputeFramework):
    """First compute framework used in capability tests."""

    pass


class CapabilityFwB(ComputeFramework):
    """Second compute framework used in capability tests."""

    pass


class PlainCapabilityFeatureGroup(FeatureGroup):
    """Plain feature group that does NOT override supports_compute_framework.

    Used to assert the default hook behaviour and no-regression resolution.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class RejectBCapabilityFeatureGroup(FeatureGroup):
    """Feature group that declares the operation unsupported on CapabilityFwB.

    supports_compute_framework returns False for CapabilityFwB and True otherwise.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return compute_framework is not CapabilityFwB

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class RejectAllCapabilityFeatureGroup(FeatureGroup):
    """Feature group that declares the operation unsupported on every framework."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class DualDeclaringFG_782(FeatureGroup):
    """Declares BOTH frameworks, supports the op on CapabilityFwA only, and rejects CapabilityFwB.

    One candidate declaring both is what makes "declared but not enabled" observable per candidate.
    An earlier fixture split the two frameworks across two feature groups and relied on the message
    pooling supported/rejected across candidates; #791 removed that union, so each line now names only
    its OWN candidate's frameworks.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CapabilityFwA, CapabilityFwB}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return compute_framework is not CapabilityFwB

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class TestComputeFrameworkCapabilityHook:
    """Capability-hook contract for compute-framework support at match time."""

    def test_default_hook_is_true_and_no_regression(self) -> None:
        """Default hook returns True and resolution keeps every declared framework."""
        feature = Feature(CAPABILITY_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PlainCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        feature_group_class, compute_frameworks = identify_winner(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
        assert feature_group_class is PlainCapabilityFeatureGroup
        assert compute_frameworks == {CapabilityFwA, CapabilityFwB}, (
            f"Default behaviour must keep all declared frameworks; got {compute_frameworks}"
        )

        # The hook must exist on the base class and default to True for any framework.
        assert FeatureGroup.supports_compute_framework(CAPABILITY_FEATURE, feature.options, CapabilityFwA) is True
        assert FeatureGroup.supports_compute_framework(CAPABILITY_FEATURE, feature.options, CapabilityFwB) is True

    def test_route_around_unsupported_framework_without_pin(self) -> None:
        """Without a pin, an unsupported framework is dropped, the supported one kept."""
        feature = Feature(CAPABILITY_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RejectBCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        feature_group_class, compute_frameworks = identify_winner(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
        assert feature_group_class is RejectBCapabilityFeatureGroup
        assert compute_frameworks == {CapabilityFwA}, (
            f"Unsupported CapabilityFwB must be narrowed out, leaving only CapabilityFwA; got {compute_frameworks}"
        )

    def test_distinguishable_error_when_pinned_to_unsupported_framework(self) -> None:
        """Pinning to the unsupported framework names the pin as a compute-framework-pin near-miss."""
        feature = Feature(CAPABILITY_FEATURE)
        feature.compute_frameworks = {CapabilityFwB}

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RejectBCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "Feature group(s) eliminated while matching" in error_message, (
            f"Capability pin rejection must render as a near-miss line, but got: {error_message}"
        )
        assert "compute framework pin" in error_message, (
            f"A pin rejection must be labelled a compute framework pin, but got: {error_message}"
        )
        assert CapabilityFwB.get_class_name() in error_message, (
            f"Error must name the pinned unsupported framework '{CapabilityFwB.get_class_name()}', "
            f"but got: {error_message}"
        )
        assert CapabilityFwA.get_class_name() in error_message, (
            f"Error must name the supported framework '{CapabilityFwA.get_class_name()}', but got: {error_message}"
        )
        # The near-miss block already names the eliminated candidate; a fuzzy suggestion must not echo it.
        suggestion_line = next((line for line in error_message.split("\n") if line.startswith("Did you mean")), "")
        assert RejectBCapabilityFeatureGroup.get_class_name() not in suggestion_line, (
            f"the 'Did you mean' suggestion must not echo the already-named eliminated candidate, got: {error_message}"
        )

    def test_all_frameworks_reject_operation(self) -> None:
        """When every framework rejects the op, a graceful error names the feature."""
        feature = Feature(CAPABILITY_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RejectAllCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert CAPABILITY_FEATURE in error_message, f"Error must mention the feature name, but got: {error_message}"
        assert "Feature group(s) eliminated while matching" in error_message, (
            f"An all-rejected candidate must render as a capability near-miss, but got: {error_message}"
        )
        assert "supports_compute_framework rejected" in error_message, (
            f"Error must signal that supports_compute_framework rejected the frameworks, but got: {error_message}"
        )
        assert CapabilityFwA.get_class_name() in error_message, (
            f"Error must name rejected framework '{CapabilityFwA.get_class_name()}', but got: {error_message}"
        )
        assert CapabilityFwB.get_class_name() in error_message, (
            f"Error must name rejected framework '{CapabilityFwB.get_class_name()}', but got: {error_message}"
        )

    def test_capability_error_distinguishable_from_unknown_feature(self) -> None:
        """The capability error is clearly different from the unknown-feature error."""
        # Genuinely unknown feature -> classic resolution error.
        unknown_feature = Feature("totally_unknown_capability_feature_xyz")
        unknown_accessible: FeatureGroupEnvironmentMapping = {
            RejectAllCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        with pytest.raises(ValueError) as unknown_exc:
            evaluate_or_raise(
                feature=unknown_feature,
                accessible_plugins=unknown_accessible,
                links=None,
                data_access_collection=None,
            )
        unknown_message = str(unknown_exc.value)
        assert "No feature groups found" in unknown_message, (
            f"Unknown feature must keep the classic error, but got: {unknown_message}"
        )

        # Capability rejection -> distinct, dedicated error.
        pinned_feature = Feature(CAPABILITY_FEATURE)
        pinned_feature.compute_frameworks = {CapabilityFwB}
        capability_accessible: FeatureGroupEnvironmentMapping = {
            RejectBCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        with pytest.raises(ValueError) as capability_exc:
            evaluate_or_raise(
                feature=pinned_feature,
                accessible_plugins=capability_accessible,
                links=None,
                data_access_collection=None,
            )
        capability_message = str(capability_exc.value)

        # Both share the base "No feature groups found" line now; the capability rejection is
        # distinguished by its near-miss block, which the unknown-feature error never carries.
        assert "Feature group(s) eliminated while matching" in capability_message, (
            f"Capability error must name the eliminated near-miss candidate, but got: {capability_message}"
        )
        assert "Feature group(s) eliminated while matching" not in unknown_message, (
            f"Unknown-feature error must carry no near-miss block, but got: {unknown_message}"
        )

    def test_capability_reason_names_only_run_enabled_rejected_framework(self) -> None:
        """Run-only capability reason: only the run-enabled framework the candidate rejected is named.

        os-011 (Q2 = run-only) dropped the wider declared-available universe. ``DualDeclaringFG_782``
        declares {CapabilityFwA, CapabilityFwB} and supports the op on CapabilityFwA, but only CapabilityFwB
        is enabled for this run and CapabilityFwB is exactly what it rejects. The capability reason names only
        the run-enabled, rejected CapabilityFwB; the declared-but-not-enabled CapabilityFwA no longer appears.
        """
        feature = Feature(CAPABILITY_FEATURE)
        feature.compute_frameworks = {CapabilityFwB}

        # Only FwB is enabled for the run, and it is exactly the framework this candidate rejects.
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            DualDeclaringFG_782: {CapabilityFwB},
        }

        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "Feature group(s) eliminated while matching" in error_message, (
            f"Capability rejection must render as a near-miss line, but got: {error_message}"
        )
        assert f"supports_compute_framework rejected ['{CapabilityFwB.get_class_name()}']" in error_message, (
            f"Reason must name the run-enabled rejected framework '{CapabilityFwB.get_class_name()}', "
            f"but got: {error_message}"
        )
        assert CapabilityFwA.get_class_name() not in error_message, (
            "run-only capability reason must not name the declared-but-not-enabled framework "
            f"'{CapabilityFwA.get_class_name()}', but got: {error_message}"
        )
        # The eliminated candidate is named in the near-miss block; a fuzzy suggestion must not echo it.
        suggestion_line = next((line for line in error_message.split("\n") if line.startswith("Did you mean")), "")
        assert DualDeclaringFG_782.get_class_name() not in suggestion_line, (
            f"the 'Did you mean' suggestion must not echo the already-named eliminated candidate, got: {error_message}"
        )


def _toy_operation() -> Any:
    """Imported lazily so a missing OperationFeatureGroup fails only the tests that need it."""
    return importlib.import_module(
        "tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group"
    )


class TestOperationGroupImplementationCapability:
    """An Implementation's supported and accepts remove its framework at match time."""

    @pytest.mark.parametrize(
        "kind, extra, expected",
        [
            ("sum", {}, {"PandasDataFrame", "PyArrowTable"}),
            ("median", {}, {"PandasDataFrame"}),
            ("sum", {"optoy_pandas_off": "1"}, {"PyArrowTable"}),
            ("median", {"optoy_pandas_off": "1"}, set()),
        ],
    )
    def test_supported_and_accepts_narrow_the_frameworks(
        self, kind: str, extra: dict[str, str], expected: set[str]
    ) -> None:
        toy = _toy_operation()
        feature = Feature(toy.OPTOY_FEATURE, {toy.OPTOY_KEY: kind, **extra})
        frameworks = {toy.PandasDataFrame, toy.PyArrowTable}
        plugins: FeatureGroupEnvironmentMapping = {toy.OpToyFG: frameworks}

        if expected:
            _, winners = identify_winner(feature=feature, accessible_plugins=plugins)
            assert {fw.get_class_name() for fw in winners} == expected
            return
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(feature=feature, accessible_plugins=plugins)
        elimination = exc_info.value.result.eliminations[toy.OpToyFG]  # type: ignore[attr-defined]
        assert elimination.stage == "capability"

    @pytest.mark.parametrize(
        ("child_accepts", "parent_options", "succeeds"),
        [(False, {}, True), (True, {"optoy_pandas_off": "1"}, False)],
        ids=["disabled_child_rejects_over_accepting_parent", "disabled_child_accepts_over_rejecting_parent"],
    )
    def test_a_disabled_implementation_does_not_decide_the_capability(
        self, make_implementation: Any, child_accepts: bool, parent_options: dict[str, str], succeeds: bool
    ) -> None:
        toy = _toy_operation()
        child = make_implementation(
            toy.OpToyFG,
            toy.PandasDataFrame,
            lambda g, m, f: pd.DataFrame({toy.OPTOY_FEATURE: [toy.PANDAS_MARK]}),
            base=toy.OpToyPandas,
        )
        child.accepts = classmethod(lambda cls, feature_name, options: child_accepts)
        collector = PluginCollector.enabled_feature_groups({toy.OpToyFG})
        collector.add_disabled_implementation_classes({child})
        feature = Feature(toy.OPTOY_FEATURE, {toy.OPTOY_KEY: "sum", **parent_options})

        def run() -> list[Any]:
            return mloda.run_all([feature], compute_frameworks=[toy.PandasDataFrame], plugin_collector=collector)

        if succeeds:
            assert run()
            return
        with pytest.raises(ValueError):
            run()

    def test_group_level_supported_narrows_the_matrix_and_the_check(self) -> None:
        toy = _toy_operation()

        matrix = toy.OpToyNarrowFG.subtype_support_matrix()
        assert matrix["PandasDataFrame"] == frozenset({"sum"})
        assert matrix["PyArrowTable"] == frozenset({"sum"})
        median = Options({toy.OPTOY_KEY: "median"})
        total = Options({toy.OPTOY_KEY: "sum"})
        feature = toy.OpToyNarrowFG.FEATURE
        assert toy.OpToyNarrowFG.supports_compute_framework(feature, median, toy.PandasDataFrame) is False
        assert toy.OpToyNarrowFG.supports_compute_framework(feature, total, toy.PandasDataFrame) is True

    @pytest.mark.parametrize(
        ("group_name", "message"),
        [("OpToyHookFG", "supports_compute_framework"), ("OpToyUnservedFG", "PythonDictFramework")],
    )
    def test_matrix_raises_for_hook_override_or_unserved_framework(self, group_name: str, message: str) -> None:
        group = getattr(_toy_operation(), group_name)

        with pytest.raises(ValueError, match=message):
            group.subtype_support_matrix()

    def test_hook_override_group_surfaces_subtype_error_in_docs(self) -> None:
        toy = _toy_operation()

        docs = get_feature_group_docs(plugin_collector=PluginCollector.enabled_feature_groups(toy.OpToyHookFG))

        assert len(docs) == 1
        assert docs[0].subtype_error is not None
        assert "supports_compute_framework" in docs[0].subtype_error

    def test_subtype_support_matrix_reads_the_implementations(self, make_implementation: Any) -> None:
        toy = _toy_operation()

        matrix = toy.OpToyFG.subtype_support_matrix()
        assert matrix["PandasDataFrame"] == frozenset({"sum", "median"})
        assert matrix["PyArrowTable"] == frozenset({"sum"})
        assert "PythonDictFramework" not in matrix

        child = make_implementation(
            toy.OpToyFG,
            toy.PandasDataFrame,
            lambda g, m, f: pd.DataFrame({toy.OPTOY_FEATURE: [toy.PANDAS_MARK]}),
            base=toy.OpToyPandas,
        )
        child.accepts = classmethod(
            lambda cls, feature_name, options: (
                not (options.get(toy.OPTOY_KEY) == "median" and options.get(toy.OPTOY_PANDAS_OFF) is not None)
            )
        )

        def probe(kind: str, off: bool) -> bool:
            options = Options({toy.OPTOY_KEY: kind, **({toy.OPTOY_PANDAS_OFF: "1"} if off else {})})
            return bool(toy.OpToyFG.supports_compute_framework(toy.OPTOY_FEATURE, options, toy.PandasDataFrame))

        with accessible_implementations({child, toy.OpToyPyArrow}):
            # The matrix does not reflect accepts.
            assert toy.OpToyFG.subtype_support_matrix()["PandasDataFrame"] == frozenset({"sum", "median"})
            assert probe("median", True) is False
            assert probe("sum", True) is True
            assert probe("median", False) is True
