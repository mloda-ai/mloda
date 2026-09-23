"""
Test Suite for Missing Links Validation

This test verifies that the framework provides a helpful error message when a user
creates a feature with multiple dependencies but forgets to provide explicit Links.

Expected Behavior:
When a feature declares multiple input_features() but no Links are provided,
the Engine should raise a clear ValueError during initialization (not at runtime)
with guidance on how to fix the issue.

This prevents confusing KeyError messages at runtime and educates users about
the requirement for explicit Links when merging multiple dependencies.
"""

from typing import Any, cast

import pyarrow.compute as pc
import pytest

from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.user import DataType
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# Test Feature Groups
class RootFeatureA(FeatureGroup):
    """First root feature for testing"""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], cls.get_class_name(): [10, 20, 30]}


class RootFeatureB(FeatureGroup):
    """Second root feature for testing"""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], cls.get_class_name(): [100, 200, 300]}


class MultiDependencyFeature(FeatureGroup):
    """
    Feature with multiple dependencies - requires explicit Links to merge them.
    This will trigger the validation error when Links are not provided.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature.int32_of("RootFeatureA"),
            Feature.int32_of("RootFeatureB"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # This would fail at runtime if Links are missing
        col_a = data.column("RootFeatureA")
        col_b = data.column("RootFeatureB")
        return {cls.get_class_name(): pc.add(col_a, col_b)}


class SplitMetricSource(FeatureGroup):
    """Root feature group producing two columns from one DataCreator; requested with
    differing Options it plans as two separate option buckets."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"metric_a", "metric_b"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = {"metric_a": [10, 20, 30], "metric_b": [100, 200, 300]}
        return {"id": [1, 2, 3], **{name: values[name] for name in features.get_all_names() if name in values}}


class MetricConverter(FeatureGroup):
    """Requests metric_a with an extra group option, splitting SplitMetricSource into a
    second option bucket relative to a sibling requesting metric_b without it."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("metric_a", options={"unit": "x"})}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data.column("metric_a")}


class MetricCombiner(FeatureGroup):
    """Declares inputs spanning both option buckets of SplitMetricSource (via MetricConverter
    and directly via metric_b) with no Link, triggering the runtime KeyError."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature.int32_of("MetricConverter"),
            Feature.int32_of("metric_b"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        col_conv = data.column("MetricConverter")
        col_b = data.column("metric_b")
        return {cls.get_class_name(): pc.add(col_conv, col_b)}


class MetricConverterScale(FeatureGroup):
    """Requests metric_a with unit AND scale, giving SplitMetricSource a third option bucket that
    MetricCombiner never spans (it only spans the {unit: 'x'} and {} buckets)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("metric_a", options={"unit": "x", "scale": 2})}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data.column("metric_a")}


class SplitSourceDtype(FeatureGroup):
    """Root feature group producing two columns, requested with differing data types (no options)."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"d_a", "d_b"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = {"d_a": [10, 20, 30], "d_b": [100, 200, 300]}
        return {"id": [1, 2, 3], **{name: values[name] for name in features.get_all_names() if name in values}}


class DtypeCombiner(FeatureGroup):
    """Requests d_a as int32 and d_b as int64 with no Link: a data-type-only split of
    SplitSourceDtype, unrelated to any option."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("d_a"), Feature.int64_of("d_b")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): pc.add(data.column("d_a"), data.column("d_b"))}


class DtypeOptionSibling(FeatureGroup):
    """Requests d_a with an unrelated option, giving SplitSourceDtype a third, option-differing
    bucket elsewhere in the plan (unrelated to DtypeCombiner's dtype-only split)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("d_a", options={"unit": "x"})}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data.column("d_a")}


class R5Root(FeatureGroup):
    """Single root feature; X5Intermediate splits on top of it without ever breaking anything."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"r5"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], "r5": [1, 2, 3]}


def _non_forwarded_k(value: int) -> Options:
    options = Options(group={"k": value})
    options.mark_non_forwarded("k")
    return options


class X5Intermediate(FeatureGroup):
    """Non-root feature group: forwards r5 unchanged under two names. Requesting it with two
    different (non-forwarded) values of 'k' splits it into two option buckets, but since 'k' is
    never forwarded to r5, both buckets compute the exact same r5 values and merging them succeeds
    without any Link."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("r5", forward_group=False)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(str(features.get_all_names()[0]), data.column("r5"))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"x5_a", "x5_b"}


class Z5UnrelatedTypo(FeatureGroup):
    """Merges the two X5Intermediate option buckets (which succeeds, since 'k' never forwards),
    then hits an unrelated KeyError from a typo'd column access."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("x5_a", options=_non_forwarded_k(1), data_type=DataType.INT32),
            Feature("x5_b", options=_non_forwarded_k(2), data_type=DataType.INT32),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        base = pc.add(data.column("x5_a"), data.column("x5_b"))
        data.column("column_that_does_not_exist")
        return data.append_column(cls.get_class_name(), base)


class LinkedSplitSource(FeatureGroup):
    """Root feature group split into two option buckets that the user already Linked."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"l_a", "l_b"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = {"l_a": [10, 20, 30], "l_b": [100, 200, 300]}
        return {"id": [1, 2, 3], **{name: values[name] for name in features.get_all_names() if name in values}}


class LinkedConverter(FeatureGroup):
    """Requests l_a with an extra option, splitting LinkedSplitSource into a second bucket."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("l_a", options={"unit": "x"})}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column(cls.get_class_name(), data.column("l_a"))


class LinkedCombiner(FeatureGroup):
    """Merges the two (already-Linked) LinkedSplitSource buckets, then hits an unrelated
    KeyError from a typo'd column access."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature.int32_of("LinkedConverter"),
            Feature.int32_of("l_b", options={"unit": "y"}),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        base = pc.add(data.column("LinkedConverter"), data.column("l_b"))
        data.column("column_that_does_not_exist")
        return data.append_column(cls.get_class_name(), base)


class MixedKeySource(FeatureGroup):
    """Root feature group split by two option keys of different types (int and str)."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"m_x"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"id": [1, 2, 3], "m_x": [10, 20, 30]}


class ConvIntKey(FeatureGroup):
    """Requests m_x with an int option key (1), one of the two MixedKeySource buckets."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(
                "m_x",
                options=Options(group=cast("dict[str, Any]", {1: "a"})),
                data_type=DataType.INT32,
            )
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data.column("m_x")}


class ConvStrKey(FeatureGroup):
    """Requests m_x with a str option key ('unit'), the other MixedKeySource bucket."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("m_x", options={"unit": "b"})}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data.column("m_x")}


class CombMixedKeys(FeatureGroup):
    """Merges the two MixedKeySource buckets (int key vs str key) with no Link."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.int32_of("ConvIntKey"), Feature.int32_of("ConvStrKey")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): pc.add(data.column("ConvIntKey"), data.column("ConvStrKey"))}


class TestMissingLinksError:
    """Test suite for missing Links validation"""

    def test_missing_links_raises_helpful_error(self) -> None:
        """
        Test that a helpful error is raised when a feature has multiple dependencies
        but no Links are provided.

        Expected Error Location: During runtime
        Expected Error Type: Exception (wraps ValueError)
        Expected Error Content:
            - Mentions "Links" or "multiple dependencies"
            - Lists the feature name (MultiDependencyFeature)
            - Provides example code with Link.inner() and JoinSpec
            - Shows Link.inner_on() for feature groups without index_columns()
            - Lists available join types
        """
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MultiDependencyFeature")],
                links=set(),  # EMPTY - this should trigger the error
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {RootFeatureA, RootFeatureB, MultiDependencyFeature}
                ),
            )

        error_message = str(exc_info.value)

        # Verify error message contains helpful information
        assert "MultiDependencyFeature" in error_message, "Error should mention the feature with missing Links"

        # Check for guidance keywords
        assert any(keyword in error_message.lower() for keyword in ["link", "multiple", "dependencies"]), (
            "Error should mention Links or dependencies"
        )

        # Check for example code
        assert "Link.inner" in error_message, "Error should show Link.inner() example"
        assert "JoinSpec" in error_message, "Error should show JoinSpec usage"

        # Check for join type documentation
        assert any(join_type in error_message for join_type in ["inner", "left", "right", "outer"]), (
            "Error should list available join types"
        )

    def test_missing_links_error_has_no_option_split_hint_when_no_split_occurred(self) -> None:
        """Regression pin: without an option-split root, the error must not carry the new hint."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MultiDependencyFeature")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {RootFeatureA, RootFeatureB, MultiDependencyFeature}
                ),
            )

        error_message = str(exc_info.value)
        assert "differing option" not in error_message.lower(), (
            "Error should not mention a differing-option split when the root feature groups never split by options"
        )

    def test_missing_links_error_hints_option_split_when_root_splits_by_options(self) -> None:
        """When a downstream step's ancestors span two option buckets of one root feature group,
        the error must name that root and the differing option key."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MetricCombiner")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {SplitMetricSource, MetricConverter, MetricCombiner}
                ),
            )

        error_message = str(exc_info.value)

        assert "differing option" in error_message.lower(), "Error should mention the differing-option split hint"
        assert "SplitMetricSource" in error_message, (
            "Error should name the feature group whose option-based split caused the mismatch"
        )
        assert "unit" in error_message, "Error should name the differing option key that caused the split"

    def test_option_split_hint_only_names_keys_differing_across_the_spanned_buckets(self) -> None:
        """Bug 1: SplitMetricSource has three buckets ({unit: 'x'}, {}, {unit: 'x', scale: 2}), but
        MetricCombiner's ancestors only span the first two. The hint must name only 'unit', the key
        that actually differs between those two buckets, not 'scale', which only differs because of
        the unrelated third bucket."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("MetricCombiner"), Feature.int32_of("MetricConverterScale")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {SplitMetricSource, MetricConverter, MetricConverterScale, MetricCombiner}
                ),
            )

        error_message = str(exc_info.value)

        assert "unit" in error_message, "Error should still name 'unit', which differs across the spanned buckets"
        assert "scale" not in error_message, (
            "Error should not name 'scale': it only differs in an unrelated third bucket MetricCombiner never spans"
        )

    def test_option_split_hint_silent_when_spanned_buckets_differ_only_by_data_type(self) -> None:
        """Bug 1 (related false positive): DtypeCombiner splits SplitSourceDtype into two buckets
        that differ only by data type (no option at all). An unrelated third bucket of the same
        class (DtypeOptionSibling) differs by 'unit' elsewhere in the plan, but DtypeCombiner never
        spans that bucket, so its error must not claim a differing-option cause."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("DtypeCombiner"), Feature.int32_of("DtypeOptionSibling")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {SplitSourceDtype, DtypeCombiner, DtypeOptionSibling}
                ),
            )

        error_message = str(exc_info.value)

        assert "differing option" not in error_message.lower(), (
            "The two buckets DtypeCombiner spans differ only by data type, not by any option; the "
            "unrelated 'unit' split elsewhere must not be blamed"
        )

    def test_option_split_hint_does_not_blame_a_harmless_intermediate_split(self) -> None:
        """Bug 2: X5Intermediate (non-root) splits into two option buckets, but that split causes no
        actual problem since the option is never forwarded (both buckets compute identical data and
        merge without a Link). The real failure is an unrelated typo'd column access. The error must
        not blame X5Intermediate's harmless split."""
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("Z5UnrelatedTypo")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups({R5Root, X5Intermediate, Z5UnrelatedTypo}),
            )

        error_message = str(exc_info.value)

        assert "differing option" not in error_message.lower(), (
            "X5Intermediate's harmless option split (never forwarded, buckets compute identical "
            "data) must not be blamed for the unrelated typo'd column access"
        )
        assert "X5Intermediate" not in error_message

    def test_option_split_hint_does_not_blame_an_already_linked_split(self) -> None:
        """Bug 3: LinkedSplitSource splits into two option buckets, but the user already added a
        Link that resolves the split (the run succeeds for that part). The real failure is an
        unrelated typo'd column access. The error must not claim the already-linked split is the
        likely cause."""
        link = Link.inner(
            JoinSpec(LinkedSplitSource, "id"),
            JoinSpec(LinkedSplitSource, "id"),
            left_discriminator={"unit": "x"},
            right_discriminator={"unit": "y"},
        )

        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("LinkedCombiner")],
                links={link},
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {LinkedSplitSource, LinkedConverter, LinkedCombiner}
                ),
            )

        error_message = str(exc_info.value)

        assert "differing option" not in error_message.lower(), (
            "The split of LinkedSplitSource is already resolved by an explicit Link; it must not be "
            "named as the likely cause of the unrelated typo'd column access"
        )
        assert "is the likely cause" not in error_message

    def test_option_split_hint_does_not_crash_on_non_str_option_key(self) -> None:
        """Bug 4: MixedKeySource splits by an int option key (1) and a str option key ('unit'). The
        diagnostic must not let a TypeError from sorting mixed-type keys mask the intended, helpful
        ValueError."""
        with pytest.raises(ValueError) as exc_info:
            mloda.run_all(
                features=[Feature.int32_of("CombMixedKeys")],
                links=set(),
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {MixedKeySource, ConvIntKey, ConvStrKey, CombMixedKeys}
                ),
            )

        assert "CombMixedKeys" in str(exc_info.value)
