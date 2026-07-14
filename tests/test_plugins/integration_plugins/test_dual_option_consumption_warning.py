"""Failing tests pinning the dual option consumption warning contract.

During feature resolution (mloda.run_all), the engine must log exactly ONE
WARNING record per (consumer feature group class, resolved child feature group
class, overlapping key set) triple when an option key satisfies ALL of:
  (a) it was inherited onto an input feature from its consumer (it is in the
      child Options' inherited_group_keys; equal-value forwards count),
  (b) it is declared in the consumer feature group's PROPERTY_MAPPING,
  (c) it is declared in the resolved child feature group's PROPERTY_MAPPING.
The message must contain the option key name, the consumer feature group class
name, the child feature group class name, and the string "forward_group_exclude"
as the suggested remedy. No warning is logged when any of (a)-(c) is false.

The triple is per CONSUMER: when one shared child Feature instance is declared
by TWO consumer feature groups, each consumer forms its own triple and warns
separately (the second merge must not erase the first consumer's provenance).

All fixture feature names carry a "dualwarn579" marker so they cannot collide
with other tests in the global plugin registry.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import pandas as pd
import pytest

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import PropertySpec
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


SOURCE_NAME = "dualwarn579_source"
CLEAN_SOURCE_NAME = "dualwarn579_clean_source"
MODE_KEY = "dualwarn579_mode"
SALT_KEY = "dualwarn579_salt"


class DualWarn579SourceGroup(FeatureGroup):
    """Upstream root group that declares the mode key and records resolved options."""

    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 source group", context=False),
    }

    seen_options: ClassVar[list[dict[str, dict[str, Any]]]] = []

    @classmethod
    def reset_recording(cls) -> None:
        cls.seen_options = []

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({SOURCE_NAME})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            cls.seen_options.append(
                {
                    "group": dict(feature.options.group),
                    "context": dict(feature.options.context),
                }
            )
        return pd.DataFrame({SOURCE_NAME: [1, 2, 3]})


class DualWarn579CleanSourceGroup(FeatureGroup):
    """Upstream root group WITHOUT the mode key in its PROPERTY_MAPPING."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({CLEAN_SOURCE_NAME})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({CLEAN_SOURCE_NAME: [1, 2, 3]})


class _DualWarn579ConsumerBase(FeatureGroup):
    """Shared consumer behavior: double the upstream column when mode is "fast"."""

    FEATURE_NAME: ClassVar[str] = ""
    SOURCE_COLUMN: ClassVar[str] = SOURCE_NAME

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == cls.FEATURE_NAME

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            factor = 2 if feature.options.get(MODE_KEY) == "fast" else 1
            data[feature.name] = data[cls.SOURCE_COLUMN] * factor
        return data


class DualWarn579ConsumerGroup(_DualWarn579ConsumerBase):
    """Consumer that declares the mode key and forwards everything by default."""

    FEATURE_NAME = "dualwarn579_consumer"
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 consumer group", context=False),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME)}


class DualWarn579StringConsumerGroup(_DualWarn579ConsumerBase):
    """Consumer that declares the mode key but requests its child as a BARE STRING.

    The unified forwarding path gives the bare-string child its own fresh Options and
    runs the real inherit_from merge, so it records forwarding provenance exactly like
    an object child and must trigger the same dual-consumption warning.
    """

    FEATURE_NAME = "dualwarn579_string_consumer"
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 string consumer group", context=False),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Any]]:
        return {SOURCE_NAME}


class DualWarn579RepeatConsumerGroup(_DualWarn579ConsumerBase):
    """Consumer matching TWO feature names so one class can be requested twice in one run."""

    FEATURE_NAME = "dualwarn579_repeat_consumer_a"
    SECOND_FEATURE_NAME: ClassVar[str] = "dualwarn579_repeat_consumer_b"
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 repeat consumer group", context=False),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) in {cls.FEATURE_NAME, cls.SECOND_FEATURE_NAME}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME)}


class DualWarn579CleanConsumerGroup(_DualWarn579ConsumerBase):
    """Consumer that declares the mode key but resolves a child WITHOUT the key."""

    FEATURE_NAME = "dualwarn579_clean_consumer"
    SOURCE_COLUMN = CLEAN_SOURCE_NAME
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 clean consumer group", context=False),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(CLEAN_SOURCE_NAME)}


class DualWarn579ExcludeConsumerGroup(_DualWarn579ConsumerBase):
    """Consumer that declares the mode key but carves it out of forwarding."""

    FEATURE_NAME = "dualwarn579_exclude_consumer"
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 exclude consumer group", context=False),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, forward_group_exclude={MODE_KEY})}


# Holder for ONE child Feature instance shared by BOTH shared-consumer groups below.
# The test assigns a fresh Feature before each run, so no state leaks between tests
# (xdist workers are separate processes; only the shared-child test touches this).
_dualwarn579_shared_child: dict[str, Feature] = {}


class DualWarn579SharedConsumerAGroup(_DualWarn579ConsumerBase):
    """First consumer declaring the mode key; declares the SAME child instance as B."""

    FEATURE_NAME = "dualwarn579_shared_consumer_a"
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 shared consumer A group", context=False),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {_dualwarn579_shared_child["child"]}


class DualWarn579SharedConsumerBGroup(_DualWarn579ConsumerBase):
    """Second consumer declaring the mode key; declares the SAME child instance as A."""

    FEATURE_NAME = "dualwarn579_shared_consumer_b"
    PROPERTY_MAPPING = {
        MODE_KEY: PropertySpec("Execution mode consumed by the dualwarn579 shared consumer B group", context=False),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {_dualwarn579_shared_child["child"]}


class DualWarn579UndeclaredConsumerGroup(_DualWarn579ConsumerBase):
    """Consumer WITHOUT any PROPERTY_MAPPING; its child source DOES declare the key."""

    FEATURE_NAME = "dualwarn579_undeclared_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME)}


def _frame_with_column(results: list[Any], column: str) -> Any:
    for frame in results:
        if column in frame.columns:
            return frame
    raise AssertionError(f"Column '{column}' not found in any result frame: {[list(r.columns) for r in results]}")


def _run(features: list[Feature | str], groups: set[type[FeatureGroup]]) -> list[Any]:
    return mloda.run_all(
        features,
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


def _mode_warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records if record.levelno == logging.WARNING and MODE_KEY in record.getMessage()
    ]


class TestDualOptionConsumptionWarning:
    def test_warning_fires_on_dual_declared_inherited_key(self, caplog: pytest.LogCaptureFixture) -> None:
        """Inherited key declared by BOTH consumer and child triggers exactly one WARNING."""
        DualWarn579SourceGroup.reset_recording()

        consumer = Feature(
            DualWarn579ConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run([consumer], {DualWarn579SourceGroup, DualWarn579ConsumerGroup})

        # The run itself must succeed: the forwarded mode doubles the source values.
        frame = _frame_with_column(results, DualWarn579ConsumerGroup.FEATURE_NAME)
        assert frame[DualWarn579ConsumerGroup.FEATURE_NAME].tolist() == [2, 4, 6]

        records = _mode_warning_records(caplog)
        assert len(records) == 1, (
            f"Expected exactly one WARNING mentioning '{MODE_KEY}', got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "DualWarn579ConsumerGroup" in message
        assert "DualWarn579SourceGroup" in message
        assert "forward_group_exclude" in message

    def test_no_warning_when_child_group_does_not_declare_key(self, caplog: pytest.LogCaptureFixture) -> None:
        """No WARNING when the resolved child group does not declare the key."""
        consumer = Feature(
            DualWarn579CleanConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run([consumer], {DualWarn579CleanSourceGroup, DualWarn579CleanConsumerGroup})

        frame = _frame_with_column(results, DualWarn579CleanConsumerGroup.FEATURE_NAME)
        assert frame[DualWarn579CleanConsumerGroup.FEATURE_NAME].tolist() == [2, 4, 6]

        assert _mode_warning_records(caplog) == []

    def test_no_warning_when_key_is_excluded_from_forwarding(self, caplog: pytest.LogCaptureFixture) -> None:
        """No WARNING when forward_group_exclude keeps the key from being inherited."""
        DualWarn579SourceGroup.reset_recording()

        consumer = Feature(
            DualWarn579ExcludeConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run([consumer], {DualWarn579SourceGroup, DualWarn579ExcludeConsumerGroup})

        frame = _frame_with_column(results, DualWarn579ExcludeConsumerGroup.FEATURE_NAME)
        assert frame[DualWarn579ExcludeConsumerGroup.FEATURE_NAME].tolist() == [2, 4, 6]

        assert _mode_warning_records(caplog) == []

        # The exclusion must hold end to end: the source never received the key.
        assert len(DualWarn579SourceGroup.seen_options) == 1
        upstream = DualWarn579SourceGroup.seen_options[0]
        assert MODE_KEY not in upstream["group"]
        assert MODE_KEY not in upstream["context"]

    def test_no_warning_when_consumer_does_not_declare_key(self, caplog: pytest.LogCaptureFixture) -> None:
        """No WARNING when only the child group declares the key, not the consumer."""
        DualWarn579SourceGroup.reset_recording()

        consumer = Feature(
            DualWarn579UndeclaredConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run([consumer], {DualWarn579SourceGroup, DualWarn579UndeclaredConsumerGroup})

        frame = _frame_with_column(results, DualWarn579UndeclaredConsumerGroup.FEATURE_NAME)
        assert frame[DualWarn579UndeclaredConsumerGroup.FEATURE_NAME].tolist() == [2, 4, 6]

        assert _mode_warning_records(caplog) == []


class TestDualOptionConsumptionWarningDedup:
    """
    Pins the warning dedup contract: within ONE engine run, at most one WARNING per
    (consumer feature group class, resolved child feature group class, overlapping
    key set) triple, even when that triple is processed multiple times (fan-in).
    Distinct consumer classes still warn separately.
    """

    def test_warning_fires_once_for_repeated_consumer_child_pair(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two consumer features of the SAME class with the same dual-declared child warn only ONCE."""
        DualWarn579SourceGroup.reset_recording()

        # The salt is consumer-only bookkeeping (not in any PROPERTY_MAPPING). It makes
        # the two requested features and their forwarded child options distinct, so the
        # (consumer class, child class, {MODE_KEY}) triple is genuinely processed twice.
        consumer_a = Feature(
            DualWarn579RepeatConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast", SALT_KEY: "a"}),
        )
        consumer_b = Feature(
            DualWarn579RepeatConsumerGroup.SECOND_FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast", SALT_KEY: "b"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run([consumer_a, consumer_b], {DualWarn579SourceGroup, DualWarn579RepeatConsumerGroup})

        # The run itself must succeed for both requested consumer features.
        frame_a = _frame_with_column(results, DualWarn579RepeatConsumerGroup.FEATURE_NAME)
        assert frame_a[DualWarn579RepeatConsumerGroup.FEATURE_NAME].tolist() == [2, 4, 6]
        frame_b = _frame_with_column(results, DualWarn579RepeatConsumerGroup.SECOND_FEATURE_NAME)
        assert frame_b[DualWarn579RepeatConsumerGroup.SECOND_FEATURE_NAME].tolist() == [2, 4, 6]

        records = _mode_warning_records(caplog)
        assert len(records) == 1, (
            f"Expected exactly ONE WARNING for the repeated "
            f"(DualWarn579RepeatConsumerGroup, DualWarn579SourceGroup, {{'{MODE_KEY}'}}) triple, "
            f"got {len(records)}: {[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "DualWarn579RepeatConsumerGroup" in message
        assert "DualWarn579SourceGroup" in message

    def test_distinct_consumer_classes_still_warn_separately(self, caplog: pytest.LogCaptureFixture) -> None:
        """Dedup keys on the triple: two consumer CLASSES yield two warnings, repeats within a class none extra."""
        DualWarn579SourceGroup.reset_recording()

        plain_consumer = Feature(
            DualWarn579ConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        repeat_a = Feature(
            DualWarn579RepeatConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast", SALT_KEY: "a"}),
        )
        repeat_b = Feature(
            DualWarn579RepeatConsumerGroup.SECOND_FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast", SALT_KEY: "b"}),
        )
        with caplog.at_level(logging.WARNING):
            _run(
                [plain_consumer, repeat_a, repeat_b],
                {DualWarn579SourceGroup, DualWarn579ConsumerGroup, DualWarn579RepeatConsumerGroup},
            )

        records = _mode_warning_records(caplog)
        assert len(records) == 2, (
            f"Expected exactly TWO WARNINGs (one per consumer class), got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        messages = "\n".join(record.getMessage() for record in records)
        assert "DualWarn579ConsumerGroup" in messages
        assert "DualWarn579RepeatConsumerGroup" in messages


class TestDualOptionConsumptionWarningSharedChildInstance:
    """
    Pins consumer attribution accumulation end to end: ONE shared child Feature
    instance declared by TWO distinct consumer feature groups (both declaring the
    same PROPERTY_MAPPING key, forwarding EQUAL values so there is no conflict)
    must produce TWO warnings, one naming each consumer. The second consumer's
    merge onto the shared Options instance must not erase the first consumer's
    provenance (inherited_group_keys union, consumer_attributions append), and
    the dedup on the (consumer class, child class, keys) triple must treat the
    two consumers as two distinct triples.
    """

    def test_shared_child_instance_warns_once_per_consumer(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two consumer classes sharing ONE child Feature instance warn once EACH."""
        DualWarn579SourceGroup.reset_recording()
        _dualwarn579_shared_child["child"] = Feature(SOURCE_NAME)

        consumer_a = Feature(
            DualWarn579SharedConsumerAGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        consumer_b = Feature(
            DualWarn579SharedConsumerBGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run(
                [consumer_a, consumer_b],
                {DualWarn579SourceGroup, DualWarn579SharedConsumerAGroup, DualWarn579SharedConsumerBGroup},
            )

        # The run itself must succeed: equal forwarded values never conflict.
        frame_a = _frame_with_column(results, DualWarn579SharedConsumerAGroup.FEATURE_NAME)
        assert frame_a[DualWarn579SharedConsumerAGroup.FEATURE_NAME].tolist() == [2, 4, 6]
        frame_b = _frame_with_column(results, DualWarn579SharedConsumerBGroup.FEATURE_NAME)
        assert frame_b[DualWarn579SharedConsumerBGroup.FEATURE_NAME].tolist() == [2, 4, 6]

        records = _mode_warning_records(caplog)
        assert len(records) == 2, (
            f"Expected exactly TWO WARNINGs for the shared child instance (one per consumer "
            f"class), got {len(records)}: {[record.getMessage() for record in records]}"
        )
        messages = "\n".join(record.getMessage() for record in records)
        assert "DualWarn579SharedConsumerAGroup" in messages
        assert "DualWarn579SharedConsumerBGroup" in messages
        for record in records:
            assert "DualWarn579SourceGroup" in record.getMessage()


class TestDualOptionConsumptionWarningForwardingAttribution:
    """
    False-positive regression: ONE shared child Feature instance declared by TWO
    consumer feature groups that BOTH declare the dual-declared key in their
    PROPERTY_MAPPING, but only ONE of which actually FORWARDS it (the other carries
    no value for the key), must warn ONLY the forwarding consumer.

    The buggy engine intersected each consumer's PROPERTY_MAPPING keys with the
    child's GLOBAL inherited_group_keys union, so a key forwarded by consumer A
    lit up consumer B's warning too (B declares the key but never forwarded it).
    The fixed engine attributes the warning from each consumer's OWN forwarded set
    (Options.last_forwarded_group_keys intersected with that consumer's
    PROPERTY_MAPPING keys), so the non-forwarding consumer is silent.
    """

    def test_only_forwarding_consumer_of_shared_child_is_warned(self, caplog: pytest.LogCaptureFixture) -> None:
        """A shared child forwarded by A (carries mode) but not by B (no mode) warns ONLY A."""
        DualWarn579SourceGroup.reset_recording()
        _dualwarn579_shared_child["child"] = Feature(SOURCE_NAME)

        # Consumer A carries the mode key, so it forwards it onto the shared child.
        consumer_a = Feature(
            DualWarn579SharedConsumerAGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        # Consumer B carries NO mode key, so it forwards nothing onto the shared child,
        # even though it declares the mode key in its PROPERTY_MAPPING.
        consumer_b = Feature(DualWarn579SharedConsumerBGroup.FEATURE_NAME)

        with caplog.at_level(logging.WARNING):
            results = _run(
                [consumer_a, consumer_b],
                {DualWarn579SourceGroup, DualWarn579SharedConsumerAGroup, DualWarn579SharedConsumerBGroup},
            )

        # A doubles the source (its own mode is "fast"); B leaves it unchanged (no mode).
        frame_a = _frame_with_column(results, DualWarn579SharedConsumerAGroup.FEATURE_NAME)
        assert frame_a[DualWarn579SharedConsumerAGroup.FEATURE_NAME].tolist() == [2, 4, 6]
        frame_b = _frame_with_column(results, DualWarn579SharedConsumerBGroup.FEATURE_NAME)
        assert frame_b[DualWarn579SharedConsumerBGroup.FEATURE_NAME].tolist() == [1, 2, 3]

        records = _mode_warning_records(caplog)
        assert len(records) == 1, (
            f"Expected exactly ONE WARNING naming only the forwarding consumer A, got "
            f"{len(records)}: {[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "DualWarn579SharedConsumerAGroup" in message
        assert "DualWarn579SourceGroup" in message
        # The non-forwarding consumer B must NOT be warned.
        assert "DualWarn579SharedConsumerBGroup" not in message


class TestDualOptionConsumptionWarningStringChild:
    """
    Pins the dual-consumption warning for the BARE-STRING input-feature path (#620).

    A bare-string child (input_features returning {SOURCE_NAME}, not {Feature(...)})
    now runs the real inherit_from merge on its own fresh Options and records the same
    forwarding provenance as an object child. The dual-consumption warning must fire
    for it exactly as it does for an object child.
    """

    def test_warning_fires_for_bare_string_input_feature(self, caplog: pytest.LogCaptureFixture) -> None:
        """A bare-string child that inherits a dual-declared key triggers exactly one WARNING."""
        DualWarn579SourceGroup.reset_recording()

        consumer = Feature(
            DualWarn579StringConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            results = _run([consumer], {DualWarn579SourceGroup, DualWarn579StringConsumerGroup})

        # The run itself must succeed for the requested consumer feature.
        frame = _frame_with_column(results, DualWarn579StringConsumerGroup.FEATURE_NAME)
        assert frame[DualWarn579StringConsumerGroup.FEATURE_NAME].tolist() == [2, 4, 6]

        records = _mode_warning_records(caplog)
        assert len(records) == 1, (
            f"Expected exactly one WARNING mentioning '{MODE_KEY}', got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "DualWarn579StringConsumerGroup" in message
        assert "DualWarn579SourceGroup" in message
        assert "forward_group_exclude" in message

        # The warning plus this seen_options check are the proof that the bare-string
        # child inherited the forwarded key: the mode key actually reached the upstream
        # source (the [2, 4, 6] value alone comes from the consumer's OWN mode).
        assert len(DualWarn579SourceGroup.seen_options) == 1
        assert MODE_KEY in DualWarn579SourceGroup.seen_options[0]["group"]

    def test_bare_string_child_warns_and_forwards_but_wrapped_feature_exclude_decouples(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bare-string input feature carries no per-child opt-out: it warns and forwards
        the dual-declared key to the upstream source. The documented decouple is wrapping
        it as Feature(name, forward_group_exclude={...}), which then warns not at all and
        withholds the key. Both behaviors are contrasted on the SAME upstream seen_options.
        """
        # Bare-string child: no per-child opt-out, so it warns AND forwards the key.
        DualWarn579SourceGroup.reset_recording()
        bare_consumer = Feature(
            DualWarn579StringConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            _run([bare_consumer], {DualWarn579SourceGroup, DualWarn579StringConsumerGroup})

        assert len(_mode_warning_records(caplog)) == 1
        assert len(DualWarn579SourceGroup.seen_options) == 1
        assert MODE_KEY in DualWarn579SourceGroup.seen_options[0]["group"]

        # Wrapped child with forward_group_exclude: no warning AND the key is withheld.
        caplog.clear()
        DualWarn579SourceGroup.reset_recording()
        wrapped_consumer = Feature(
            DualWarn579ExcludeConsumerGroup.FEATURE_NAME,
            options=Options(group={MODE_KEY: "fast"}),
        )
        with caplog.at_level(logging.WARNING):
            _run([wrapped_consumer], {DualWarn579SourceGroup, DualWarn579ExcludeConsumerGroup})

        assert _mode_warning_records(caplog) == []
        assert len(DualWarn579SourceGroup.seen_options) == 1
        upstream = DualWarn579SourceGroup.seen_options[0]
        assert MODE_KEY not in upstream["group"]
        assert MODE_KEY not in upstream["context"]
