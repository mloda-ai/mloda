"""Warn about group-scoped options lost before their resolved consumers (#2942)."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.property_spec import property_spec
from mloda.core.core.engine import Engine
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


KEY = "pii_keep_clear"
ROUTE = "warning_2942_route"
PRIVATE_VALUE = "PRIVATE_OPTION_VALUE_2942"
OTHER_PRIVATE_VALUE = "OTHER_PRIVATE_OPTION_VALUE_2942"
GROUP_ROOT = "warning_2942_group_root"
PLAIN_ROOT = "warning_2942_plain_root"
CONTEXT_ROOT = "warning_2942_context_root"
UNUSED_ROOT = "warning_2942_unused_root"
DIRECT = "warning_2942_direct"
STRING_DIRECT = "warning_2942_string_direct"
RELAY = "warning_2942_relay"
DEEP = "warning_2942_deep"
PULL_DIRECT = "warning_2942_pull_direct"
PULL_RELAY = "warning_2942_pull_relay"
PULL_DEEP = "warning_2942_pull_deep"
MANUAL_DIRECT = "warning_2942_manual_direct"
PLAIN = "warning_2942_plain"
CONTEXT_DIRECT = "warning_2942_context_direct"
PII_MIDDLE = "warning_2942_pii_middle"
PII_TOP = "warning_2942_pii_top"
OWN_GROUP = "warning_2942_own_group"
OWN_CONTEXT = "warning_2942_own_context"
SELECTOR = "warning_2942_selector"


class Warning2942PlainRoot(DataCreatorFeatureGroup):
    NAME: ClassVar[str] = PLAIN_ROOT

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.NAME}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): [feature.options.get(KEY)] for feature in features.features}


class Warning2942GroupRoot(Warning2942PlainRoot):
    NAME = GROUP_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Values to keep clear", context=False, default=None)}


class Warning2942ContextRoot(Warning2942PlainRoot):
    NAME = CONTEXT_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Local metadata", context=True, default=None)}


class Warning2942UnusedRoot(Warning2942GroupRoot):
    NAME = UNUSED_ROOT


class Warning2942Consumer(FeatureGroup):
    INPUTS: ClassVar[dict[str, str]] = {
        DIRECT: GROUP_ROOT,
        STRING_DIRECT: GROUP_ROOT,
        RELAY: GROUP_ROOT,
        DEEP: RELAY,
        PULL_DIRECT: GROUP_ROOT,
        PULL_RELAY: GROUP_ROOT,
        PULL_DEEP: PULL_RELAY,
        MANUAL_DIRECT: GROUP_ROOT,
        PLAIN: PLAIN_ROOT,
        CONTEXT_DIRECT: CONTEXT_ROOT,
        PII_TOP: PII_MIDDLE,
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(cls.INPUTS)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        name = str(feature_name)
        child = self.INPUTS[name]
        if name == STRING_DIRECT:
            return {child}
        if name in {PULL_DIRECT, PULL_RELAY, PULL_DEEP}:
            return {Feature(child, inherit_context_keys={KEY})}
        if name == MANUAL_DIRECT:
            return {Feature(child, Options(context={KEY: options.get(KEY)}))}
        return {Feature(child)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise AssertionError("Plan-time diagnostics must not execute a feature")


class Warning2942PiiMiddle(Warning2942Consumer):
    INPUTS = {PII_MIDDLE: PLAIN_ROOT}
    PROPERTY_MAPPING = {KEY: property_spec("Values consumed by the detection input", context=False, default=None)}


class Warning2942OwnGroup(Warning2942Consumer):
    INPUTS = {OWN_GROUP: GROUP_ROOT}
    PROPERTY_MAPPING = {KEY: property_spec("Requested group's own option", context=False, default=None)}


class Warning2942OwnContext(Warning2942Consumer):
    INPUTS = {OWN_CONTEXT: GROUP_ROOT}
    PROPERTY_MAPPING = {KEY: property_spec("Requested group's local metadata", context=True, default=None)}


class Warning2942Selector(Warning2942Consumer):
    INPUTS = {SELECTOR: PLAIN_ROOT}
    PROPERTY_MAPPING = {ROUTE: property_spec("Choose an input", context=True, default="plain")}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(GROUP_ROOT if options.get(ROUTE) == "group" else PLAIN_ROOT)}


GROUPS: set[type[FeatureGroup]] = {
    Warning2942PlainRoot,
    Warning2942GroupRoot,
    Warning2942ContextRoot,
    Warning2942UnusedRoot,
    Warning2942Consumer,
    Warning2942PiiMiddle,
    Warning2942OwnGroup,
    Warning2942OwnContext,
    Warning2942Selector,
}


def _prepare(requests: Sequence[Feature | str]) -> Engine:
    session = mloda.prepare(
        requests,
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups(GROUPS),
    )
    assert session.engine is not None
    return session.engine


def _feature(engine: Engine, name: str) -> Feature:
    matches = [
        feature for group in engine.feature_group_collection.values() for feature in group if str(feature.name) == name
    ]
    assert len(matches) == 1
    return matches[0]


def _warnings(caplog: pytest.LogCaptureFixture, key: str = KEY) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING and key in record.getMessage()]


def _assert_actionable_warning(caplog: pytest.LogCaptureFixture) -> None:
    records = _warnings(caplog)
    assert records, "An uncascaded group-scoped key must warn at plan time"
    messages = "\n".join(record.getMessage() for record in records)
    assert "group=" in messages
    assert "cascade_context_keys" in messages
    assert PRIVATE_VALUE not in caplog.text
    assert OTHER_PRIVATE_VALUE not in caplog.text


@pytest.mark.parametrize("requested", [DIRECT, STRING_DIRECT, DEEP])
@pytest.mark.parametrize("value", [PRIVATE_VALUE, [PRIVATE_VALUE, OTHER_PRIVATE_VALUE]], ids=["scalar", "list"])
def test_uncascaded_key_warns_before_execution(requested: str, value: Any, caplog: pytest.LogCaptureFixture) -> None:
    options = Options(context={KEY: value, "unrelated_metadata": OTHER_PRIVATE_VALUE})
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, options)])
    assert _feature(engine, requested).options.get(KEY) == value
    assert _feature(engine, GROUP_ROOT).options.get(KEY) is None
    _assert_actionable_warning(caplog)


def test_pii_like_declaration_above_the_reader_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(PII_TOP, Options(context={KEY: [PRIVATE_VALUE]}))])
    assert _feature(engine, PII_MIDDLE).options.get(KEY) is None
    assert _feature(engine, PLAIN_ROOT).options.get(KEY) is None
    _assert_actionable_warning(caplog)


def test_one_hop_push_stopping_before_the_declaring_grandchild_warns(caplog: pytest.LogCaptureFixture) -> None:
    options = Options(context={KEY: PRIVATE_VALUE}, propagate_context_keys=frozenset({KEY}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(DEEP, options)])
    assert _feature(engine, RELAY).options.get(KEY) == PRIVATE_VALUE
    assert _feature(engine, GROUP_ROOT).options.get(KEY) is None
    _assert_actionable_warning(caplog)


def test_only_the_selected_declaring_dependency_triggers_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(SELECTOR, Options(context={KEY: PRIVATE_VALUE, ROUTE: "group"}))])
    assert Warning2942GroupRoot in engine.feature_group_collection
    assert Warning2942PlainRoot not in engine.feature_group_collection
    _assert_actionable_warning(caplog)


@pytest.mark.parametrize("requested", [DIRECT, STRING_DIRECT, DEEP, PII_TOP])
@pytest.mark.parametrize("mode", ["group", "cascade"])
def test_group_and_cascade_delivery_are_quiet(requested: str, mode: str, caplog: pytest.LogCaptureFixture) -> None:
    options = (
        Options(group={KEY: PRIVATE_VALUE})
        if mode == "group"
        else Options(context={KEY: PRIVATE_VALUE}, cascade_context_keys=frozenset({KEY}))
    )
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, options)])
    root = PLAIN_ROOT if requested == PII_TOP else GROUP_ROOT
    assert _feature(engine, root).options.get(KEY) == PRIVATE_VALUE
    assert _warnings(caplog) == []


@pytest.mark.parametrize("requested", [PULL_DIRECT, PULL_DEEP, MANUAL_DIRECT])
def test_delivered_pull_or_explicit_input_context_is_quiet(requested: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, GROUP_ROOT).options.get(KEY) == PRIVATE_VALUE
    assert _warnings(caplog) == []


def test_one_hop_push_to_the_actual_consumer_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    options = Options(context={KEY: PRIVATE_VALUE}, propagate_context_keys=frozenset({KEY}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(DIRECT, options)])
    assert _feature(engine, GROUP_ROOT).options.get(KEY) == PRIVATE_VALUE
    assert _warnings(caplog) == []


@pytest.mark.parametrize("requested", [OWN_GROUP, OWN_CONTEXT])
def test_requested_group_consuming_the_key_is_quiet(requested: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, requested).options.get(KEY) == PRIVATE_VALUE
    assert _feature(engine, GROUP_ROOT).options.get(KEY) is None
    assert _warnings(caplog) == []


def test_unrelated_available_group_does_not_trigger_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(PLAIN, Options(context={KEY: PRIVATE_VALUE}))])
    assert Warning2942GroupRoot not in engine.feature_group_collection
    assert Warning2942UnusedRoot not in engine.feature_group_collection
    assert _warnings(caplog) == []


@pytest.mark.parametrize("unrelated_first", [False, True])
def test_declaring_group_in_an_unrelated_requested_branch_is_quiet(
    unrelated_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    requests = [Feature(PLAIN, Options(context={KEY: PRIVATE_VALUE})), Feature(GROUP_ROOT)]
    with caplog.at_level(logging.WARNING):
        engine = _prepare(requests[::-1] if unrelated_first else requests)
    assert Warning2942GroupRoot in engine.feature_group_collection
    assert _feature(engine, PLAIN_ROOT).options.get(KEY) is None
    assert _warnings(caplog) == []


def test_unselected_declaring_input_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(SELECTOR, Options(context={KEY: PRIVATE_VALUE, ROUTE: "plain"}))])
    assert Warning2942PlainRoot in engine.feature_group_collection
    assert Warning2942GroupRoot not in engine.feature_group_collection
    assert _warnings(caplog) == []


def test_downstream_context_only_declaration_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(CONTEXT_DIRECT, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, CONTEXT_ROOT).options.get(KEY) is None
    assert _warnings(caplog) == []


def test_undeclared_context_metadata_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    key = "unconsumed_metadata_2942"
    with caplog.at_level(logging.WARNING):
        _prepare([Feature(DEEP, Options(context={key: PRIVATE_VALUE}))])
    assert _warnings(caplog, key) == []


@pytest.mark.parametrize("direct_first", [False, True])
def test_forwarding_to_an_equal_direct_root_is_quiet(direct_first: bool, caplog: pytest.LogCaptureFixture) -> None:
    requests = [
        Feature(GROUP_ROOT, Options(context={KEY: PRIVATE_VALUE})),
        Feature(PULL_DIRECT, Options(context={KEY: PRIVATE_VALUE})),
    ]
    with caplog.at_level(logging.WARNING):
        engine = _prepare(requests if direct_first else requests[::-1])
    root = _feature(engine, GROUP_ROOT)
    assert root.initial_requested_data is True
    assert root.options.get(KEY) == PRIVATE_VALUE
    assert engine.feature_link_parents[_feature(engine, PULL_DIRECT).uuid] == {root.uuid}
    assert _warnings(caplog) == []


FOLLOWUP_DEFAULT = "DEFAULT_OPTION_VALUE_2942"
FOLLOWUP_OVERRIDE = "EXPLICIT_OVERRIDE_VALUE_2942"
DEFAULT_ROOT = "warning_2942_default_root"
DEFAULT_CONSUMER = "warning_2942_default_consumer"
DEFAULT_OVERRIDE = "warning_2942_default_override"
INHERITED_OVERRIDE = "warning_2942_inherited_override"
DEFAULT_RELAY = "warning_2942_default_relay"
OVERRIDE_GROUP = "warning_2942_override_group"
OVERRIDE_CONTEXT = "warning_2942_override_context"
PARTIAL_DELIVERY = "warning_2942_partial_delivery"
PARTIAL_DEFAULT = "warning_2942_partial_default"


class Warning2942DefaultRoot(Warning2942PlainRoot):
    NAME = DEFAULT_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Defaulted group option", context=False, default=FOLLOWUP_DEFAULT)}


class Warning2942DefaultConsumer(Warning2942Consumer):
    INPUTS = {DEFAULT_CONSUMER: DEFAULT_ROOT, DEFAULT_RELAY: DEFAULT_ROOT}


class Warning2942DefaultOverride(Warning2942Consumer):
    INPUTS = {DEFAULT_OVERRIDE: DEFAULT_ROOT, INHERITED_OVERRIDE: DEFAULT_RELAY}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(self.INPUTS[str(feature_name)], Options(group={KEY: FOLLOWUP_DEFAULT}))}


class Warning2942OverrideConsumer(Warning2942Consumer):
    INPUTS = {OVERRIDE_GROUP: DEFAULT_ROOT, OVERRIDE_CONTEXT: DEFAULT_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        override = (
            Options(group={KEY: FOLLOWUP_OVERRIDE})
            if str(feature_name) == OVERRIDE_GROUP
            else Options(context={KEY: FOLLOWUP_OVERRIDE})
        )
        return {Feature(DEFAULT_ROOT, override)}


class Warning2942PartialDelivery(Warning2942Consumer):
    INPUTS = {PARTIAL_DELIVERY: UNUSED_ROOT, PARTIAL_DEFAULT: DEFAULT_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {
            Feature(GROUP_ROOT, inherit_context_keys={KEY}),
            Feature(self.INPUTS[str(feature_name)]),
        }


def _prepare_followup(
    requests: Sequence[Feature | str],
    joined_source: type[Warning2942PlainRoot] | None = None,
    broader_frameworks: bool = False,
) -> Engine:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.user import JoinSpec, Link
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    frameworks: list[type[ComputeFramework]] = [PythonDictFramework]
    if broader_frameworks:
        frameworks.append(PyArrowTable)
    links = (
        None
        if joined_source is None
        else {Link.inner(JoinSpec(Warning2942GroupRoot, GROUP_ROOT), JoinSpec(joined_source, joined_source.NAME))}
    )
    session = mloda.prepare(
        requests,
        compute_frameworks=frameworks,
        links=links,
        plugin_collector=PluginCollector.enabled_feature_groups(
            GROUPS
            | {
                Warning2942DefaultRoot,
                Warning2942DefaultConsumer,
                Warning2942DefaultOverride,
                Warning2942OverrideConsumer,
                Warning2942PartialDelivery,
            }
        ),
    )
    assert session.engine is not None
    return session.engine


def _followup_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in _warnings(caplog) if "default-equivalent options" not in record.getMessage()]


def _assert_followup_warning(caplog: pytest.LogCaptureFixture) -> None:
    records = _followup_warnings(caplog)
    assert records, "A resolved consumer retaining a different default must still warn"
    message = "\n".join(record.getMessage() for record in records)
    assert "group=" in message
    assert "cascade_context_keys" in message
    assert all(value not in message for value in (PRIVATE_VALUE, FOLLOWUP_DEFAULT, FOLLOWUP_OVERRIDE))


def test_followup_non_null_default_does_not_count_as_delivery(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare_followup([Feature(DEFAULT_CONSUMER, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == FOLLOWUP_DEFAULT
    _assert_followup_warning(caplog)


def test_followup_equal_effective_default_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare_followup([Feature(DEFAULT_CONSUMER, Options(context={KEY: FOLLOWUP_DEFAULT}))])
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == FOLLOWUP_DEFAULT
    assert _followup_warnings(caplog) == []


@pytest.mark.parametrize("requested", [OVERRIDE_GROUP, OVERRIDE_CONTEXT])
def test_followup_explicit_child_override_is_quiet(requested: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare_followup([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == FOLLOWUP_OVERRIDE
    assert _followup_warnings(caplog) == []


@pytest.mark.parametrize("caller_value", [PRIVATE_VALUE, FOLLOWUP_DEFAULT], ids=["different-default", "equal-default"])
@pytest.mark.parametrize("requested", [DEFAULT_CONSUMER, DEFAULT_OVERRIDE, INHERITED_OVERRIDE])
@pytest.mark.parametrize("direct_first", [False, True], ids=["dependency-first", "direct-first"])
def test_followup_folded_default_keeps_the_requesting_path_diagnostic(
    caller_value: str, requested: str, direct_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    direct = Feature(DEFAULT_ROOT)
    consumer = Feature(requested, Options(context={KEY: caller_value}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare_followup([direct, consumer] if direct_first else [consumer, direct])
    root = _feature(engine, DEFAULT_ROOT)
    assert root.initial_requested_data is True
    assert root.options.get(KEY) == FOLLOWUP_DEFAULT
    parent = DEFAULT_RELAY if requested == INHERITED_OVERRIDE else requested
    assert engine.feature_link_parents[_feature(engine, parent).uuid] == {root.uuid}
    if requested == DEFAULT_CONSUMER and caller_value != FOLLOWUP_DEFAULT:
        _assert_followup_warning(caplog)
    else:
        assert _followup_warnings(caplog) == []


@pytest.mark.parametrize("requested", [PARTIAL_DELIVERY, PARTIAL_DEFAULT])
def test_followup_delivery_to_one_branch_does_not_hide_the_other(
    requested: str, caplog: pytest.LogCaptureFixture
) -> None:
    joined_source = Warning2942UnusedRoot if requested == PARTIAL_DELIVERY else Warning2942DefaultRoot
    with caplog.at_level(logging.WARNING):
        engine = _prepare_followup([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))], joined_source)
    assert _feature(engine, GROUP_ROOT).options.get(KEY) == PRIVATE_VALUE
    missing = UNUSED_ROOT if requested == PARTIAL_DELIVERY else DEFAULT_ROOT
    assert _feature(engine, missing).options.get(KEY) == (None if missing == UNUSED_ROOT else FOLLOWUP_DEFAULT)
    _assert_followup_warning(caplog)


@pytest.mark.parametrize("requested", [DEFAULT_OVERRIDE, INHERITED_OVERRIDE])
@pytest.mark.parametrize("direct_first", [False, True], ids=["dependency-first", "direct-first"])
def test_followup_late_narrower_fold_keeps_configured_default_override(
    requested: str, direct_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    from unittest.mock import patch

    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    direct = Feature(DEFAULT_ROOT, compute_framework=PythonDictFramework.get_class_name())
    consumer = Feature(requested, Options(context={KEY: PRIVATE_VALUE}))
    requests = [direct, consumer] if direct_first else [consumer, direct]
    with (
        caplog.at_level(logging.WARNING),
        patch.object(Engine, "_merge_host_into", autospec=True, side_effect=Engine._merge_host_into) as merge_host,
    ):
        engine = _prepare_followup(requests, broader_frameworks=True)

    root = _feature(engine, DEFAULT_ROOT)
    root_folds = [
        (call.args[1], call.args[2])
        for call in merge_host.call_args_list
        if str(call.args[1].name) == DEFAULT_ROOT and str(call.args[2].name) == DEFAULT_ROOT
    ]
    assert len(root_folds) == 1, "The real late narrower-framework merge must run"
    survivor, host = root_folds[0]
    assert survivor is root
    assert survivor.compute_frameworks == {PythonDictFramework}
    assert host.compute_frameworks == {PythonDictFramework, PyArrowTable}
    assert root.initial_requested_data is True
    assert root.options.get(KEY) == FOLLOWUP_DEFAULT
    assert engine._declared_options_by_uuid[root.uuid].get(KEY) is None
    parent = DEFAULT_RELAY if requested == INHERITED_OVERRIDE else requested
    assert engine.feature_link_parents[_feature(engine, parent).uuid] == {root.uuid}
    assert _followup_warnings(caplog) == []
