"""Warn about group-scoped options lost before their resolved consumers.

One warning per plan names the lost keys, the requested features and the input groups,
and gives the remedy each key allows. Option values never appear in the text.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.credential_handle import credential_handle_spec
from mloda.core.abstract_plugins.components.property_spec import property_spec
from mloda.core.core.engine import Engine
from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import (
    Feature,
    FeatureName,
    FilterType,
    GlobalFilter,
    JoinSpec,
    Link,
    Options,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


KEY = "pii_keep_clear"
ONLY_KEY = "uncascaded_ctx_only_key"
ROUTE = "uncascaded_ctx_route"
PRIVATE_VALUE = "PRIVATE_OPTION_VALUE"
OTHER_PRIVATE_VALUE = "OTHER_PRIVATE_OPTION_VALUE"
GROUP_ROOT = "uncascaded_ctx_group_root"
PLAIN_ROOT = "uncascaded_ctx_plain_root"
CONTEXT_ROOT = "uncascaded_ctx_context_root"
UNUSED_ROOT = "uncascaded_ctx_unused_root"
GROUP_ONLY_ROOT = "uncascaded_ctx_group_only_root"
HANDLE_ROOT = "uncascaded_ctx_handle_root"
MIXED_ROOT = "uncascaded_ctx_mixed_root"
FILTER_ROOT = "uncascaded_ctx_filter_root"
FILTER_COL = "uncascaded_ctx_filter_col"
DIRECT = "uncascaded_ctx_direct"
STRING_DIRECT = "uncascaded_ctx_string_direct"
RELAY = "uncascaded_ctx_relay"
DEEP = "uncascaded_ctx_deep"
PULL_DIRECT = "uncascaded_ctx_pull_direct"
PULL_RELAY = "uncascaded_ctx_pull_relay"
PULL_DEEP = "uncascaded_ctx_pull_deep"
MANUAL_DIRECT = "uncascaded_ctx_manual_direct"
PLAIN = "uncascaded_ctx_plain"
CONTEXT_DIRECT = "uncascaded_ctx_context_direct"
PII_MIDDLE = "uncascaded_ctx_pii_middle"
PII_TOP = "uncascaded_ctx_pii_top"
OWN_GROUP = "uncascaded_ctx_own_group"
OWN_CONTEXT = "uncascaded_ctx_own_context"
SELECTOR = "uncascaded_ctx_selector"
GROUP_ONLY_DIRECT = "uncascaded_ctx_group_only_direct"
HANDLE_DIRECT = "uncascaded_ctx_handle_direct"
MIXED_DIRECT = "uncascaded_ctx_mixed_direct"
FILTER_CONSUMER = "uncascaded_ctx_filter_consumer"


class UncascadedCtxPlainRoot(DataCreatorFeatureGroup):
    NAME: ClassVar[str] = PLAIN_ROOT

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.NAME}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): [feature.options.get(KEY)] for feature in features.features}


class UncascadedCtxGroupRoot(UncascadedCtxPlainRoot):
    NAME = GROUP_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Values to keep clear", context=False, default=None)}


class UncascadedCtxContextRoot(UncascadedCtxPlainRoot):
    NAME = CONTEXT_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Local metadata", context=True, default=None)}


class UncascadedCtxUnusedRoot(UncascadedCtxGroupRoot):
    NAME = UNUSED_ROOT


class UncascadedCtxGroupOnlyRoot(UncascadedCtxPlainRoot):
    NAME = GROUP_ONLY_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Group-only values", context=False, group_only=True, default=None)}


class UncascadedCtxHandleRoot(UncascadedCtxPlainRoot):
    NAME = HANDLE_ROOT
    PROPERTY_MAPPING = {KEY: credential_handle_spec("Handle of a credential", default=None)}


class UncascadedCtxMixedRoot(UncascadedCtxPlainRoot):
    NAME = MIXED_ROOT
    PROPERTY_MAPPING = {
        KEY: property_spec("Cascadable values", context=False, default=None),
        ONLY_KEY: property_spec("Group-only values", context=False, group_only=True, default=None),
    }


class UncascadedCtxFilterRoot(UncascadedCtxPlainRoot):
    NAME = FILTER_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Values to keep clear", context=False, default=None)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.NAME, FILTER_COL}


class UncascadedCtxConsumer(FeatureGroup):
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


class UncascadedCtxPiiMiddle(UncascadedCtxConsumer):
    INPUTS = {PII_MIDDLE: PLAIN_ROOT}
    PROPERTY_MAPPING = {KEY: property_spec("Values consumed by the detection input", context=False, default=None)}


class UncascadedCtxOwnGroup(UncascadedCtxConsumer):
    INPUTS = {OWN_GROUP: GROUP_ROOT}
    PROPERTY_MAPPING = {KEY: property_spec("Requested group's own option", context=False, default=None)}


class UncascadedCtxOwnContext(UncascadedCtxConsumer):
    INPUTS = {OWN_CONTEXT: GROUP_ROOT}
    PROPERTY_MAPPING = {KEY: property_spec("Requested group's local metadata", context=True, default=None)}


class UncascadedCtxSelector(UncascadedCtxConsumer):
    INPUTS = {SELECTOR: PLAIN_ROOT}
    PROPERTY_MAPPING = {ROUTE: property_spec("Choose an input", context=True, default="plain")}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(GROUP_ROOT if options.get(ROUTE) == "group" else PLAIN_ROOT)}


class UncascadedCtxGroupOnlyConsumer(UncascadedCtxConsumer):
    INPUTS = {GROUP_ONLY_DIRECT: GROUP_ONLY_ROOT, HANDLE_DIRECT: HANDLE_ROOT, MIXED_DIRECT: MIXED_ROOT}


class UncascadedCtxFilterConsumer(UncascadedCtxConsumer):
    INPUTS = {FILTER_CONSUMER: FILTER_ROOT}


DECLARED_DEFAULT = "DEFAULT_OPTION_VALUE"
EXPLICIT_OVERRIDE = "EXPLICIT_OVERRIDE_VALUE"
DEFAULT_ROOT = "uncascaded_ctx_default_root"
DEFAULT_CONSUMER = "uncascaded_ctx_default_consumer"
DEFAULT_OVERRIDE = "uncascaded_ctx_default_override"
INHERITED_OVERRIDE = "uncascaded_ctx_inherited_override"
DEFAULT_RELAY = "uncascaded_ctx_default_relay"
OVERRIDE_GROUP = "uncascaded_ctx_override_group"
OVERRIDE_CONTEXT = "uncascaded_ctx_override_context"
PARTIAL_DELIVERY = "uncascaded_ctx_partial_delivery"
PARTIAL_DEFAULT = "uncascaded_ctx_partial_default"


class UncascadedCtxDefaultRoot(UncascadedCtxPlainRoot):
    NAME = DEFAULT_ROOT
    PROPERTY_MAPPING = {KEY: property_spec("Defaulted group option", context=False, default=DECLARED_DEFAULT)}


class UncascadedCtxDefaultConsumer(UncascadedCtxConsumer):
    INPUTS = {DEFAULT_CONSUMER: DEFAULT_ROOT, DEFAULT_RELAY: DEFAULT_ROOT}


class UncascadedCtxDefaultOverride(UncascadedCtxConsumer):
    INPUTS = {DEFAULT_OVERRIDE: DEFAULT_ROOT, INHERITED_OVERRIDE: DEFAULT_RELAY}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(self.INPUTS[str(feature_name)], Options(group={KEY: DECLARED_DEFAULT}))}


class UncascadedCtxOverrideConsumer(UncascadedCtxConsumer):
    INPUTS = {OVERRIDE_GROUP: DEFAULT_ROOT, OVERRIDE_CONTEXT: DEFAULT_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        override = (
            Options(group={KEY: EXPLICIT_OVERRIDE})
            if str(feature_name) == OVERRIDE_GROUP
            else Options(context={KEY: EXPLICIT_OVERRIDE})
        )
        return {Feature(DEFAULT_ROOT, override)}


class UncascadedCtxPartialDelivery(UncascadedCtxConsumer):
    INPUTS = {PARTIAL_DELIVERY: UNUSED_ROOT, PARTIAL_DEFAULT: DEFAULT_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {
            Feature(GROUP_ROOT, inherit_context_keys={KEY}),
            Feature(self.INPUTS[str(feature_name)]),
        }


TWO_ROOTS = "uncascaded_ctx_two_roots"


class UncascadedCtxTwoRoots(UncascadedCtxConsumer):
    INPUTS = {TWO_ROOTS: GROUP_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(GROUP_ROOT), Feature(UNUSED_ROOT)}


DIAMOND = "uncascaded_ctx_diamond"
TWIN_A = "uncascaded_ctx_twin_a"
TWIN_B = "uncascaded_ctx_twin_b"
TWIN_VALUE = "TWIN_GROUP_VALUE"


class UncascadedCtxDiamond(UncascadedCtxConsumer):
    INPUTS = {DIAMOND: DEFAULT_OVERRIDE}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(DEFAULT_OVERRIDE), Feature(DEFAULT_CONSUMER)}


class UncascadedCtxTwinA(UncascadedCtxConsumer):
    INPUTS = {TWIN_A: FILTER_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(FILTER_ROOT, Options(group={KEY: TWIN_VALUE}))}


class UncascadedCtxTwinB(UncascadedCtxConsumer):
    INPUTS = {TWIN_B: FILTER_COL}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {Feature(FILTER_COL, Options(group={KEY: TWIN_VALUE}))}


SHARED_A = "uncascaded_ctx_shared_a"
SHARED_B = "uncascaded_ctx_shared_b"
SHARED_SET_A = "uncascaded_ctx_shared_set_a"
SHARED_SET_B = "uncascaded_ctx_shared_set_b"
SHARED_INPUT = Feature(DEFAULT_ROOT)
SHARED_SET_INPUT = Feature(DEFAULT_ROOT, Options(group={KEY: EXPLICIT_OVERRIDE}))


class UncascadedCtxSharedInput(UncascadedCtxConsumer):
    INPUTS = {SHARED_A: DEFAULT_ROOT, SHARED_B: DEFAULT_ROOT, SHARED_SET_A: DEFAULT_ROOT, SHARED_SET_B: DEFAULT_ROOT}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Any]:
        return {SHARED_INPUT if str(feature_name) in {SHARED_A, SHARED_B} else SHARED_SET_INPUT}


GROUPS: set[type[FeatureGroup]] = {
    UncascadedCtxSharedInput,
    UncascadedCtxDiamond,
    UncascadedCtxTwinA,
    UncascadedCtxTwinB,
    UncascadedCtxTwoRoots,
    UncascadedCtxPlainRoot,
    UncascadedCtxGroupRoot,
    UncascadedCtxContextRoot,
    UncascadedCtxUnusedRoot,
    UncascadedCtxGroupOnlyRoot,
    UncascadedCtxHandleRoot,
    UncascadedCtxMixedRoot,
    UncascadedCtxFilterRoot,
    UncascadedCtxConsumer,
    UncascadedCtxPiiMiddle,
    UncascadedCtxOwnGroup,
    UncascadedCtxOwnContext,
    UncascadedCtxSelector,
    UncascadedCtxGroupOnlyConsumer,
    UncascadedCtxFilterConsumer,
    UncascadedCtxDefaultRoot,
    UncascadedCtxDefaultConsumer,
    UncascadedCtxDefaultOverride,
    UncascadedCtxOverrideConsumer,
    UncascadedCtxPartialDelivery,
}


def _prepare(
    requests: Sequence[Feature | str],
    joined_source: type[UncascadedCtxPlainRoot] | None = None,
    broader_frameworks: bool = False,
    global_filter: GlobalFilter | None = None,
) -> Engine:
    frameworks: list[type[ComputeFramework]] = [PythonDictFramework]
    if broader_frameworks:
        frameworks.append(PyArrowTable)
    links = (
        None
        if joined_source is None
        else {Link.inner(JoinSpec(UncascadedCtxGroupRoot, GROUP_ROOT), JoinSpec(joined_source, joined_source.NAME))}
    )
    session = mloda.prepare(
        requests,
        compute_frameworks=frameworks,
        links=links,
        global_filter=global_filter,
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


def _context_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in _warnings(caplog) if "default-equivalent options" not in record.getMessage()]


def _cascade_remedy(keys: list[str]) -> str:
    return f"Pass {sorted(keys)} in Options(group=...) or list them in cascade_context_keys."


def _group_only_remedy(keys: list[str]) -> str:
    return (
        f"Pass {sorted(keys)} in Options(group=...); a group-only or credential handle key is refused in the context."
    )


def _assert_warning(
    caplog: pytest.LogCaptureFixture,
    requested: list[str],
    groups: list[type[FeatureGroup]],
    keys: list[str] | None = None,
    remedy: str | None = None,
) -> str:
    """Assert the one plan warning with its exact text; return the message."""
    keys = [KEY] if keys is None else keys
    remedy = _cascade_remedy(keys) if remedy is None else remedy
    records = _context_warnings(caplog)
    assert len(records) == 1, "An uncascaded group-scoped key must warn once at plan time"
    assert records[0].name == "mloda.core.core.engine"
    message = records[0].getMessage()
    assert message == (
        f"Context option(s) {sorted(keys)} of requested feature(s) {sorted(requested)} "
        f"do not reach input feature group(s) {sorted(group.get_class_name() for group in groups)}, "
        f"which declare them as group options. {remedy}"
    )
    for value in (PRIVATE_VALUE, OTHER_PRIVATE_VALUE, DECLARED_DEFAULT, EXPLICIT_OVERRIDE):
        assert value not in caplog.text
    return message


@pytest.mark.parametrize(
    "requested, value",
    [
        (DIRECT, PRIVATE_VALUE),
        (STRING_DIRECT, PRIVATE_VALUE),
        (DEEP, PRIVATE_VALUE),
        (DIRECT, [PRIVATE_VALUE, OTHER_PRIVATE_VALUE]),
    ],
    ids=["direct-scalar", "string-direct-scalar", "deep-scalar", "direct-list"],
)
def test_uncascaded_key_warns_before_execution(requested: str, value: Any, caplog: pytest.LogCaptureFixture) -> None:
    options = Options(context={KEY: value, "unrelated_metadata": OTHER_PRIVATE_VALUE})
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, options)])
    assert _feature(engine, requested).options.get(KEY) == value
    assert _feature(engine, GROUP_ROOT).options.get(KEY) is None
    _assert_warning(caplog, [requested], [UncascadedCtxGroupRoot])


def test_pii_like_declaration_above_the_reader_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(PII_TOP, Options(context={KEY: [PRIVATE_VALUE]}))])
    assert _feature(engine, PII_MIDDLE).options.get(KEY) is None
    assert _feature(engine, PLAIN_ROOT).options.get(KEY) is None
    _assert_warning(caplog, [PII_TOP], [UncascadedCtxPiiMiddle])


def test_one_hop_push_stopping_before_the_declaring_grandchild_warns(caplog: pytest.LogCaptureFixture) -> None:
    options = Options(context={KEY: PRIVATE_VALUE}, propagate_context_keys=frozenset({KEY}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(DEEP, options)])
    assert _feature(engine, RELAY).options.get(KEY) == PRIVATE_VALUE
    assert _feature(engine, GROUP_ROOT).options.get(KEY) is None
    _assert_warning(caplog, [DEEP], [UncascadedCtxGroupRoot])


def test_only_the_selected_declaring_dependency_triggers_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(SELECTOR, Options(context={KEY: PRIVATE_VALUE, ROUTE: "group"}))])
    assert UncascadedCtxGroupRoot in engine.feature_group_collection
    assert UncascadedCtxPlainRoot not in engine.feature_group_collection
    _assert_warning(caplog, [SELECTOR], [UncascadedCtxGroupRoot])


@pytest.mark.parametrize("requested", [DEEP, PII_TOP])
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
    assert UncascadedCtxGroupRoot not in engine.feature_group_collection
    assert UncascadedCtxUnusedRoot not in engine.feature_group_collection
    assert _warnings(caplog) == []


@pytest.mark.parametrize("unrelated_first", [False, True])
def test_declaring_group_in_an_unrelated_requested_branch_is_quiet(
    unrelated_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    requests = [Feature(PLAIN, Options(context={KEY: PRIVATE_VALUE})), Feature(GROUP_ROOT)]
    with caplog.at_level(logging.WARNING):
        engine = _prepare(requests[::-1] if unrelated_first else requests)
    assert UncascadedCtxGroupRoot in engine.feature_group_collection
    assert _feature(engine, PLAIN_ROOT).options.get(KEY) is None
    assert _warnings(caplog) == []


def test_unselected_declaring_input_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(SELECTOR, Options(context={KEY: PRIVATE_VALUE, ROUTE: "plain"}))])
    assert UncascadedCtxPlainRoot in engine.feature_group_collection
    assert UncascadedCtxGroupRoot not in engine.feature_group_collection
    assert _warnings(caplog) == []


def test_downstream_context_only_declaration_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(CONTEXT_DIRECT, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, CONTEXT_ROOT).options.get(KEY) is None
    assert _warnings(caplog) == []


def test_undeclared_context_metadata_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    key = "unconsumed_metadata"
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


def test_non_null_default_does_not_count_as_delivery(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(DEFAULT_CONSUMER, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == DECLARED_DEFAULT
    _assert_warning(caplog, [DEFAULT_CONSUMER], [UncascadedCtxDefaultRoot])


def test_equal_effective_default_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(DEFAULT_CONSUMER, Options(context={KEY: DECLARED_DEFAULT}))])
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == DECLARED_DEFAULT
    assert _context_warnings(caplog) == []


@pytest.mark.parametrize("requested", [OVERRIDE_GROUP, OVERRIDE_CONTEXT])
def test_explicit_child_override_is_quiet(requested: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))])
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == EXPLICIT_OVERRIDE
    assert _context_warnings(caplog) == []


@pytest.mark.parametrize(
    "caller_value, requested",
    [(PRIVATE_VALUE, DEFAULT_CONSUMER), (PRIVATE_VALUE, DEFAULT_OVERRIDE), (PRIVATE_VALUE, INHERITED_OVERRIDE)]
    + [(DECLARED_DEFAULT, DEFAULT_CONSUMER)],
    ids=["different-default-consumer", "different-default-override", "different-default-inherited", "equal-default"],
)
@pytest.mark.parametrize("direct_first", [False, True], ids=["dependency-first", "direct-first"])
def test_folded_default_keeps_the_requesting_path_diagnostic(
    caller_value: str, requested: str, direct_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    direct = Feature(DEFAULT_ROOT)
    consumer = Feature(requested, Options(context={KEY: caller_value}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare([direct, consumer] if direct_first else [consumer, direct])
    root = _feature(engine, DEFAULT_ROOT)
    assert root.initial_requested_data is True
    assert root.options.get(KEY) == DECLARED_DEFAULT
    parent = DEFAULT_RELAY if requested == INHERITED_OVERRIDE else requested
    assert engine.feature_link_parents[_feature(engine, parent).uuid] == {root.uuid}
    if requested == DEFAULT_CONSUMER and caller_value != DECLARED_DEFAULT:
        _assert_warning(caplog, [DEFAULT_CONSUMER], [UncascadedCtxDefaultRoot])
    else:
        assert _context_warnings(caplog) == []


@pytest.mark.parametrize("direct_first", [False, True], ids=["dependency-first", "direct-first"])
def test_direct_request_with_an_explicit_key_does_not_silence_a_lost_key(
    direct_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    direct = Feature(DEFAULT_ROOT, Options(group={KEY: DECLARED_DEFAULT}))
    consumer = Feature(DEFAULT_CONSUMER, Options(context={KEY: PRIVATE_VALUE}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare([direct, consumer] if direct_first else [consumer, direct])
    root = _feature(engine, DEFAULT_ROOT)
    assert root.initial_requested_data is True
    assert root.options.get(KEY) == DECLARED_DEFAULT
    assert engine.feature_link_parents[_feature(engine, DEFAULT_CONSUMER).uuid] == {root.uuid}
    _assert_warning(caplog, [DEFAULT_CONSUMER], [UncascadedCtxDefaultRoot])


@pytest.mark.parametrize("override_first", [False, True], ids=["consumer-first", "override-first"])
def test_one_consumer_delivering_does_not_hide_the_other_on_a_merged_input(
    override_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    override = Feature(DEFAULT_OVERRIDE, Options(context={KEY: PRIVATE_VALUE}))
    consumer = Feature(DEFAULT_CONSUMER, Options(context={KEY: PRIVATE_VALUE}))
    with caplog.at_level(logging.WARNING):
        engine = _prepare([override, consumer] if override_first else [consumer, override])
    root = _feature(engine, DEFAULT_ROOT)
    assert engine.feature_link_parents[_feature(engine, DEFAULT_OVERRIDE).uuid] == {root.uuid}
    assert engine.feature_link_parents[_feature(engine, DEFAULT_CONSUMER).uuid] == {root.uuid}
    _assert_warning(caplog, [DEFAULT_CONSUMER], [UncascadedCtxDefaultRoot])


def test_diamond_consumer_over_a_merged_input_still_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _prepare([Feature(DIAMOND, Options(context={KEY: PRIVATE_VALUE}))])
    _assert_warning(caplog, [DIAMOND], [UncascadedCtxDefaultRoot])


def test_consumer_side_merge_keeps_the_delivery_of_a_pinned_self_filter(caplog: pytest.LogCaptureFixture) -> None:
    global_filter = GlobalFilter()
    pinned = Feature(DEFAULT_OVERRIDE, compute_framework=PythonDictFramework.get_class_name())
    global_filter.add_filter(pinned, FilterType.MIN, {"value": 1})
    requests = [Feature(DEFAULT_OVERRIDE, Options(context={KEY: PRIVATE_VALUE}))]
    with (
        caplog.at_level(logging.WARNING),
        patch.object(Engine, "_merge_host_into", autospec=True, side_effect=Engine._merge_host_into) as merge_host,
    ):
        _prepare(requests, broader_frameworks=True, global_filter=global_filter)
    consumer_folds = [
        call
        for call in merge_host.call_args_list
        if str(call.args[1].name) == str(call.args[2].name) == DEFAULT_OVERRIDE
    ]
    assert len(consumer_folds) == 1, "The consumer-side narrowing merge must run"
    assert _context_warnings(caplog) == []


@pytest.mark.parametrize("requested", [PARTIAL_DELIVERY, PARTIAL_DEFAULT])
def test_delivery_to_one_branch_does_not_hide_the_other(requested: str, caplog: pytest.LogCaptureFixture) -> None:
    joined_source = UncascadedCtxUnusedRoot if requested == PARTIAL_DELIVERY else UncascadedCtxDefaultRoot
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))], joined_source)
    assert _feature(engine, GROUP_ROOT).options.get(KEY) == PRIVATE_VALUE
    missing = UNUSED_ROOT if requested == PARTIAL_DELIVERY else DEFAULT_ROOT
    assert _feature(engine, missing).options.get(KEY) == (None if missing == UNUSED_ROOT else DECLARED_DEFAULT)
    _assert_warning(caplog, [requested], [joined_source])


def test_two_declaring_input_groups_are_named_in_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _prepare([Feature(TWO_ROOTS, Options(context={KEY: PRIVATE_VALUE}))], UncascadedCtxUnusedRoot)
    _assert_warning(caplog, [TWO_ROOTS], [UncascadedCtxGroupRoot, UncascadedCtxUnusedRoot])


@pytest.mark.parametrize("requested", [DEFAULT_OVERRIDE, INHERITED_OVERRIDE])
@pytest.mark.parametrize("direct_first", [False, True], ids=["dependency-first", "direct-first"])
def test_late_narrower_fold_keeps_configured_default_override(
    requested: str, direct_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    direct = Feature(DEFAULT_ROOT, compute_framework=PythonDictFramework.get_class_name())
    consumer = Feature(requested, Options(context={KEY: PRIVATE_VALUE}))
    requests = [direct, consumer] if direct_first else [consumer, direct]
    with (
        caplog.at_level(logging.WARNING),
        patch.object(Engine, "_merge_host_into", autospec=True, side_effect=Engine._merge_host_into) as merge_host,
    ):
        engine = _prepare(requests, broader_frameworks=True)

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
    assert root.options.get(KEY) == DECLARED_DEFAULT
    assert engine._declared_options_by_uuid[root.uuid].get(KEY) is None
    parent = DEFAULT_RELAY if requested == INHERITED_OVERRIDE else requested
    assert engine.feature_link_parents[_feature(engine, parent).uuid] == {root.uuid}
    assert _context_warnings(caplog) == []


@pytest.mark.parametrize(
    "requested, group",
    [(GROUP_ONLY_DIRECT, UncascadedCtxGroupOnlyRoot), (HANDLE_DIRECT, UncascadedCtxHandleRoot)],
    ids=["group-only", "credential-handle"],
)
def test_lost_group_only_or_handle_key_names_the_group_remedy(
    requested: str, group: type[FeatureGroup], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _prepare([Feature(requested, Options(context={KEY: PRIVATE_VALUE}))])
    message = _assert_warning(caplog, [requested], [group], remedy=_group_only_remedy([KEY]))
    assert "cascade_context_keys" not in message


def test_mixed_request_names_each_remedy_with_its_own_keys(caplog: pytest.LogCaptureFixture) -> None:
    options = Options(context={KEY: PRIVATE_VALUE, ONLY_KEY: OTHER_PRIVATE_VALUE})
    with caplog.at_level(logging.WARNING):
        _prepare([Feature(MIXED_DIRECT, options)])
    remedy = f"{_cascade_remedy([KEY])} {_group_only_remedy([ONLY_KEY])}"
    _assert_warning(caplog, [MIXED_DIRECT], [UncascadedCtxMixedRoot], keys=[KEY, ONLY_KEY], remedy=remedy)


def test_two_requested_features_losing_keys_log_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    requests = [
        Feature(DIRECT, Options(context={KEY: PRIVATE_VALUE})),
        Feature(DEEP, Options(context={KEY: PRIVATE_VALUE})),
    ]
    with caplog.at_level(logging.WARNING):
        _prepare(requests)
    _assert_warning(caplog, [DIRECT, DEEP], [UncascadedCtxGroupRoot])


@pytest.mark.parametrize("delivered", [True, False], ids=["delivered", "lost"])
def test_global_filter_injected_feature_does_not_hide_the_input_walk(
    delivered: bool, caplog: pytest.LogCaptureFixture
) -> None:
    global_filter = GlobalFilter()
    global_filter.add_filter(FILTER_COL, FilterType.MIN, {"value": 1})
    options = (
        Options(context={KEY: PRIVATE_VALUE}, propagate_context_keys=frozenset({KEY}))
        if delivered
        else Options(context={KEY: PRIVATE_VALUE})
    )
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(FILTER_CONSUMER, options)], global_filter=global_filter)
    root_features = engine.feature_group_collection[UncascadedCtxFilterRoot]
    assert {str(feature.name) for feature in root_features} == {FILTER_ROOT, FILTER_COL}
    if delivered:
        assert _feature(engine, FILTER_ROOT).options.get(KEY) == PRIVATE_VALUE
        assert _warnings(caplog) == []
    else:
        _assert_warning(caplog, [FILTER_CONSUMER], [UncascadedCtxFilterRoot])


@pytest.mark.parametrize("a_propagates", [False, True], ids=["both-lost", "only-b-lost"])
def test_shared_input_object_keeps_each_consumers_lost_key(
    a_propagates: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # The propagating consumer goes first; reversed, planning raises on the already-defaulted group key.
    if a_propagates:
        options_a = Options(context={KEY: PRIVATE_VALUE}, propagate_context_keys=frozenset({KEY}))
        options_b = Options(context={KEY: OTHER_PRIVATE_VALUE})
        expected = [SHARED_B]
    else:
        options_a = Options(context={KEY: PRIVATE_VALUE})
        options_b = Options(context={KEY: PRIVATE_VALUE})
        expected = [SHARED_A, SHARED_B]
    with caplog.at_level(logging.WARNING):
        engine = _prepare([Feature(SHARED_A, options_a), Feature(SHARED_B, options_b)])
    assert [r for r in _warnings(caplog, DEFAULT_ROOT) if "default-equivalent options" in r.getMessage()] == []
    _assert_warning(caplog, expected, [UncascadedCtxDefaultRoot])
    root = _feature(engine, DEFAULT_ROOT)
    assert engine.feature_link_parents[_feature(engine, SHARED_A).uuid] == {root.uuid}
    assert engine.feature_link_parents[_feature(engine, SHARED_B).uuid] == {root.uuid}


def test_shared_input_object_setting_the_key_itself_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    requests = [
        Feature(SHARED_SET_A, Options(context={KEY: PRIVATE_VALUE})),
        Feature(SHARED_SET_B, Options(context={KEY: PRIVATE_VALUE})),
    ]
    with caplog.at_level(logging.WARNING):
        engine = _prepare(requests)
    assert _feature(engine, DEFAULT_ROOT).options.get(KEY) == EXPLICIT_OVERRIDE
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == "mloda.core.core.engine"] == []


@pytest.mark.parametrize("twin_first", [True, False], ids=["b-first", "a-first"])
def test_global_filter_twin_deduped_into_another_consumers_input_is_quiet(
    twin_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    global_filter = GlobalFilter()
    global_filter.add_filter(FILTER_COL, FilterType.MIN, {"value": 1})
    consumer_a = Feature(TWIN_A, Options(context={KEY: PRIVATE_VALUE}))
    consumer_b = Feature(TWIN_B)
    with caplog.at_level(logging.WARNING):
        _prepare([consumer_b, consumer_a] if twin_first else [consumer_a, consumer_b], global_filter=global_filter)
    assert _context_warnings(caplog) == []
