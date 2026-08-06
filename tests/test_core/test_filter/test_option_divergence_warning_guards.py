"""Pin the guard clauses and the rendered text of the filter-option divergence warning (#911).

``test_unify_options_divergence_warning.py`` drives the same behaviour through the engine. These cases
call the seam directly: every guard below decides nothing but whether one log line is written.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.filter.global_filter import GlobalFilter
from mloda.provider import PropertySpec


# PROPERTY_MAPPING keys/values for the throwaway probes; the dwg_ prefix keeps them unique to this module.
DWG_KEY = "dwg_key"
DWG_CTX_KEY = "dwg_ctx_key"
DWG_UNMAPPED_KEY = "dwg_unmapped_key"
DWG_DEFAULT = "dwg_default_val"
DWG_HOST_VAL = "dwg_host_val"
DWG_FILTER_VAL = "dwg_filter_val"


class _Ambiguous:
    """A comparison result that refuses to be a bool, like the array a numpy ``==`` returns."""

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array with more than one element is ambiguous.")


class _ArrayLikeDefault:
    """A PROPERTY_MAPPING default whose ``==`` yields a non-bool, the shape numpy and pandas defaults have."""

    def __eq__(self, other: object) -> Any:
        return _Ambiguous()

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "<array-like default>"


def _emit(mapping: Optional[dict[str, PropertySpec]], feat_options: Options, filter_options: Options) -> None:
    """Run the seam once against a throwaway group; ``mapping`` None means no resolving group.

    The probe class stays local to this frame, so it cannot leak into the registry.
    """
    feature_group: Optional[type[FeatureGroup]] = None
    if mapping is not None:

        class DwgProbeFeatureGroup(FeatureGroup):
            PROPERTY_MAPPING = mapping

        feature_group = DwgProbeFeatureGroup

    GlobalFilter()._warn_on_diverging_options(feature_group, feat_options, filter_options)


def _messages(caplog: pytest.LogCaptureFixture, key: str) -> list[str]:
    """Every warning-level message naming ``key``."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and key in record.getMessage()
    ]


def test_stays_silent_when_intake_materializes_the_features_own_value(caplog: pytest.LogCaptureFixture) -> None:
    """The control: without it every guard case below could pass on a seam that never warns."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_KEY: PropertySpec("A group key with a concrete default.", context=False, default=DWG_DEFAULT)},
            Options(group={DWG_KEY: DWG_DEFAULT}),
            Options(group={DWG_KEY: None}),
        )

    assert _messages(caplog, DWG_KEY) == []


def test_warns_without_a_resolving_feature_group(caplog: pytest.LogCaptureFixture) -> None:
    """No group means no spec to consult, so nothing is suppressed."""
    with caplog.at_level(logging.WARNING):
        _emit(None, Options(group={DWG_KEY: DWG_DEFAULT}), Options(group={DWG_KEY: None}))

    assert _messages(caplog, DWG_KEY), "without a spec the fallback must report the divergence"


def test_warns_when_the_key_is_absent_from_property_mapping(caplog: pytest.LogCaptureFixture) -> None:
    """An unmapped key is never materialized by intake, so its divergence survives."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_KEY: PropertySpec("An unrelated mapped key.", context=False, default=DWG_DEFAULT)},
            Options(group={DWG_UNMAPPED_KEY: DWG_DEFAULT}),
            Options(group={DWG_UNMAPPED_KEY: None}),
        )

    assert _messages(caplog, DWG_UNMAPPED_KEY), "a key outside PROPERTY_MAPPING must keep warning"


def test_warns_when_the_spec_declares_no_default(caplog: pytest.LogCaptureFixture) -> None:
    """NO_DEFAULT fills nothing, so the declared None reaches compute time and stays divergent."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_KEY: PropertySpec("A required group key.", context=False)},
            Options(group={DWG_KEY: DWG_DEFAULT}),
            Options(group={DWG_KEY: None}),
        )

    assert _messages(caplog, DWG_KEY), "a key without a declared default must keep warning"


def test_warns_when_the_spec_default_is_none(caplog: pytest.LogCaptureFixture) -> None:
    """A declared default of None applies no value, so intake leaves the declared None alone."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_KEY: PropertySpec("A group key defaulting to None.", context=False, default=None)},
            Options(group={DWG_KEY: DWG_DEFAULT}),
            Options(group={DWG_KEY: None}),
        )

    assert _messages(caplog, DWG_KEY), "a spec default of None must keep warning"


def test_warns_when_the_spec_default_cannot_be_compared(caplog: pytest.LogCaptureFixture) -> None:
    """An uncomparable default must not abort the run from a decision that only picks a log line."""
    spec = PropertySpec("A group key defaulting to an array-like.", context=False, default=_ArrayLikeDefault())
    with caplog.at_level(logging.WARNING):
        _emit({DWG_KEY: spec}, Options(group={DWG_KEY: DWG_HOST_VAL}), Options(group={DWG_KEY: None}))

    # The rendered fill proves the uncomparable default was reached, not skipped by an earlier guard.
    assert _messages(caplog, DWG_KEY) == [
        f"Options are not the same. {DWG_KEY} is different. "
        f"None (intake fills <array-like default>) != '{DWG_HOST_VAL}'"
    ]


def test_warns_for_a_context_key_divergence(caplog: pytest.LogCaptureFixture) -> None:
    """The context namespace is reported too, not only the group one."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_CTX_KEY: PropertySpec("A context key with a concrete default.", default=DWG_DEFAULT)},
            Options(context={DWG_CTX_KEY: DWG_HOST_VAL}),
            Options(context={DWG_CTX_KEY: DWG_FILTER_VAL}),
        )

    assert _messages(caplog, DWG_CTX_KEY), "a context key divergence must be reported"


def test_the_surviving_none_warning_names_the_value_intake_materializes(caplog: pytest.LogCaptureFixture) -> None:
    """The filter feature computes with the spec default, so naming the feature's value alone misleads."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_KEY: PropertySpec("A group key with a concrete default.", context=False, default=DWG_DEFAULT)},
            Options(group={DWG_KEY: DWG_HOST_VAL}),
            Options(group={DWG_KEY: None}),
        )

    assert _messages(caplog, DWG_KEY) == [
        f"Options are not the same. {DWG_KEY} is different. None (intake fills '{DWG_DEFAULT}') != '{DWG_HOST_VAL}'"
    ]


def test_stays_silent_for_a_non_forwarded_key_while_an_ordinary_key_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """unify_options can never converge a non-forwarded key, so reporting its divergence is unactionable."""
    blocked = DefaultOptionKeys.in_features.value
    with caplog.at_level(logging.WARNING):
        _emit(
            None,
            Options(group={blocked: DWG_HOST_VAL, DWG_KEY: DWG_HOST_VAL}),
            Options(group={blocked: DWG_FILTER_VAL, DWG_KEY: DWG_FILTER_VAL}),
        )

    assert _messages(caplog, blocked) == [], "a non-forwarded key must not be reported"
    assert _messages(caplog, DWG_KEY) == [
        f"Options are not the same. {DWG_KEY} is different. {DWG_FILTER_VAL} != {DWG_HOST_VAL}"
    ], "an ordinary diverging key must still warn"


def test_the_plain_divergence_message_is_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    """Two explicit values reach compute time as declared, so their message keeps its original wording."""
    with caplog.at_level(logging.WARNING):
        _emit(
            {DWG_KEY: PropertySpec("A group key with a concrete default.", context=False, default=DWG_DEFAULT)},
            Options(group={DWG_KEY: DWG_HOST_VAL}),
            Options(group={DWG_KEY: DWG_FILTER_VAL}),
        )

    assert _messages(caplog, DWG_KEY) == [
        f"Options are not the same. {DWG_KEY} is different. {DWG_FILTER_VAL} != {DWG_HOST_VAL}"
    ]
