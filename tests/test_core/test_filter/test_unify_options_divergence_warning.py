"""Pin the option divergence warning for a filter feature declaring a defaulted key as None (#911).

Filter matching compares the host feature's EFFECTIVE options against the filter feature's DECLARED
ones, so a defaulted key declared as None is reported as diverging from a value intake materializes
into it one call later. The warning must fire on a divergence that survives intake, not on one that
converges. Every case drives the engine seam, because only there is the resolving feature group,
and with it the spec that decides what intake will materialize, available.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec
from mloda.user import FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# PROPERTY_MAPPING keys/values for the throwaway probes; the unw_ prefix keeps them unique to this module.
UNW_GRP_KEY = "unw_grp_key"
UNW_CTX_KEY = "unw_ctx_key"
UNW_OPTIN_KEY = "unw_optin_key"
UNW_DEFAULT = "unw_default_val"
UNW_HOST_VAL = "unw_host_val"
UNW_FILTER_VAL = "unw_filter_val"
UNW_CTX_VAL = "unw_ctx_val"

UNW_HOST = "unw_host_feature"  # the requested host feature the filter attaches to
UNW_TARGET = "unw_target_feature"  # never requested, only reachable as a matched filter feature

GRP_SPEC = PropertySpec("A group key with a concrete default.", context=False, default=UNW_DEFAULT)
CTX_SPEC = PropertySpec("A context key with a concrete default.", context=True, default=UNW_DEFAULT)
OPTIN_SPEC = PropertySpec(
    "A group key honoring an explicit None.", context=False, default=UNW_DEFAULT, allow_explicit_none=True
)


def _make_probe_fg(mapping: dict[str, PropertySpec]) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup serving the host and the filter target, echoing its effective options."""

    class UnwWarningProbeFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = mapping

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({UNW_HOST, UNW_TARGET})

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data, so the framework's row elimination must not run against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            observed = {str(f.name): {key: f.options.get(key) for key in mapping} for f in features.features}
            return {str(feature.name): [observed] for feature in features.features}

    return UnwWarningProbeFeatureGroup


def _stored_filter_options(global_filter: GlobalFilter) -> list[dict[str, dict[str, Any]]]:
    """Plain-data view of every collected filter feature's post-intake options, per namespace."""
    return [
        {"group": dict(single.filter_feature.options.group), "context": dict(single.filter_feature.options.context)}
        for stored in global_filter.collection.values()
        for single in stored
    ]


def _payload_rows(frames: list[Any], column: str) -> list[Any]:
    """Every row stored under ``column``, tolerant of columnar dict or list-of-row-dicts frames."""
    rows: list[Any] = []
    for frame in frames:
        if isinstance(frame, dict):
            if column in frame:
                rows.extend(frame[column])
        else:
            rows.extend(row[column] for row in frame if column in row)
    return rows


def _run(mapping: dict[str, PropertySpec], host_options: Options, filter_options: Options) -> dict[str, Any]:
    """Run UNW_HOST under a global EQUAL filter on UNW_TARGET; return plain-data views of both sides.

    The probe class, the collector and the GlobalFilter are deleted from THIS frame before the asserts,
    so a failing assert cannot pin the throwaway class into a traceback and trip the no-leak fixture.
    """
    collector = PluginCollector.enabled_feature_groups({_make_probe_fg(mapping)})
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(UNW_TARGET, filter_options), FilterType.EQUAL, {"value": 1})
    results = mloda.run_all(
        [Feature(UNW_HOST, host_options)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )

    stored = _stored_filter_options(global_filter)
    frames = list(results)
    rows = _payload_rows(frames, UNW_HOST)
    frames_repr = repr(frames)
    del collector, global_filter, results, frames

    assert len(rows) == 1, f"expected exactly one payload row for {UNW_HOST}, got frames: {frames_repr}"
    assert isinstance(rows[0], dict), f"expected a payload dict for {UNW_HOST}, got: {rows[0]!r}"
    assert len(stored) == 1, f"exactly one filter must have matched and been collected: {stored!r}"
    return {"host_effective": rows[0][UNW_HOST], "filter_stored": stored[0]}


def _warnings_naming(caplog: pytest.LogCaptureFixture, key: str) -> list[str]:
    """Every warning-level message naming ``key``; location-agnostic, since the check may move."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and key in record.getMessage()
    ]


def _stored_value(observed: dict[str, Any], key: str) -> Any:
    """The filter feature's post-intake value for ``key``, whichever namespace it landed in."""
    stored = observed["filter_stored"]
    if key in stored["group"]:
        return stored["group"][key]
    return stored["context"].get(key)


def test_no_warning_when_a_declared_none_converges_on_the_group_default(caplog: pytest.LogCaptureFixture) -> None:
    """A defaulted group key declared as None converges on the host's value, so no warning."""
    with caplog.at_level(logging.WARNING):
        observed = _run({UNW_GRP_KEY: GRP_SPEC}, Options(), Options(group={UNW_GRP_KEY: None}))

    assert observed["host_effective"][UNW_GRP_KEY] == UNW_DEFAULT
    assert _stored_value(observed, UNW_GRP_KEY) == UNW_DEFAULT
    assert _warnings_naming(caplog, UNW_GRP_KEY) == []


def test_no_warning_when_a_declared_none_converges_on_the_context_default(caplog: pytest.LogCaptureFixture) -> None:
    """The same for a context key: both namespaces must stop reporting the converging case."""
    with caplog.at_level(logging.WARNING):
        observed = _run({UNW_CTX_KEY: CTX_SPEC}, Options(), Options(context={UNW_CTX_KEY: None}))

    assert observed["host_effective"][UNW_CTX_KEY] == UNW_DEFAULT
    assert _stored_value(observed, UNW_CTX_KEY) == UNW_DEFAULT
    assert _warnings_naming(caplog, UNW_CTX_KEY) == []


def test_warns_when_a_declared_none_diverges_from_an_explicit_host_value(caplog: pytest.LogCaptureFixture) -> None:
    """Intake fills the spec default, not the host's value, so the filter feature computes with something else."""
    with caplog.at_level(logging.WARNING):
        observed = _run(
            {UNW_GRP_KEY: GRP_SPEC}, Options(group={UNW_GRP_KEY: UNW_HOST_VAL}), Options(group={UNW_GRP_KEY: None})
        )

    assert observed["host_effective"][UNW_GRP_KEY] == UNW_HOST_VAL
    assert _stored_value(observed, UNW_GRP_KEY) == UNW_DEFAULT
    assert _warnings_naming(caplog, UNW_GRP_KEY), "a divergence that survives intake must still be reported"


def test_warns_when_two_explicit_values_differ(caplog: pytest.LogCaptureFixture) -> None:
    """Two different non-None values stay different through intake, so the classic divergence still warns."""
    with caplog.at_level(logging.WARNING):
        observed = _run(
            {UNW_GRP_KEY: GRP_SPEC},
            Options(group={UNW_GRP_KEY: UNW_HOST_VAL}),
            Options(group={UNW_GRP_KEY: UNW_FILTER_VAL}),
        )

    assert observed["host_effective"][UNW_GRP_KEY] == UNW_HOST_VAL
    assert _stored_value(observed, UNW_GRP_KEY) == UNW_FILTER_VAL
    assert _warnings_naming(caplog, UNW_GRP_KEY), "two differing explicit values must still be reported"


def test_warns_when_an_opted_in_none_stays_divergent(caplog: pytest.LogCaptureFixture) -> None:
    """allow_explicit_none=True honors the None, so it is never replaced by the default and stays divergent."""
    with caplog.at_level(logging.WARNING):
        observed = _run({UNW_OPTIN_KEY: OPTIN_SPEC}, Options(), Options(group={UNW_OPTIN_KEY: None}))

    assert observed["host_effective"][UNW_OPTIN_KEY] == UNW_DEFAULT
    assert _stored_value(observed, UNW_OPTIN_KEY) is None
    assert _warnings_naming(caplog, UNW_OPTIN_KEY), "an honored None against a concrete host value must be reported"


def test_absent_filter_keys_are_copied_into_their_own_namespace_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keys absent from the filter options are copied per namespace and never warned about (#712)."""
    with caplog.at_level(logging.WARNING):
        observed = _run(
            {UNW_GRP_KEY: GRP_SPEC, UNW_CTX_KEY: CTX_SPEC},
            Options(group={UNW_GRP_KEY: UNW_HOST_VAL}, context={UNW_CTX_KEY: UNW_CTX_VAL}),
            Options(),
        )

    stored = observed["filter_stored"]
    assert stored["group"].get(UNW_GRP_KEY) == UNW_HOST_VAL
    assert stored["context"].get(UNW_CTX_KEY) == UNW_CTX_VAL
    assert UNW_CTX_KEY not in stored["group"], f"a context key must not leak into group: {stored!r}"
    assert _warnings_naming(caplog, UNW_GRP_KEY) == []
    assert _warnings_naming(caplog, UNW_CTX_KEY) == []
