"""Pin the os-008 intake-default-canonicalization contract.

Declared PropertySpec defaults are materialized at feature intake on the resolved feature's own
options, so default-equivalent same-name twins canonicalize through the standard duplicate path.
Dependency declaration keeps declared pre-default semantics: input_features and child option
inheritance observe the pre-default options. The compute boundary remains an idempotent safety net.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec
from mloda.user import FeatureName, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    The tests below define FeatureGroup subclasses inside factory functions. Those class
    objects sit in reference cycles, so they linger in ``FeatureGroup.__subclasses__()``
    until a GC cycle runs. While they linger, other tests that enumerate feature groups via
    ``get_all_subclasses(FeatureGroup)`` trip over them. After each test we force a
    collection to reclaim the now-unreferenced classes and assert that none of this
    module's classes remain registered, pinning the no-pollution contract.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


# PROPERTY_MAPPING keys for the throwaway probes; the idc_ prefix keeps them unique to this module.
IDC_CTX_KEY = "idc_ctx_default"  # context concrete default (twin merge, test 1)
IDC_GRP_KEY = "idc_grp_default"  # group concrete default (twin merge, test 2)
IDC_DEP_KEY = "idc_dep_ctx_default"  # context concrete default steering input_features (test 3)
IDC_INHERIT_KEY = "idc_inherit_grp_default"  # group concrete default on a consumer parent (test 4)

IDC_CTX_DEFAULT = "idc_ctx_val"
IDC_GRP_DEFAULT = "idc_grp_val"
IDC_DEP_DEFAULT = "idc_dep_val"
IDC_INHERIT_DEFAULT = "idc_inherit_val"

IDC_CHILD_DECLARED = "idc_child_declared"  # resolved when IDC_DEP_KEY is absent (pre-default view)
IDC_CHILD_EFFECTIVE = "idc_child_effective"  # resolved when IDC_DEP_KEY is present
IDC_INHERIT_CHILD = "idc_inherit_child"  # root child under the group-default consumer parent

IDC_ALIAS_KEY = "idc_alias_ctx_default"  # context concrete default on the shared-Options alias probe
IDC_ALIAS_DEFAULT = "idc_alias_val"
IDC_ALIAS_NAME_A = "idc_alias_feature_a"  # two different names served by ONE root FG, sharing ONE Options
IDC_ALIAS_NAME_B = "idc_alias_feature_b"
IDC_GRP_NON_DEFAULT = "idc_grp_other_val"  # explicit non-default group value; NOT default-equivalent

IDC_DUP_KEY = "idc_dup_ctx_default"  # context concrete default on the plain-duplicate child probe
IDC_DUP_DEFAULT = "idc_dup_val"
IDC_DUP_CHILD = "idc_dup_child"  # child declared IDENTICALLY by both parents (plain duplicate)
IDC_DUP_PARENT_A = "idc_dup_parent_a"
IDC_DUP_PARENT_B = "idc_dup_parent_b"

# Verified against the module logger definition in mloda/core/core/engine.py (logging.getLogger(__name__)).
IDC_ENGINE_LOGGER = "mloda.core.core.engine"


def _make_ctx_default_root_fg(feature_name: str) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup with one context concrete default, echoing the observed value."""

    class IdcCtxDefaultRootFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_CTX_KEY: PropertySpec("A context concrete default.", context=True, default=IDC_CTX_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({feature_name})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [feature.options.get(IDC_CTX_KEY)] for feature in features.features}

    return IdcCtxDefaultRootFeatureGroup


def _make_grp_default_root_fg(feature_name: str) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup with one group concrete default, echoing the observed value."""

    class IdcGrpDefaultRootFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_GRP_KEY: PropertySpec("A group concrete default.", context=False, default=IDC_GRP_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({feature_name})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [feature.options.get(IDC_GRP_KEY)] for feature in features.features}

    return IdcGrpDefaultRootFeatureGroup


def _make_dep_probe_parent_fg(feature_name: str) -> type[FeatureGroup]:
    """A throwaway parent whose input_features branches on the presence of its own defaulted key.

    With pre-default (declared) semantics the key is absent for a plain request, so the parent
    must resolve IDC_CHILD_DECLARED; a premature default fill would flip it to IDC_CHILD_EFFECTIVE.
    """

    class IdcDepProbeParentFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_DEP_KEY: PropertySpec("Steers the declared dependency.", context=True, default=IDC_DEP_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {feature_name}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            if options.get(IDC_DEP_KEY) is None:
                return {Feature(IDC_CHILD_DECLARED)}
            return {Feature(IDC_CHILD_EFFECTIVE)}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            resolved = [name for name in (IDC_CHILD_DECLARED, IDC_CHILD_EFFECTIVE) if name in data]
            return {feature_name: [{"resolved_children": resolved}]}

    return IdcDepProbeParentFeatureGroup


def _make_marker_child_fg() -> type[FeatureGroup]:
    """A throwaway root child serving both dependency names with a marker payload per requested name."""

    class IdcMarkerChildFeatureGroup(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({IDC_CHILD_DECLARED, IDC_CHILD_EFFECTIVE})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [f"idc_payload_{feature.name}"] for feature in features.features}

    return IdcMarkerChildFeatureGroup


def _make_inherit_parent_fg(feature_name: str) -> type[FeatureGroup]:
    """A throwaway parent with a group concrete default that declares one root child dependency."""

    class IdcInheritParentFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_INHERIT_KEY: PropertySpec("A group concrete default.", context=False, default=IDC_INHERIT_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {feature_name}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature(IDC_INHERIT_CHILD)}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {feature_name: [data[IDC_INHERIT_CHILD][0]]}

    return IdcInheritParentFeatureGroup


def _make_option_echo_child_fg() -> type[FeatureGroup]:
    """A throwaway root child echoing whether the parent's group-default key reached its options."""

    class IdcOptionEchoChildFeatureGroup(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({IDC_INHERIT_CHILD})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {
                str(feature.name): [{"observed_parent_grp_key": feature.options.get(IDC_INHERIT_KEY)}]
                for feature in features.features
            }

    return IdcOptionEchoChildFeatureGroup


def _make_alias_probe_root_fg() -> type[FeatureGroup]:
    """A throwaway root FG serving both alias names, echoing options-object sharing and the observed default."""

    class IdcAliasProbeRootFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_ALIAS_KEY: PropertySpec("A context concrete default.", context=True, default=IDC_ALIAS_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({IDC_ALIAS_NAME_A, IDC_ALIAS_NAME_B})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            options_identities = len({id(feature.options) for feature in features.features})
            return {
                str(feature.name): [
                    {
                        "features_in_set": len(list(features.features)),
                        "options_identities": options_identities,
                        "observed_default": feature.options.get(IDC_ALIAS_KEY),
                    }
                ]
                for feature in features.features
            }

    return IdcAliasProbeRootFeatureGroup


def _make_dup_child_root_fg() -> type[FeatureGroup]:
    """A throwaway root child with a context concrete default, echoing the observed value."""

    class IdcDupChildRootFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_DUP_KEY: PropertySpec("A context concrete default.", context=True, default=IDC_DUP_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({IDC_DUP_CHILD})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [feature.options.get(IDC_DUP_KEY)] for feature in features.features}

    return IdcDupChildRootFeatureGroup


def _make_dup_parents_fg() -> type[FeatureGroup]:
    """A throwaway consumer serving two names, each declaring the SAME fully-explicit child feature."""

    class IdcDupParentsFeatureGroup(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {IDC_DUP_PARENT_A, IDC_DUP_PARENT_B}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature(IDC_DUP_CHILD, Options(context={IDC_DUP_KEY: IDC_DUP_DEFAULT}))}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [data[IDC_DUP_CHILD][0]] for feature in features.features}

    return IdcDupParentsFeatureGroup


def _single_row(frame: Any, column: str) -> Any:
    """Extract the single payload row, tolerant of columnar dict or list-of-row-dicts results."""
    if isinstance(frame, dict):
        values = list(frame[column])
    else:
        values = [row[column] for row in frame]
    assert len(values) == 1, f"expected exactly one row for {column}, got {values!r}"
    return values[0]


def _run_twins(
    make_fg: Callable[[str], type[FeatureGroup]], feature_name: str, twin_options: list[Options]
) -> list[Any]:
    """Run same-name twin requests end-to-end and return the raw result frames.

    The probe class and collector stay locals of THIS frame, so a failing assert in the
    caller never pins the throwaway class in a traceback and the no-leak guard stays green.
    """
    fg = make_fg(feature_name)
    collector = PluginCollector.enabled_feature_groups({fg})
    results = mloda.run_all(
        [Feature(feature_name, options) for options in twin_options],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
    )
    return list(results)


def _run_parent_child(
    make_parent: Callable[[str], type[FeatureGroup]],
    make_child: Callable[[], type[FeatureGroup]],
    parent_name: str,
    options: Options,
) -> dict[str, Any]:
    """Run one parent-child computation and return the parent's single payload row.

    Both throwaway classes stay locals of THIS frame (see _run_twins) so caller asserts
    never pin them.
    """
    parent_fg = make_parent(parent_name)
    child_fg = make_child()
    collector = PluginCollector.enabled_feature_groups({parent_fg, child_fg})
    results = mloda.run_all(
        [Feature(parent_name, options)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
    )
    assert len(results) == 1, f"expected exactly one result frame, got: {results!r}"
    payload = _single_row(results[0], parent_name)
    assert isinstance(payload, dict)
    return payload


def _run_alias_probe() -> list[Any]:
    """Run two different-name features sharing ONE Options instance; return the raw result frames.

    The probe class and collector stay locals of THIS frame (see _run_twins) so caller asserts
    never pin the throwaway class.
    """
    fg = _make_alias_probe_root_fg()
    collector = PluginCollector.enabled_feature_groups({fg})
    shared_options = Options()
    results = mloda.run_all(
        [Feature(IDC_ALIAS_NAME_A, shared_options), Feature(IDC_ALIAS_NAME_B, shared_options)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
    )
    return list(results)


def _run_dup_parents() -> list[Any]:
    """Run two parents that both declare the SAME fully-explicit child; return the raw result frames.

    Both throwaway classes stay locals of THIS frame (see _run_twins) so caller asserts never
    pin them.
    """
    parent_fg = _make_dup_parents_fg()
    child_fg = _make_dup_child_root_fg()
    collector = PluginCollector.enabled_feature_groups({parent_fg, child_fg})
    results = mloda.run_all(
        [Feature(IDC_DUP_PARENT_A), Feature(IDC_DUP_PARENT_B)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
    )
    return list(results)


def _default_equivalent_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Collect the engine logger's default-equivalent merge warning messages captured by caplog."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == IDC_ENGINE_LOGGER
        and record.levelno >= logging.WARNING
        and "default-equivalent" in record.getMessage()
    ]


def test_context_default_twins_merge_instead_of_raising() -> None:
    """Same-name twins split only by an explicitly-set declared context default merge at intake.

    Pre-os-008 the absent twin is filled at the compute boundary, collapsing the FeatureSet
    with the "collapsed duplicate features" ValueError; intake canonicalization must instead
    merge the twins through the standard duplicate path and compute once.
    """
    name = "idc_ctx_twin_feature"
    frames = _run_twins(
        _make_ctx_default_root_fg,
        name,
        [Options(context={IDC_CTX_KEY: IDC_CTX_DEFAULT}), Options()],
    )
    assert len(frames) == 1, f"merged twins must produce exactly one result frame, got: {frames!r}"
    observed = _single_row(frames[0], name)
    assert observed == IDC_CTX_DEFAULT, f"the merged feature must observe the declared default: {observed!r}"


def test_group_default_twins_merge_into_single_computation() -> None:
    """Same-name twins split only by an explicitly-set declared group default merge at intake.

    Pre-os-008 the pre-default group options differ, so the twins split into two FeatureSets
    and yield two result frames; intake canonicalization must merge them into ONE computation.
    """
    name = "idc_grp_twin_feature"
    frames = _run_twins(
        _make_grp_default_root_fg,
        name,
        [Options(group={IDC_GRP_KEY: IDC_GRP_DEFAULT}), Options()],
    )
    assert len(frames) == 1, f"default-equivalent group twins must merge into one frame, got {len(frames)} frames"
    observed = _single_row(frames[0], name)
    assert observed == IDC_GRP_DEFAULT, f"the merged feature must observe the declared default: {observed!r}"


def test_input_features_observes_pre_default_options() -> None:
    """input_features sees the declared pre-default options: the defaulted key stays absent (guard).

    A naive implementation materializing defaults BEFORE the dependency recursion would make the
    parent resolve IDC_CHILD_EFFECTIVE instead of IDC_CHILD_DECLARED.
    """
    payload = _run_parent_child(_make_dep_probe_parent_fg, _make_marker_child_fg, "idc_dep_parent_feature", Options())
    assert payload["resolved_children"] == [IDC_CHILD_DECLARED], (
        f"input_features must observe pre-default options and resolve the declared child: {payload!r}"
    )


def test_child_does_not_inherit_parent_default_group_key() -> None:
    """Child inheritance flows from the declared pre-default options: no default group key leaks (guard).

    Children inherit ALL consumer group options at planning; materializing the parent's group
    default before the recursion would leak it into the child's options.
    """
    payload = _run_parent_child(
        _make_inherit_parent_fg, _make_option_echo_child_fg, "idc_inherit_parent_feature", Options()
    )
    assert payload["observed_parent_grp_key"] is None, (
        f"the child must not inherit the parent's declared group default: {payload!r}"
    )


@pytest.mark.parametrize("explicit_first", [True, False], ids=["explicit_then_absent", "absent_then_explicit"])
def test_default_equivalent_twin_merge_warns_in_both_orders(
    explicit_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """A materialization-driven twin merge warns on the engine logger in either request order.

    The twins are unequal pre-materialization and become equal only through intake default
    canonicalization, so the engine must name the feature in a "default-equivalent" warning
    advising deduplication; plain duplicates stay silent (pinned separately).
    """
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    name = f"idc_ctx_warn_twin_{'ea' if explicit_first else 'ae'}"
    explicit = Options(context={IDC_CTX_KEY: IDC_CTX_DEFAULT})
    absent = Options()
    twin_options = [explicit, absent] if explicit_first else [absent, explicit]
    frames = _run_twins(_make_ctx_default_root_fg, name, twin_options)
    assert len(frames) == 1, f"default-equivalent twins must still merge into one frame, got: {len(frames)}"
    observed = _single_row(frames[0], name)
    assert observed == IDC_CTX_DEFAULT, f"the merged feature must observe the declared default: {observed!r}"
    warnings = _default_equivalent_warnings(caplog)
    matching = [message for message in warnings if name in message]
    assert matching, (
        f"expected a default-equivalent merge warning naming {name!r} on logger {IDC_ENGINE_LOGGER!r}, "
        f"captured engine warnings: {warnings!r}"
    )


def test_plain_duplicate_requests_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Fully identical duplicate features merge silently through the duplicate path: no warning (guard).

    Identical user-level requests are rejected at API setup ("Duplicate feature setup"), so the
    engine's plain-duplicate path is exercised via two parents declaring the SAME fully-explicit
    child. The twins are equal before AND after materialization: no default-equivalent warning.
    """
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    frames = _run_dup_parents()
    assert len(frames) == 1, f"both parents must compute in one frame, got: {len(frames)}"
    for name in (IDC_DUP_PARENT_A, IDC_DUP_PARENT_B):
        observed = _single_row(frames[0], name)
        assert observed == IDC_DUP_DEFAULT, f"each parent must observe the shared child's value: {observed!r}"
    warnings = _default_equivalent_warnings(caplog)
    assert not warnings, f"plain duplicates must not trigger a default-equivalent warning, got: {warnings!r}"


def test_shared_options_stay_aliased_through_intake() -> None:
    """One Options instance shared by two features keeps ONE effective Options identity after intake.

    Intake materialization must be memoized per (feature group class, source Options identity):
    two features sharing a pre-default Options instance must share the effective instance too,
    not receive two separate equal-but-distinct objects.
    """
    frames = _run_alias_probe()
    assert len(frames) == 1, f"both alias features must compute in one frame, got: {len(frames)}"
    for name in (IDC_ALIAS_NAME_A, IDC_ALIAS_NAME_B):
        payload = _single_row(frames[0], name)
        assert isinstance(payload, dict)
        assert payload["features_in_set"] == 2, f"both alias features must share one FeatureSet: {payload!r}"
        assert payload["observed_default"] == IDC_ALIAS_DEFAULT, (
            f"the declared default must be materialized on the shared options: {payload!r}"
        )
        assert payload["options_identities"] == 1, (
            f"intake must keep the shared Options instance aliased across both features: {payload!r}"
        )


def test_non_default_equivalent_twins_do_not_merge(caplog: pytest.LogCaptureFixture) -> None:
    """An explicit NON-default group value against an absent twin stays split with no warning (guard)."""
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    name = "idc_grp_split_feature"
    frames = _run_twins(
        _make_grp_default_root_fg,
        name,
        [Options(group={IDC_GRP_KEY: IDC_GRP_NON_DEFAULT}), Options()],
    )
    assert len(frames) == 2, f"non-default-equivalent twins must stay split into two frames, got: {len(frames)}"
    observed = {_single_row(frame, name) for frame in frames}
    assert observed == {IDC_GRP_NON_DEFAULT, IDC_GRP_DEFAULT}, (
        f"the split twins must observe the explicit non-default and the materialized default: {observed!r}"
    )
    warnings = _default_equivalent_warnings(caplog)
    assert not warnings, f"non-default-equivalent twins must not trigger a warning, got: {warnings!r}"
