"""Pin the os-008 intake-default-canonicalization contract.

Declared PropertySpec defaults are materialized at feature intake on the resolved feature's own
options, so default-equivalent same-name twins canonicalize through the standard duplicate path.
Dependency declaration keeps declared pre-default semantics: input_features and child option
inheritance observe the pre-default options. The compute boundary remains an idempotent safety net.
"""

from __future__ import annotations

import gc
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
