"""Pin the intake default-canonicalization contract.

Declared PropertySpec defaults are materialized at feature intake on the resolved feature's own
options, so default-equivalent same-name twins canonicalize through the standard duplicate path.
Dependency declaration keeps declared pre-default semantics: input_features and child option
inheritance observe the pre-default options. The compute boundary remains an idempotent safety net.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreatorFeatureGroup, PropertySpec
from mloda.user import FeatureName, GlobalFilter, Index, JoinSpec, Link, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_abstract_plugins.name_bound_fixtures import make_name_bound_fg, make_root_fg


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


def _make_ctx_default_root_fg(*feature_names: str) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup with one context concrete default, echoing the observed value."""

    class IdcCtxDefaultRootFeatureGroup(DataCreatorFeatureGroup):
        PROPERTY_MAPPING = {
            IDC_CTX_KEY: PropertySpec("A context concrete default.", context=True, default=IDC_CTX_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(feature_names)

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [feature.options.get(IDC_CTX_KEY)] for feature in features.features}

    return IdcCtxDefaultRootFeatureGroup


def _make_grp_default_root_fg(*feature_names: str) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup with one group concrete default, echoing the observed value."""

    class IdcGrpDefaultRootFeatureGroup(DataCreatorFeatureGroup):
        PROPERTY_MAPPING = {
            IDC_GRP_KEY: PropertySpec("A group concrete default.", context=False, default=IDC_GRP_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(feature_names)

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [feature.options.get(IDC_GRP_KEY)] for feature in features.features}

    return IdcGrpDefaultRootFeatureGroup


def _make_dep_probe_parent_fg(feature_name: str, resolve_defaults: bool = False) -> type[FeatureGroup]:
    """A throwaway parent whose input_features branches on the presence of its own defaulted key.

    With pre-default (declared) semantics the key is absent for a plain request, so the parent
    must resolve IDC_CHILD_DECLARED; a premature default fill would flip it to IDC_CHILD_EFFECTIVE.
    With resolve_defaults it first calls options_with_defaults, so it reaches the declared default.
    """

    class IdcDepProbeParentFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            IDC_DEP_KEY: PropertySpec("Steers the declared dependency.", context=True, default=IDC_DEP_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {feature_name}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            if resolve_defaults:
                options = self.options_with_defaults(options)
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

    class IdcMarkerChildFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {IDC_CHILD_DECLARED, IDC_CHILD_EFFECTIVE}

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

    class IdcOptionEchoChildFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {IDC_INHERIT_CHILD}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {
                str(feature.name): [{"observed_parent_grp_key": feature.options.get(IDC_INHERIT_KEY)}]
                for feature in features.features
            }

    return IdcOptionEchoChildFeatureGroup


def _make_alias_probe_root_fg() -> type[FeatureGroup]:
    """A throwaway root FG serving both alias names, echoing options-object sharing and the observed default."""

    class IdcAliasProbeRootFeatureGroup(DataCreatorFeatureGroup):
        PROPERTY_MAPPING = {
            IDC_ALIAS_KEY: PropertySpec("A context concrete default.", context=True, default=IDC_ALIAS_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {IDC_ALIAS_NAME_A, IDC_ALIAS_NAME_B}

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

    class IdcDupChildRootFeatureGroup(DataCreatorFeatureGroup):
        PROPERTY_MAPPING = {
            IDC_DUP_KEY: PropertySpec("A context concrete default.", context=True, default=IDC_DUP_DEFAULT),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {IDC_DUP_CHILD}

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
        compute_frameworks=[PythonDictFramework],
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
        compute_frameworks=[PythonDictFramework],
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
        compute_frameworks=[PythonDictFramework],
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
        compute_frameworks=[PythonDictFramework],
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

    Without intake canonicalization the absent twin is filled at the compute boundary, collapsing the FeatureSet
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

    Without intake canonicalization the pre-default group options differ, so the twins split into two FeatureSets
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


def test_input_features_reaches_declared_default_through_options_with_defaults() -> None:
    """input_features can reach the declared default by calling options_with_defaults itself."""
    payload = _run_parent_child(
        lambda name: _make_dep_probe_parent_fg(name, resolve_defaults=True),
        _make_marker_child_fg,
        "idc_dep_defaulting_parent_feature",
        Options(),
    )
    assert payload["resolved_children"] == [IDC_CHILD_EFFECTIVE], (
        f"options_with_defaults inside input_features must materialize the declared default: {payload!r}"
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


IDC_BOUND_A = "idc_bound_a"  # name-bound keys of the pattern probe
IDC_BOUND_B = "idc_bound_b"
IDC_BOUND_CHILD = "idc_bound_child"


def _run_bound(bound_context: bool = True) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    parent = make_name_bound_fg(
        "IdcBound",
        __name__,
        "idcop",
        (IDC_BOUND_A,),
        (IDC_BOUND_B,),
        defaults={IDC_BOUND_A: "a", IDC_BOUND_B: "b"},
        seen=seen,
        input_column=IDC_BOUND_CHILD,
        bound_context=bound_context,
    )
    child = make_root_fg("IdcBound", __name__, IDC_BOUND_CHILD, [1], seen=seen, observe_key=IDC_BOUND_A)
    mloda.run_all(
        [Feature(f"{IDC_BOUND_CHILD}__idcop_q")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({parent, child}),
    )
    return seen


def test_child_does_not_inherit_name_bound_key() -> None:
    seen = _run_bound(bound_context=False)
    assert seen["child"] is None


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


IDC_CONV_KEY = "idc_conv_k"
IDC_CONV_COLUMN = "idc_conv_child"


def _run_converted(converter: Callable[[str], Any], requests: list[Options | None]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    parent = make_name_bound_fg(
        "IdcConv",
        __name__,
        "idcvt",
        (IDC_CONV_KEY,),
        seen=seen,
        input_column=IDC_CONV_COLUMN,
        converters={IDC_CONV_KEY: converter},
        capture=r"\d",
    )
    child = make_root_fg("IdcConv", __name__, IDC_CONV_COLUMN, [1])
    features = [Feature(f"{IDC_CONV_COLUMN}__idcvt_5", options) for options in requests]
    mloda.run_all(
        features,
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({parent, child}),
    )
    return seen


def test_name_converter_gives_calculate_feature_a_typed_value() -> None:
    seen = _run_converted(int, [None])
    assert seen[f"{IDC_CONV_COLUMN}__idcvt_5"] == (5,)
    assert type(seen[f"{IDC_CONV_COLUMN}__idcvt_5"][0]) is int


@pytest.mark.parametrize("name_only_first", [True, False])
def test_name_only_and_typed_context_request_merge_to_the_int_in_both_orders(name_only_first: bool) -> None:
    requests: list[Options | None] = [None, Options(context={IDC_CONV_KEY: 5})]
    seen = _run_converted(int, requests if name_only_first else requests[::-1])
    assert seen[f"{IDC_CONV_COLUMN}__idcvt_5"] == (5,)
    assert type(seen[f"{IDC_CONV_COLUMN}__idcvt_5"][0]) is int


def test_name_converter_returning_an_unhashable_value_works_through_intake() -> None:
    seen = _run_converted(lambda text: [int(text)], [None])
    assert seen[f"{IDC_CONV_COLUMN}__idcvt_5"] == ([5],)


def _run_index_twin(layout: str, prefix: str) -> list[Any]:
    """Run an index-twin request layout end-to-end and return the raw result frames."""
    key, value, left_key = f"{prefix}_key", f"{prefix}_val", f"{prefix}_left_key"
    explicit = Options(context={IDC_CTX_KEY: IDC_CTX_DEFAULT})
    right_fg = _make_ctx_default_root_fg(key, value)
    left_fg = make_root_fg("IdcTwinLeft", __name__, left_key, [1])
    links: set[Link] | None = (
        None if layout == "keep_index" else {Link.inner(JoinSpec(left_fg, left_key), JoinSpec(right_fg, key))}
    )
    requests: dict[str, list[Feature]] = {
        "key_first": [Feature(left_key), Feature(key), Feature(value)],
        "value_first": [Feature(left_key), Feature(value), Feature(key)],
        "value_explicit": [Feature(left_key), Feature(value, explicit), Feature(key)],
        "keep_index": [Feature(value, index=Index((key,)), keep_index=True), Feature(key)],
        "twin_before": [Feature(left_key), Feature(value), Feature(key), Feature(key, explicit)],
        "twin_after": [Feature(left_key), Feature(key), Feature(key, explicit), Feature(value)],
    }
    results = mloda.run_all(
        requests[layout],
        compute_frameworks=[PythonDictFramework],
        links=links,
        plugin_collector=PluginCollector.enabled_feature_groups({left_fg, right_fg}),
    )
    return list(results)


def _frame_with(frames: list[Any], column: str) -> Any:
    matching = [frame for frame in frames if (column in frame if isinstance(frame, dict) else column in frame[0])]
    assert len(matching) == 1, f"expected exactly one frame holding {column!r}, got: {frames!r}"
    return matching[0]


@pytest.mark.parametrize(
    "layout",
    ["key_first", "value_first", "value_explicit", "keep_index"],
    ids=["link_key_first", "link_value_first", "link_value_explicit", "keep_index_no_link"],
)
def test_injected_index_twin_merge_does_not_warn(layout: str, caplog: pytest.LogCaptureFixture) -> None:
    """An index twin merging with the single request of its column is not a double request."""
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    prefix = f"idc_twin_{layout}"
    frames = _run_index_twin(layout, prefix)
    frame = _frame_with(frames, f"{prefix}_val")
    assert _single_row(frame, f"{prefix}_key") == IDC_CTX_DEFAULT
    assert _single_row(frame, f"{prefix}_val") == IDC_CTX_DEFAULT
    warnings = _default_equivalent_warnings(caplog)
    assert not warnings, f"an injected index twin must not trigger a default-equivalent warning, got: {warnings!r}"


@pytest.mark.parametrize("layout", ["twin_before", "twin_after"])
def test_double_requested_link_index_still_warns_once(layout: str, caplog: pytest.LogCaptureFixture) -> None:
    """Two user requests of a link index column differing only in an explicit default warn exactly once."""
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    prefix = f"idc_dbl_{layout}"
    _run_index_twin(layout, prefix)
    warnings = _default_equivalent_warnings(caplog)
    assert len(warnings) == 1, f"expected exactly one default-equivalent warning, got: {warnings!r}"
    assert f"{prefix}_key" in warnings[0], f"the warning must name the key column: {warnings[0]!r}"


def _run_filter_twin(
    layout: str, prefix: str, default_kind: str, pinned: bool = False
) -> tuple[list[Any], list[str], GlobalFilter]:
    """Run a filtered-request layout end-to-end; return the frames, plain requested names and the filter."""
    key, value = f"{prefix}_key", f"{prefix}_val"
    if default_kind == "ctx":
        fg = _make_ctx_default_root_fg(key, value)
        explicit = Options(context={IDC_CTX_KEY: IDC_CTX_DEFAULT})
        default = IDC_CTX_DEFAULT
    else:
        fg = _make_grp_default_root_fg(key, value)
        explicit = Options(group={IDC_GRP_KEY: IDC_GRP_DEFAULT})
        default = IDC_GRP_DEFAULT
    requests: dict[str, list[Feature]] = {
        "key_only": [Feature(key)],
        "value_then_key": [Feature(value), Feature(key)],
        "key_then_value": [Feature(key), Feature(value)],
        "value_only": [Feature(value)],
        "filter_explicit": [Feature(key)],
        "two_filters": [Feature(key)],
        "twin_before": [Feature(value), Feature(key), Feature(key, explicit)],
        "twin_after": [Feature(key), Feature(key, explicit), Feature(value)],
    }
    cf = "PythonDictFramework" if pinned else None

    def plain(name: str) -> str | Feature:
        return Feature(name, compute_framework=cf) if pinned else name

    def with_explicit(name: str) -> Feature:
        return Feature(name, explicit, compute_framework=cf)

    global_filter = GlobalFilter()
    if layout == "filter_explicit":
        global_filter.add_filter(with_explicit(key), "equal", {"value": default})
    elif layout == "two_filters":
        global_filter.add_filter(plain(key), "equal", {"value": default})
        global_filter.add_filter(with_explicit(key), "equal", {"value": default})
    else:
        global_filter.add_filter(plain(value if layout == "value_only" else key), "equal", {"value": default})
    frameworks = [PythonDictFramework, PyArrowTable] if pinned else [PythonDictFramework]
    results = mloda.run_all(
        requests[layout],
        compute_frameworks=frameworks,
        global_filter=global_filter,
        plugin_collector=PluginCollector.enabled_feature_groups({fg}),
    )
    names = sorted({str(f.name) for f in requests[layout]})
    return list(results), names, global_filter


def _run_user_pinned_twin(name: str, default_kind: str, pinned_first: bool) -> tuple[list[Any], list[int]]:
    """Run pinned/unpinned user twins without a filter and count features per calculation."""
    key = IDC_CTX_KEY if default_kind == "ctx" else IDC_GRP_KEY
    default = IDC_CTX_DEFAULT if default_kind == "ctx" else IDC_GRP_DEFAULT
    calculations: list[int] = []

    class IdcUserPinnedTwinFeatureGroup(DataCreatorFeatureGroup):
        PROPERTY_MAPPING = {
            key: PropertySpec("A declared default.", context=default_kind == "ctx", default=default),
        }

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {name}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            calculations.append(len(features.features))
            return {str(feature.name): [feature.options.get(key)] for feature in features.features}

    explicit = Options(context={key: default}) if default_kind == "ctx" else Options(group={key: default})
    requests = [Feature(name, compute_framework="PythonDictFramework"), Feature(name, explicit)]
    results = mloda.run_all(
        requests if pinned_first else requests[::-1],
        compute_frameworks=[PythonDictFramework, PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({IdcUserPinnedTwinFeatureGroup}),
    )
    return list(results), calculations


@pytest.mark.parametrize("default_kind", ["ctx", "grp"])
@pytest.mark.parametrize("pinned_first", [True, False], ids=["pinned-first", "pinned-last"])
def test_user_pinned_default_equivalent_twin_warns_once_without_filter(
    default_kind: str, pinned_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    name = f"idc_userpin_{default_kind}_{'first' if pinned_first else 'last'}"
    default = IDC_CTX_DEFAULT if default_kind == "ctx" else IDC_GRP_DEFAULT
    frames, calculations = _run_user_pinned_twin(name, default_kind, pinned_first)
    assert len(frames) == 1
    assert _single_row(frames[0], name) == default
    assert calculations == [1], "the twins must merge into one feature calculated once"
    warnings = _default_equivalent_warnings(caplog)
    assert len(warnings) == 1, f"expected exactly one default-equivalent warning, got: {warnings!r}"
    assert name in warnings[0], f"the warning must name the requested column: {warnings[0]!r}"


@pytest.mark.parametrize("default_kind", ["ctx", "grp"])
@pytest.mark.parametrize("pinned", [False, True], ids=["unpinned", "pinned"])
@pytest.mark.parametrize(
    "layout", ["key_only", "value_then_key", "key_then_value", "value_only", "filter_explicit", "two_filters"]
)
def test_injected_filter_twin_merge_does_not_warn(
    layout: str, pinned: bool, default_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A filter twin merging with the single request of its column is not a double request."""
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    prefix = f"idc_ftwin_{default_kind}_{layout}_{'pin' if pinned else 'unpin'}"
    default = IDC_CTX_DEFAULT if default_kind == "ctx" else IDC_GRP_DEFAULT
    frames, requested, global_filter = _run_filter_twin(layout, prefix, default_kind, pinned)
    for column in requested:
        assert _single_row(_frame_with(frames, column), column) == default
    declared = {f.uuid for f in global_filter.filters}
    assert declared <= global_filter.matched_filter_uuids, "every declared filter must have matched a feature"
    warnings = _default_equivalent_warnings(caplog)
    assert not warnings, f"an injected filter twin must not trigger a default-equivalent warning, got: {warnings!r}"


@pytest.mark.parametrize("default_kind", ["ctx", "grp"])
@pytest.mark.parametrize("pinned", [False, True], ids=["unpinned", "pinned"])
@pytest.mark.parametrize("layout", ["twin_before", "twin_after"])
def test_double_requested_filtered_column_still_warns_once(
    layout: str, pinned: bool, default_kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Two user requests of a filtered column differing only in an explicit default warn exactly once."""
    caplog.set_level(logging.WARNING, logger=IDC_ENGINE_LOGGER)
    prefix = f"idc_fdbl_{default_kind}_{layout}_{'pin' if pinned else 'unpin'}"
    default = IDC_CTX_DEFAULT if default_kind == "ctx" else IDC_GRP_DEFAULT
    frames, _names, _filter = _run_filter_twin(layout, prefix, default_kind, pinned)
    for column in (f"{prefix}_key", f"{prefix}_val"):
        assert _single_row(_frame_with(frames, column), column) == default
    warnings = _default_equivalent_warnings(caplog)
    assert len(warnings) == 1, f"expected exactly one default-equivalent warning, got: {warnings!r}"
    assert f"{prefix}_key" in warnings[0], f"the warning must name the key column: {warnings[0]!r}"


IDC_GO_KEY = "idc_go_bound"
IDC_GO_SRC = "idc_go_src"


def test_a_name_bound_group_only_key_is_not_refused() -> None:
    """A group-only key bound from the feature name is not a caller's context placement, so it is not refused."""
    seen: dict[str, Any] = {}
    host = make_name_bound_fg(
        "IdcGo", __name__, "idcgo", (IDC_GO_KEY,), seen=seen, input_column=IDC_GO_SRC, group_only=(IDC_GO_KEY,)
    )
    src = make_root_fg("IdcGoSrc", __name__, IDC_GO_SRC, [1])
    name = f"{IDC_GO_SRC}__idcgo_q"
    results = mloda.run_all(
        [Feature(name)],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({host, src}),
    )
    assert results
    assert seen[name] == ("q",)


IDC_GO_ID = "idc_go_id"


def test_a_group_only_name_bound_key_is_not_refused_on_the_index_twin() -> None:
    """An index twin carries the host's name-bound group-only value; it is not refused again."""
    host = make_name_bound_fg(
        "IdcGoTwin", __name__, "idcgo", (IDC_GO_KEY,), input_column=IDC_GO_SRC, group_only=(IDC_GO_KEY,)
    )
    src = make_root_fg("IdcGoTwinSrc", __name__, IDC_GO_SRC, [1])
    left = make_root_fg("IdcGoTwinLeft", __name__, IDC_GO_ID, [1])
    link = Link.inner(JoinSpec(left, IDC_GO_ID), JoinSpec(host, IDC_GO_ID))
    mloda.prepare(
        [Feature(IDC_GO_ID), Feature(f"{IDC_GO_SRC}__idcgo_q")],
        compute_frameworks=[PythonDictFramework],
        links={link},
        plugin_collector=PluginCollector.enabled_feature_groups({host, src, left}),
    )
