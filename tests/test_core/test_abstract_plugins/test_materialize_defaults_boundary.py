"""Pin runtime materialization of declared PropertySpec defaults at the calculate_feature boundary (#796).

``FeatureGroup.options_with_defaults`` (#766) fills absent PROPERTY_MAPPING keys that declare a concrete
default, but the framework still hands the ORIGINAL pre-default FeatureSet to ``calculate_feature``, so a
plugin reading ``feature.options`` directly never sees declared defaults. The pinned contract:

- ``FeatureSet.materialize_option_defaults(feature_group)`` rebinds every ``feature.options`` (and
  ``self.options``) through ``feature_group.options_with_defaults``, memoized by Options object identity so
  aliased Options stay aliased, rebuilding ``self.features`` when anything was rebound (Feature.__hash__
  includes options). It is an identity no-op when the feature group declares no concrete defaults, and it
  never mutates the input Options objects.
- ``ComputeFramework.run_calculate_feature`` invokes it up front, so BOTH the direct ``calculate_feature``
  call and the FEATURE_GROUP_CALCULATE_FEATURE extender path observe materialized options: absent keys carry
  their declared defaults, present values (even falsy) win, and an ``allow_explicit_none=True`` explicit
  None is retained.
"""

from __future__ import annotations

import gc
from typing import Any, Callable
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.provider import DataCreator, PropertySpec
from mloda.user import PluginCollector, mloda
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


# PROPERTY_MAPPING keys for the throwaway probes; the mdb_ prefix keeps them unique to this module.
MDB_CTX_KEY = "mdb_ctx_default"  # context concrete default
MDB_GRP_KEY = "mdb_grp_default"  # group concrete default (context=False)
MDB_FALSY_KEY = "mdb_falsy_default"  # concrete default overridable by a falsy explicit value
MDB_EN_KEY = "mdb_explicit_none_optin"  # allow_explicit_none=True with a concrete default
MDB_SIBLING_KEY = "mdb_en_sibling_default"  # plain concrete default next to the opted-in key
MDB_REQ_KEY = "mdb_required"  # NO_DEFAULT (no concrete default, never filled)

CTX_DEFAULT = "mdb_ctx_val"
GRP_DEFAULT = "mdb_grp_val"
FALSY_DEFAULT = "mdb_nonfalsy"
EN_DEFAULT = "mdb_en_val"
SIBLING_DEFAULT = "mdb_sibling_val"


def _make_boundary_probe_fg(feature_name: str) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup whose calculate_feature reads feature.options DIRECTLY.

    It never calls options_with_defaults itself: the observed payload therefore proves (or
    disproves) that the framework materialized the declared defaults before the call.
    """

    class MdbBoundaryProbeFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            MDB_CTX_KEY: PropertySpec("A context concrete default.", context=True, default=CTX_DEFAULT),
            MDB_GRP_KEY: PropertySpec("A group concrete default.", context=False, default=GRP_DEFAULT),
            MDB_FALSY_KEY: PropertySpec("Overridable by a falsy explicit value.", context=True, default=FALSY_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({feature_name})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            out: dict[str, list[Any]] = {}
            for feature in features.features:
                out[str(feature.name)] = [
                    {
                        "ctx": feature.options.get(MDB_CTX_KEY),
                        "grp": feature.options.get(MDB_GRP_KEY),
                        "falsy": feature.options.get(MDB_FALSY_KEY),
                    }
                ]
            return out

    return MdbBoundaryProbeFeatureGroup


def _make_explicit_none_probe_fg(feature_name: str) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup pairing an opted-in explicit-None key with a plain default."""

    class MdbExplicitNoneProbeFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            MDB_EN_KEY: PropertySpec(
                "Opted-in: an explicit None is honored.", context=True, default=EN_DEFAULT, allow_explicit_none=True
            ),
            MDB_SIBLING_KEY: PropertySpec("A plain context concrete default.", context=True, default=SIBLING_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({feature_name})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            out: dict[str, list[Any]] = {}
            for feature in features.features:
                out[str(feature.name)] = [
                    {
                        "en": feature.options.get(MDB_EN_KEY),
                        "sibling": feature.options.get(MDB_SIBLING_KEY),
                    }
                ]
            return out

    return MdbExplicitNoneProbeFeatureGroup


def _make_unit_group_default_fg() -> type[FeatureGroup]:
    """A throwaway FeatureGroup with a single group concrete default, for the FeatureSet unit tests."""

    class MdbUnitGroupDefaultFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            MDB_GRP_KEY: PropertySpec("A group concrete default.", context=False, default=GRP_DEFAULT),
        }

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return data

    return MdbUnitGroupDefaultFeatureGroup


def _make_unit_context_default_fg() -> type[FeatureGroup]:
    """A throwaway FeatureGroup with a single context concrete default, for the collapse unit test."""

    class MdbUnitContextDefaultFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            MDB_CTX_KEY: PropertySpec("A context concrete default.", context=True, default=CTX_DEFAULT),
        }

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return data

    return MdbUnitContextDefaultFeatureGroup


def _make_unit_no_defaults_fg() -> type[FeatureGroup]:
    """A throwaway FeatureGroup declaring no concrete defaults (NO_DEFAULT only)."""

    class MdbUnitNoDefaultsFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {
            MDB_REQ_KEY: PropertySpec("Required by omission (NO_DEFAULT).", context=True),
        }

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return data

    return MdbUnitNoDefaultsFeatureGroup


def _single_row(frame: Any, column: str) -> Any:
    """Extract the single payload row, tolerant of columnar dict or list-of-row-dicts results."""
    if isinstance(frame, dict):
        values = list(frame[column])
    else:
        values = [row[column] for row in frame]
    assert len(values) == 1, f"expected exactly one row for {column}, got {values!r}"
    return values[0]


def _run_probe(
    make_fg: Callable[[str], type[FeatureGroup]], feature_name: str, options: Options | None = None
) -> dict[str, Any]:
    """Run one tiny end-to-end computation and return the observed payload of the single feature.

    The probe class and collector stay locals of THIS frame, so a failing assert in the
    caller never pins the throwaway class in a traceback and the no-leak guard stays green.
    """
    fg = make_fg(feature_name)
    collector = PluginCollector.enabled_feature_groups({fg})
    results = mloda.run_all(
        [Feature(feature_name, options if options is not None else Options())],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
    )
    assert len(results) == 1, f"expected exactly one result frame, got: {results!r}"
    payload = _single_row(results[0], feature_name)
    assert isinstance(payload, dict)
    return payload


def _materialize(feature_set: FeatureSet, make_fg: Callable[[], type[FeatureGroup]]) -> None:
    """Invoke ``FeatureSet.materialize_option_defaults`` directly; the method now lives on FeatureSet.

    The direct typed call retires the dynamic ``getattr`` seam the RED phase needed while the
    method was still absent.
    """
    feature_set.materialize_option_defaults(make_fg())


class TestRuntimeDefaultMaterializationEndToEnd:
    """calculate_feature observes declared defaults through the real engine (the #796 regression)."""

    def test_declared_defaults_visible_in_calculate_feature(self) -> None:
        """Absent context and group keys with concrete defaults are materialized before calculate_feature."""
        observed = _run_probe(_make_boundary_probe_fg, "mdb_e2e_defaults_feature")
        assert observed["ctx"] == CTX_DEFAULT, f"context default not materialized at runtime: {observed!r}"
        assert observed["grp"] == GRP_DEFAULT, f"group default not materialized at runtime: {observed!r}"

    def test_explicit_and_falsy_values_win_while_absent_key_materializes(self) -> None:
        """Explicit (even falsy) values are never overridden, while the untouched key still gets its default."""
        options = Options(context={MDB_CTX_KEY: "mdb_explicit", MDB_FALSY_KEY: ""})
        observed = _run_probe(_make_boundary_probe_fg, "mdb_e2e_explicit_feature", options)
        assert observed["ctx"] == "mdb_explicit", f"explicit value was overridden: {observed!r}"
        assert observed["falsy"] == "", f"falsy explicit value was overridden: {observed!r}"
        assert observed["grp"] == GRP_DEFAULT, f"absent group key not materialized in the same run: {observed!r}"

    def test_opted_in_explicit_none_retained_while_sibling_default_materializes(self) -> None:
        """An allow_explicit_none=True explicit None survives materialization; the sibling default is filled."""
        options = Options(context={MDB_EN_KEY: None})
        observed = _run_probe(_make_explicit_none_probe_fg, "mdb_e2e_explicit_none_feature", options)
        assert observed["sibling"] == SIBLING_DEFAULT, f"sibling default not materialized: {observed!r}"
        assert observed["en"] is None, f"opted-in explicit None was overwritten by the default: {observed!r}"


class _CapturingExtender(Extender):
    """Captures the ``features`` argument flowing through FEATURE_GROUP_CALCULATE_FEATURE."""

    def __init__(self) -> None:
        self.captured: list[Any] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        # _invoke_extender passes (data, features); capture the FeatureSet the plugin will see.
        self.captured.append(args[1])
        return func(*args, **kwargs)


class TestExtenderPathObservesMaterializedOptions:
    """run_calculate_feature materializes defaults BEFORE dispatching to a calculate-feature extender."""

    def test_extender_receives_feature_set_with_materialized_defaults(self) -> None:
        """The FeatureSet handed to the extender already carries the declared defaults."""
        extender = _CapturingExtender()
        cfw = PythonDictFramework(ParallelizationMode.SYNC, frozenset(), uuid4(), function_extender={extender})
        feature = Feature("mdb_extender_feature")
        feature_set = FeatureSet()
        feature_set.add(feature)

        # The probe class is created inline (no local binding), so a failing assert cannot pin it.
        cfw.run_calculate_feature(_make_boundary_probe_fg("mdb_extender_feature"), feature_set)

        assert len(extender.captured) == 1, "the extender must wrap exactly one calculate_feature call"
        captured = extender.captured[0]
        assert isinstance(captured, FeatureSet)
        observed_ctx = [f.options.get(MDB_CTX_KEY) for f in captured.features]
        observed_grp = [f.options.get(MDB_GRP_KEY) for f in captured.features]
        assert observed_ctx == [CTX_DEFAULT], f"extender saw pre-default context options: {observed_ctx!r}"
        assert observed_grp == [GRP_DEFAULT], f"extender saw pre-default group options: {observed_grp!r}"


class TestFeatureSetMaterializeOptionDefaults:
    """Unit contract of FeatureSet.materialize_option_defaults (RED: the method does not exist yet)."""

    def test_group_default_fill_rebinds_feature_and_feature_set_options(self) -> None:
        """The fill lands on the feature, the set stays coherent, and self.options reflects the fill."""
        feature = Feature("mdb_unit_fill_feature")
        original_uuid = feature.uuid
        feature_set = FeatureSet()
        feature_set.add(feature)

        _materialize(feature_set, _make_unit_group_default_fg)

        assert feature.options.get(MDB_GRP_KEY) == GRP_DEFAULT
        assert len(feature_set.features) == 1
        # Feature.__hash__ includes options: only a rebuilt set answers fresh membership correctly.
        assert feature in feature_set.features, "features set was not rebuilt after rebinding options"
        assert feature.uuid == original_uuid
        assert feature_set.options is not None
        assert feature_set.options.get(MDB_GRP_KEY) == GRP_DEFAULT, "self.options stayed the stale pre-default object"
        assert feature_set.options is feature.options, "self.options lost its alias to the feature's options"

    def test_shared_options_object_stays_shared_after_fill(self) -> None:
        """Two features sharing ONE Options object still share one materialized object afterwards."""
        shared = Options()
        feature_a = Feature("mdb_unit_alias_a", shared)
        feature_b = Feature("mdb_unit_alias_b", shared)
        feature_set = FeatureSet([feature_a, feature_b])

        _materialize(feature_set, _make_unit_group_default_fg)

        assert len(feature_set.features) == 2
        assert feature_a.options is feature_b.options, "aliased Options diverged during materialization"
        assert feature_a.options.get(MDB_GRP_KEY) == GRP_DEFAULT
        assert feature_a.options is not shared, "the fill must rebind to a new Options, not mutate in place"

    def test_no_concrete_defaults_is_identity_noop(self) -> None:
        """A feature group without concrete defaults leaves every Options object identical (is)."""
        options = Options(context={"mdb_unrelated": "kept"})
        feature = Feature("mdb_unit_noop_feature", options)
        feature_set = FeatureSet([feature])

        _materialize(feature_set, _make_unit_no_defaults_fg)

        rebound = next(iter(feature_set.features))
        assert rebound.options is options, "no-defaults materialization must be an identity no-op"
        assert feature_set.options is options
        assert rebound.options.get("mdb_unrelated") == "kept"

    def test_original_options_object_not_mutated(self) -> None:
        """Materialization rebinds; the original Options object never gains the filled key."""
        original = Options()
        feature = Feature("mdb_unit_nomutate_feature", original)
        feature_set = FeatureSet([feature])

        _materialize(feature_set, _make_unit_group_default_fg)

        assert MDB_GRP_KEY not in original.group, "the input Options was mutated"
        assert MDB_GRP_KEY not in original.context, "the input Options was mutated"

    def test_collapse_of_same_name_twins_raises_actionable_duplicate_message(self) -> None:
        """Filling a context default on one twin of a same-name pair collapses the set: the ValueError
        must name the duplicate-feature cause and list the affected names, not blame an upstream invariant."""
        twin_name = "mdb_unit_twin_feature"
        explicit = Feature(twin_name, Options(context={MDB_CTX_KEY: CTX_DEFAULT}))
        absent = Feature(twin_name, Options())
        feature_set = FeatureSet([explicit, absent])
        # Precondition: group-only hash plus context-aware eq lets the twins coexist before the fill.
        assert len(feature_set.features) == 2

        with pytest.raises(ValueError) as excinfo:
            _materialize(feature_set, _make_unit_context_default_fg)

        message = str(excinfo.value)
        # Drop the captured traceback before the message asserts: a failing assert would otherwise carry the
        # probe class through teardown via excinfo and trip this module's no-leak guard.
        del excinfo
        assert "collapsed duplicate features" in message, f"message must name the duplicate-collapse cause: {message}"
        assert twin_name in message, f"message must list the affected duplicate name: {message}"
