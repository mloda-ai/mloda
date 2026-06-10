"""Tests for the tri-state strict parameter (issue #526, work item 5).

Contract: PluginCollector carries a strict_mode of "off", "warn", or "strict",
defaulting from the MLODA_PLUGIN_REGISTRY_STRICT env var ("off" when unset,
ValueError on invalid values). PreFilterPlugins consults the mode during
resolution: "strict" keeps only FeatureGroups registered in the default
PluginRegistry, "warn" keeps everything but logs unregistered classes,
"off" keeps today's behavior. The engine honors the env var even when no
PluginCollector is passed.

Parallel-safety: other tests define many FeatureGroup subclasses, so all
assertions are membership or absence of this module's own doubles, never
global counts. Registry writes are restored by the autouse conftest fixture;
env writes go through monkeypatch only.
"""

import gc
import inspect
import linecache
import logging
import sys
import textwrap
from abc import abstractmethod
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.prepare.accessible_plugins import PreFilterPlugins, RedefinitionConflictError
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

ENV_VAR = "MLODA_PLUGIN_REGISTRY_STRICT"


class _StrictUnregisteredFG(FeatureGroup):
    """Local double that is never registered in the default registry."""


class _StrictUnregisteredFG2(FeatureGroup):
    """Second local double that is never registered, for warn-aggregation tests."""


class _StrictRegisteredFG(FeatureGroup):
    """Local double that tests register explicitly in the default registry."""


class _StrictConflictBystanderFG(FeatureGroup):
    """Local double registered alongside an unregistered redefinition conflict."""


class _StrictEnabledUnregisteredFG(FeatureGroup):
    """Local double passed as enabled but never registered; strict mode must drop it loudly."""


class _StrictEnabledRegisteredFG(FeatureGroup):
    """Local double passed as enabled and registered, so strict resolution does not end empty."""


class _StrictAbstractInfraFG(FeatureGroup):
    """Abstract local double, never registered; strict and warn must treat it as infrastructure.

    match_feature_group_criteria is overridden to False because the default falls back to
    cls(), which raises TypeError on abstract classes. Real abstract bases override matching
    too; this keeps the double inert for other tests' feature resolution in this worker.
    """

    @abstractmethod
    def _strict_abstract_infra_hook(self) -> None: ...

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        data_access_collection: Any = None,
    ) -> bool:
        return False


def _cfws() -> set[type[ComputeFramework]]:
    return {PythonDictFramework}


def _exec_fg_in_main(class_name: str, body: str, cell_label: str) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into ``__main__`` with linecache-backed source.

    Mirrors the proven Jupyter-cell recipe from
    tests/test_core/test_prepare/test_feature_group_dedup.py so
    ``inspect.getsource`` works for the exec'd class.
    """
    main_mod = sys.modules["__main__"]
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    exec(compile(src, filename, "exec"), main_mod.__dict__)  # nosec B102
    return cast(type[FeatureGroup], main_mod.__dict__[class_name])


def _make_fg_source(class_name: str, feature_name: str, extra_body: str = "") -> str:
    """Build FeatureGroup subclass source for exec into ``__main__``."""
    return f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_

class {class_name}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{feature_name}"}}
{extra_body}
"""


class TestSetStrictModeApi:
    def test_set_strict_mode_is_chainable(self) -> None:
        collector = PluginCollector()
        assert collector.set_strict_mode("warn") is collector

    @pytest.mark.parametrize("mode", ["off", "warn", "strict"])
    def test_set_strict_mode_accepts_valid_values(self, mode: str) -> None:
        collector = PluginCollector().set_strict_mode(mode)
        assert collector.strict_mode == mode

    @pytest.mark.parametrize("mode", ["on", "OFF", "Strict", "", "true", "1"])
    def test_set_strict_mode_rejects_invalid_values(self, mode: str) -> None:
        with pytest.raises(ValueError):
            PluginCollector().set_strict_mode(mode)

    def test_default_is_off_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert PluginCollector().strict_mode == "off"


class TestEnvVarDefault:
    @pytest.mark.parametrize("mode", ["warn", "strict"])
    def test_env_var_sets_default_mode(self, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        monkeypatch.setenv(ENV_VAR, mode)
        assert PluginCollector().strict_mode == mode

    def test_invalid_env_value_raises_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "bogus")
        with pytest.raises(ValueError):
            PluginCollector()

    def test_explicit_set_strict_mode_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "strict")
        collector = PluginCollector().set_strict_mode("off")
        assert collector.strict_mode == "off"

    @pytest.mark.parametrize(("raw", "normalized"), [("WARN", "warn"), ("Strict", "strict")])
    def test_env_var_value_is_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, normalized: str
    ) -> None:
        """The env var is user input, so casing must be normalized to lowercase.

        set_strict_mode() stays exact-match; only the env var path normalizes.
        """
        monkeypatch.setenv(ENV_VAR, raw)
        assert PluginCollector().strict_mode == normalized


class TestEngineStrictModeOff:
    def test_off_mode_keeps_unregistered_feature_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Non-regression guard: today's behavior, unregistered subclasses stay accessible.
        monkeypatch.delenv(ENV_VAR, raising=False)
        accessible = PreFilterPlugins(_cfws(), PluginCollector()).get_accessible_plugins()
        assert _StrictUnregisteredFG in accessible


class TestEngineStrictModeStrict:
    def test_strict_mode_keeps_only_registered_feature_groups(self) -> None:
        register_plugin(_StrictRegisteredFG)
        collector = PluginCollector().set_strict_mode("strict")
        accessible = PreFilterPlugins(_cfws(), collector).get_accessible_plugins()
        assert _StrictRegisteredFG in accessible
        assert _StrictUnregisteredFG not in accessible

    def test_strict_filter_runs_before_dedup_so_unregistered_conflicts_cannot_poison(self) -> None:
        """An unregistered same-name different-source pair must not break strict resolution.

        Strict mode must filter unregistered classes BEFORE dedup runs, so a stale
        redefinition conflict among unregistered classes never reaches dedup.
        """
        qualname = "StrictFG_RedConflictPair"
        feature_name = "strict_red_conflict_feature_unique_xyz"
        v1 = _exec_fg_in_main(qualname, _make_fg_source(qualname, feature_name), "cell-strict-red-v1")
        v2 = _exec_fg_in_main(
            qualname,
            _make_fg_source(qualname, feature_name, extra_body="    def extra_method(self):\n        return 42\n"),
            "cell-strict-red-v2",
        )
        assert v1 is not v2, "sanity: two distinct class objects with the same (module, qualname)"

        register_plugin(_StrictConflictBystanderFG)
        collector = PluginCollector().set_strict_mode("strict")

        accessible = None
        conflict_error = ""
        try:
            accessible = PreFilterPlugins(_cfws(), collector).get_accessible_plugins()
        except RedefinitionConflictError as exc:
            conflict_error = str(exc)
        finally:
            # Drop every strong ref to the conflicting pair so it cannot poison
            # other tests in this worker process via FeatureGroup.__subclasses__().
            sys.modules["__main__"].__dict__.pop(qualname, None)
            linecache.cache.pop("<cell-strict-red-v1>", None)
            linecache.cache.pop("<cell-strict-red-v2>", None)
            del v1, v2
            gc.collect()

        if accessible is None:
            pytest.fail(
                "strict mode must filter unregistered classes before dedup; "
                f"got RedefinitionConflictError: {conflict_error}"
            )
        assert _StrictConflictBystanderFG in accessible
        assert not any(fg.__qualname__ == qualname for fg in accessible), (
            "unregistered conflicting classes must not appear in strict accessible plugins"
        )


class TestEngineStrictModeEmptyResult:
    def test_strict_empty_result_error_mentions_strict_mode_and_fix(self) -> None:
        """With strict mode on and an empty registry, the error must teach the strict-mode fix.

        At least one accessible FeatureGroup subclass exists (this module's doubles),
        so the empty result is caused by strict filtering, not by missing plugins.
        The generic "Did you call PluginLoader.all()?" hint would mislead here.
        """
        PluginRegistry.default().clear()
        collector = PluginCollector().set_strict_mode("strict")
        with pytest.raises(ValueError) as exc_info:
            PreFilterPlugins(_cfws(), collector)
        message = str(exc_info.value)
        assert "strict" in message, "empty-result error under strict mode must mention strict mode"
        assert "register" in message, "empty-result error under strict mode must say how to fix it"
        assert "PluginLoader.all" not in message, (
            "strict-mode empty result must not blame plugin loading when subclasses exist"
        )


class TestEngineStrictModeEnabledButUnregistered:
    def test_strict_mode_warns_when_enabled_class_is_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Explicitly enabled but unregistered classes silently vanish today; that must warn."""
        register_plugin(_StrictEnabledRegisteredFG)
        collector = PluginCollector().set_strict_mode("strict")
        collector.add_enabled_feature_group_classes({_StrictEnabledUnregisteredFG, _StrictEnabledRegisteredFG})
        with caplog.at_level(logging.WARNING):
            accessible = PreFilterPlugins(_cfws(), collector).get_accessible_plugins()
        assert _StrictEnabledRegisteredFG in accessible
        assert _StrictEnabledUnregisteredFG not in accessible
        matching = [
            record
            for record in caplog.records
            if _StrictEnabledUnregisteredFG.__name__ in record.getMessage()
            and "enabled" in record.getMessage()
            and "not registered" in record.getMessage()
        ]
        assert matching, (
            "strict mode must warn that an explicitly enabled class was dropped because it is not registered"
        )


class TestEngineStrictModeWarn:
    def test_warn_mode_resolution_matches_off_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        off_accessible = PreFilterPlugins(_cfws(), PluginCollector()).get_accessible_plugins()
        warn_collector = PluginCollector().set_strict_mode("warn")
        warn_accessible = PreFilterPlugins(_cfws(), warn_collector).get_accessible_plugins()
        assert _StrictUnregisteredFG in warn_accessible
        assert set(warn_accessible.keys()) == set(off_accessible.keys())

    def test_warn_mode_logs_unregistered_class(self, caplog: pytest.LogCaptureFixture) -> None:
        collector = PluginCollector().set_strict_mode("warn")
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(_cfws(), collector)
        matching = [
            record
            for record in caplog.records
            if _StrictUnregisteredFG.__name__ in record.getMessage() and "not registered" in record.getMessage()
        ]
        assert matching, "warn mode must log a warning naming the unregistered class and saying 'not registered'"

    def test_warn_mode_aggregates_all_unregistered_classes_into_one_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One PreFilterPlugins construction must emit exactly ONE warning record.

        The single aggregated record names every unregistered class instead of
        emitting one record per class.
        """
        collector = PluginCollector().set_strict_mode("warn")
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(_cfws(), collector)
        records = [rec for rec in caplog.records if rec.name == "mloda.core.prepare.accessible_plugins"]
        assert len(records) == 1, (
            f"warn mode must aggregate unregistered classes into one record per construction, got {len(records)}"
        )
        message = records[0].getMessage()
        assert _StrictUnregisteredFG.__name__ in message
        assert _StrictUnregisteredFG2.__name__ in message
        assert "not registered" in message

    def test_warn_mode_warns_once_per_process_for_already_warned_classes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two constructions in a row must emit the aggregated warning only once.

        Contract: accessible_plugins keeps a module-level set _warned_unregistered of
        already-warned class names; the second construction finds every name already
        warned and stays silent. The autouse conftest fixture clears the set per test,
        so the per-construction warn tests above stay independent.
        """
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(_cfws(), PluginCollector().set_strict_mode("warn"))
            PreFilterPlugins(_cfws(), PluginCollector().set_strict_mode("warn"))
        records = [
            rec
            for rec in caplog.records
            if rec.name == "mloda.core.prepare.accessible_plugins" and "not registered" in rec.getMessage()
        ]
        assert len(records) == 1, (
            f"warn mode must deduplicate already-warned class names per process, got {len(records)} records"
        )

    def test_warn_mode_reports_same_name_classes_from_different_modules_separately(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn-mode dedup and reporting must key on module-qualified names, not bare __name__.

        Two DISTINCT unregistered FeatureGroup classes that share __name__ but live in
        different modules must BOTH appear in the aggregated warning, identified as
        f"{module}:{qualname}". Keying on bare __name__ collapses them into one identity.
        """
        fg_a = cast(
            type[FeatureGroup],
            type("WarnDedupSameNameFG", (FeatureGroup,), {"__module__": "warn_dedup_fake_module_a"}),
        )
        fg_b = cast(
            type[FeatureGroup],
            type("WarnDedupSameNameFG", (FeatureGroup,), {"__module__": "warn_dedup_fake_module_b"}),
        )
        try:
            assert fg_a is not fg_b, "sanity: two distinct class objects sharing __name__"

            collector = PluginCollector().set_strict_mode("warn")
            with caplog.at_level(logging.WARNING):
                PreFilterPlugins(_cfws(), collector)
            records = [
                rec
                for rec in caplog.records
                if rec.name == "mloda.core.prepare.accessible_plugins" and "not registered" in rec.getMessage()
            ]
            message = " ".join(rec.getMessage() for rec in records)
        finally:
            # Drop strong refs even when an assertion fails or PreFilterPlugins raises,
            # so the fake-module classes cannot leak into FeatureGroup.__subclasses__()
            # for other tests in this worker. caplog records hold only strings, so the
            # message assertions below stay valid after the classes are collected.
            del fg_a, fg_b
            gc.collect()

        assert "warn_dedup_fake_module_a:WarnDedupSameNameFG" in message, (
            "warn mode must report unregistered classes with module-qualified identifiers"
        )
        assert "warn_dedup_fake_module_b:WarnDedupSameNameFG" in message, (
            "warn mode must report BOTH same-name classes from different modules, not collapse them on bare __name__"
        )


class TestEngineStrictModeAbstractClasses:
    """Abstract classes never auto-register (they are infrastructure, not plugins), so strict and
    warn modes must leave them alone instead of filtering or flagging them as unregistered."""

    def test_strict_mode_keeps_unregistered_abstract_classes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert inspect.isabstract(_StrictAbstractInfraFG), "sanity: the local double is abstract"

        monkeypatch.delenv(ENV_VAR, raising=False)
        off_accessible = PreFilterPlugins(_cfws(), PluginCollector()).get_accessible_plugins()
        assert _StrictAbstractInfraFG in off_accessible, "sanity: off mode resolves abstract subclasses"

        register_plugin(_StrictRegisteredFG)
        collector = PluginCollector().set_strict_mode("strict")
        accessible = PreFilterPlugins(_cfws(), collector).get_accessible_plugins()
        assert _StrictAbstractInfraFG in accessible, (
            "strict mode must not filter abstract infrastructure classes (parity with off mode); "
            "abstract classes cannot be loader-registered, so filtering them breaks resolution"
        )
        assert _StrictUnregisteredFG not in accessible, (
            "sanity: strict mode still filters concrete unregistered classes"
        )

    def test_warn_mode_does_not_name_abstract_classes(self, caplog: pytest.LogCaptureFixture) -> None:
        assert inspect.isabstract(_StrictAbstractInfraFG), "sanity: the local double is abstract"

        collector = PluginCollector().set_strict_mode("warn")
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(_cfws(), collector)
        message = " ".join(
            rec.getMessage()
            for rec in caplog.records
            if rec.name == "mloda.core.prepare.accessible_plugins" and "not registered" in rec.getMessage()
        )
        assert _StrictUnregisteredFG.__name__ in message, (
            "sanity: the aggregated warning still names concrete unregistered classes"
        )
        abstract_identifier = f"{_StrictAbstractInfraFG.__module__}:{_StrictAbstractInfraFG.__qualname__}"
        assert abstract_identifier not in message, (
            "warn mode must not name abstract infrastructure classes as unregistered plugins; "
            "they cannot be registered, so warning about them is pure noise"
        )


class TestEnvWithoutCollector:
    def test_env_strict_applies_without_collector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "strict")
        register_plugin(_StrictRegisteredFG)
        accessible = PreFilterPlugins(_cfws(), None).get_accessible_plugins()
        assert _StrictRegisteredFG in accessible
        assert _StrictUnregisteredFG not in accessible

    def test_env_unset_keeps_unregistered_without_collector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Non-regression guard: today's behavior without collector and without env var.
        monkeypatch.delenv(ENV_VAR, raising=False)
        accessible = PreFilterPlugins(_cfws(), None).get_accessible_plugins()
        assert _StrictUnregisteredFG in accessible
