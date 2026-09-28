"""Tests for resolve_plugin_version, the owning-distribution lookup used to
populate HookContext.plugin_version.
"""

import importlib.metadata
from collections.abc import Iterator
from typing import Any

import pytest

import mloda.core.abstract_plugins.plugin_version as plugin_version_module
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.plugin_version import resolve_plugin_version
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


@pytest.fixture(autouse=True)
def _clear_plugin_version_caches() -> Iterator[None]:
    """Reset all module-level caches before and after every test so fakes never leak between tests."""
    resolve_plugin_version.cache_clear()
    plugin_version_module._read_packages_distributions.cache_clear()
    plugin_version_module._read_distribution.cache_clear()
    yield
    resolve_plugin_version.cache_clear()
    plugin_version_module._read_packages_distributions.cache_clear()
    plugin_version_module._read_distribution.cache_clear()


class TestResolvePluginVersion:
    """resolve_plugin_version resolves the installed distribution version owning a module."""

    def test_resolves_mloda_module_to_installed_mloda_version(self) -> None:
        result = resolve_plugin_version("mloda.core.abstract_plugins.function_extender")

        assert result is not None
        assert result != ""
        assert result == importlib.metadata.version("mloda")

    def test_returns_none_for_unknown_top_level_package(self) -> None:
        result = resolve_plugin_version("definitely_not_a_real_top_level_package_xyz.submodule")

        assert result is None

    def test_repeated_calls_with_different_modules_resolve_independently(self) -> None:
        first = resolve_plugin_version("mloda.core.abstract_plugins.function_extender")
        second = resolve_plugin_version("definitely_not_a_real_top_level_package_xyz.submodule")

        assert first == importlib.metadata.version("mloda")
        assert second is None


class _FakeDistribution:
    """Minimal Distribution-like stand-in exposing only what the ownership resolution reads."""

    def __init__(self, files: list[Any] | None) -> None:
        self.files = files


class _RaisingFilesDistribution:
    """Distribution stand-in whose `files` property raises, mimicking a broken metadata read."""

    @property
    def files(self) -> list[Any] | None:
        raise TypeError("boom")


class TestResolvePluginVersionCaching:
    """resolve_plugin_version must memoize the per-module importlib.metadata read."""

    def test_repeated_calls_for_same_module_read_metadata_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}
        real_version = importlib.metadata.version

        def counting_version(name: str) -> str:
            call_count["n"] += 1
            return real_version(name)

        monkeypatch.setattr(importlib.metadata, "version", counting_version)

        resolve_plugin_version("mloda.core.abstract_plugins.function_extender")
        resolve_plugin_version("mloda.core.abstract_plugins.function_extender")

        assert call_count["n"] == 1, "The second call must be served from cache, not re-read metadata."


class TestResolvePluginVersionOwnershipResolution:
    """A shared namespace owned by multiple distributions must resolve each module correctly."""

    @staticmethod
    def _install_fake_multi_distribution_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"faketop": ["dist-a", "dist-b"]},
        )

        def fake_distribution(name: str) -> _FakeDistribution:
            files_by_dist = {
                "dist-a": [importlib.metadata.PackagePath("faketop/module_a.py")],
                "dist-b": [importlib.metadata.PackagePath("faketop/module_b.py")],
            }
            return _FakeDistribution(files_by_dist[name])

        monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

        def fake_version(name: str) -> str:
            return {"dist-a": "1.0.0-a", "dist-b": "2.0.0-b"}[name]

        monkeypatch.setattr(importlib.metadata, "version", fake_version)

    def test_each_module_resolves_to_its_own_owning_distribution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake_multi_distribution_namespace(monkeypatch)

        result_a = resolve_plugin_version("faketop.module_a")
        result_b = resolve_plugin_version("faketop.module_b")

        assert result_a == "1.0.0-a"
        assert result_b == "2.0.0-b", (
            "module_b is shipped by dist-b, not dist-a (distributions[0]); picking the first "
            "candidate blindly reports the wrong distribution's version."
        )

    def test_returns_none_when_no_candidate_owns_the_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no distribution's files match (or files is None), resolve_plugin_version returns None,
        but only after attempting the file-based match against every candidate."""
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"othertop": ["dist-x", "dist-y"]},
        )

        distribution_calls: list[str] = []

        def fake_distribution(name: str) -> _FakeDistribution:
            distribution_calls.append(name)
            files_by_dist: dict[str, list[Any] | None] = {
                "dist-x": None,
                "dist-y": [importlib.metadata.PackagePath("unrelated/other_module.py")],
            }
            return _FakeDistribution(files_by_dist[name])

        monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

        def fake_version(name: str) -> str:
            return {"dist-x": "9.0.0-x", "dist-y": "8.0.0-y"}[name]

        monkeypatch.setattr(importlib.metadata, "version", fake_version)

        result = resolve_plugin_version("othertop.module_z")

        assert result is None
        assert distribution_calls == ["dist-x", "dist-y"], (
            "resolve_plugin_version must attempt to inspect every candidate's files before "
            "giving up, not skip straight to a fallback."
        )


class TestDistributionOwningManifestMatching:
    """_distribution_owning recognizes package, plain-module, and extension-module manifest entries."""

    @pytest.mark.parametrize(
        ("file_entry", "module_name", "expected"),
        [
            ("faketop/sub/__init__.py", "faketop.sub", "dist-a"),
            ("faketop/ext.py", "faketop.ext", "dist-a"),
            ("faketop/ext.cpython-312-x86_64-linux-gnu.so", "faketop.ext", "dist-a"),
            ("faketop/ext/__init__.py", "faketop.ext", "dist-a"),
            ("faketop/ext/other.py", "faketop.ext", None),
            ("faketop/mod.pyi", "faketop.mod", None),
            ("faketop/mod.py", "faketop.mod", "dist-a"),
            ("faketop/mod.cpython-312-x86_64-linux-gnu.so", "faketop.mod", "dist-a"),
            ("faketop/mod.pyd", "faketop.mod", "dist-a"),
        ],
    )
    def test_manifest_entry_ownership(
        self,
        monkeypatch: pytest.MonkeyPatch,
        file_entry: str,
        module_name: str,
        expected: str | None,
    ) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda name: _FakeDistribution([importlib.metadata.PackagePath(file_entry)]),
        )

        result = plugin_version_module._distribution_owning(module_name, ["dist-a"])

        assert result == expected


class TestOwnsModuleIgnoresPyiStubs:
    """A .pyi stub-only distribution must not be reported as a module's owning distribution."""

    def test_stub_distribution_is_skipped_in_favor_of_the_real_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"faketop": ["stubs-dist", "real-dist"]},
        )

        def fake_distribution(name: str) -> _FakeDistribution:
            files_by_dist = {
                "stubs-dist": [importlib.metadata.PackagePath("faketop/mod.pyi")],
                "real-dist": [importlib.metadata.PackagePath("faketop/mod.py")],
            }
            return _FakeDistribution(files_by_dist[name])

        monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: {"stubs-dist": "1.0.0-stub", "real-dist": "2.0.0-real"}[name],
        )

        result = resolve_plugin_version("faketop.mod")

        assert result == "2.0.0-real"


class TestResolvePluginVersionSingleCandidateSkipsFileManifest:
    def test_single_candidate_does_not_read_file_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"faketop": ["dist-only"]},
        )

        def must_not_be_called(name: str) -> importlib.metadata.Distribution:
            raise AssertionError("must not be called")

        monkeypatch.setattr(importlib.metadata, "distribution", must_not_be_called)
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.3.3")

        result = resolve_plugin_version("faketop.whatever")

        assert result == "3.3.3"


class TestResolvePluginVersionToleratesBrokenFilesProperty:
    def test_candidate_with_raising_files_property_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"faketop": ["dist-bad", "dist-good"]},
        )

        def fake_distribution(name: str) -> Any:
            if name == "dist-bad":
                return _RaisingFilesDistribution()
            return _FakeDistribution([importlib.metadata.PackagePath("faketop/module_g.py")])

        monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: {"dist-bad": "0.0.0-bad", "dist-good": "5.0.0-good"}[name],
        )

        result = resolve_plugin_version("faketop.module_g")

        assert result == "5.0.0-good"


class TestReadDistributionCaching:
    """_read_distribution is cached, so a distribution is read from importlib.metadata at most once."""

    def test_distribution_lookup_cached_across_modules_in_same_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"faketop": ["dist-a", "dist-b"]},
        )

        distribution_calls: list[str] = []

        def fake_distribution(name: str) -> _FakeDistribution:
            distribution_calls.append(name)
            files_by_dist = {
                "dist-a": [importlib.metadata.PackagePath("faketop/module_a.py")],
                "dist-b": [importlib.metadata.PackagePath("faketop/module_b.py")],
            }
            return _FakeDistribution(files_by_dist[name])

        monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: {"dist-a": "1.0.0-a", "dist-b": "2.0.0-b"}[name],
        )

        resolve_plugin_version("faketop.module_a")
        resolve_plugin_version("faketop.module_b")

        assert distribution_calls.count("dist-a") <= 1
        assert distribution_calls.count("dist-b") <= 1


class _PlanTimePluginVersionFeatureGroup(FeatureGroup):
    """Minimal root feature group for plan-time plugin_version resolution tests."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"plan_time_plugin_version_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"plan_time_plugin_version_col": [1, 2, 3]}


_plan_time_enabled = PluginCollector.enabled_feature_groups({_PlanTimePluginVersionFeatureGroup})


class _PluginVersionCapturingExtender(Extender):
    """Records HookContext.current().plugin_version on every wrapped call."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.recorded: list[str | None] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.recorded.append(context.plugin_version)
        return result


class TestPluginVersionResolvedAtPlanTimeUnderSync:
    """HookContext.plugin_version must come from the plan-time resolution, not a hook-time call."""

    def test_hook_context_plugin_version_matches_plan_time_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = "9.9.9-sentinel"
        monkeypatch.setattr(
            "mloda.core.core.engine.resolve_plugin_version",
            lambda module_name: sentinel,
            raising=False,
        )
        extender = _PluginVersionCapturingExtender()

        session = mloda.prepare(
            [Feature(name="plan_time_plugin_version_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_plan_time_enabled,
            function_extender={extender},
        )
        session.run()

        assert extender.recorded, "extender never observed the feature group's calculate_feature call"
        assert all(value == sentinel for value in extender.recorded)


class _PluginVersionCacheInfoRecordingExtender(Extender):
    """Records resolve_plugin_version.cache_info() after every wrapped call, without asserting inside the hook."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.cache_infos: list[Any] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.cache_infos.append(resolve_plugin_version.cache_info())
        return result


class TestPluginVersionNotResolvedDuringExecution:
    """Execution must never call resolve_plugin_version; only plan time (prepare()) may."""

    def test_resolve_plugin_version_cache_untouched_during_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {"tests": ["fake-tests-dist"]},
        )
        extender = _PluginVersionCacheInfoRecordingExtender()

        session = mloda.prepare(
            [Feature(name="plan_time_plugin_version_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_plan_time_enabled,
            function_extender={extender},
        )
        baseline = resolve_plugin_version.cache_info()

        session.run()

        assert extender.cache_infos, "extender never observed the feature group's calculate_feature call"
        assert all(info == baseline for info in extender.cache_infos), (
            "hook-time cache_info() drifted from the plan-time baseline: resolve_plugin_version was "
            "called again during execution instead of being read from RunContext.plugin_versions."
        )
        assert resolve_plugin_version.cache_info() == baseline


class TestPluginVersionNotResolvedWithoutExtenders:
    """With no function_extender registered, no distribution scan may happen at all (plan-time gate)."""

    def test_no_extenders_means_zero_distribution_scans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        def counting_packages_distributions() -> dict[str, list[str]]:
            call_count["n"] += 1
            return {}

        monkeypatch.setattr(importlib.metadata, "packages_distributions", counting_packages_distributions)

        session = mloda.prepare(
            [Feature(name="plan_time_plugin_version_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_plan_time_enabled,
        )
        session.run()

        assert call_count["n"] == 0
