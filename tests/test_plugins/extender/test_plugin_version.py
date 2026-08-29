"""Tests for resolve_plugin_version, the owning-distribution lookup used to
populate HookContext.plugin_version.
"""

import importlib.metadata
from typing import Any

import pytest

import mloda.core.abstract_plugins.plugin_version as plugin_version_module
from mloda.core.abstract_plugins.plugin_version import resolve_plugin_version


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


class TestResolvePluginVersionCaching:
    """resolve_plugin_version must memoize the per-module importlib.metadata read (Bug 3)."""

    def test_repeated_calls_for_same_module_read_metadata_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolve_plugin_version.cache_clear()
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
    """A shared namespace owned by multiple distributions must resolve each module correctly (Bug 3)."""

    @staticmethod
    def _install_fake_multi_distribution_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(plugin_version_module, "_packages_distributions_cache", None)
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

    def test_falls_back_to_first_candidate_after_attempting_to_match_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no distribution's files match (or files is None), fall back to the first
        candidate's version -- but only after actually attempting the file-based match."""
        monkeypatch.setattr(plugin_version_module, "_packages_distributions_cache", None)
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

        assert result == "9.0.0-x", "No file matched; must fall back to the first candidate's version, not raise/None."
        assert distribution_calls, (
            "resolve_plugin_version must attempt to inspect each candidate's files before "
            "falling back, not skip straight to distributions[0] as today's uncached code does."
        )
