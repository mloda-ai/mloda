"""Tests for the mloda core version helper and the facade-level ``__version__`` attributes."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def reset_version_cache() -> Iterator[None]:
    """Restores the memoization cache of mloda.core.version so tests do not poison the process."""
    import mloda.core.version as version_module

    previous = version_module._mloda_version_cache
    version_module._mloda_version_cache = None
    try:
        yield
    finally:
        version_module._mloda_version_cache = previous


def test_get_mloda_version_matches_distribution_metadata() -> None:
    from mloda.core.version import get_mloda_version

    assert get_mloda_version() == importlib.metadata.version("mloda")


def test_get_mloda_version_is_memoized(reset_version_cache: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import mloda.core.version as version_module

    calls: list[str] = []
    real_version = importlib.metadata.version

    def counting_version(name: str) -> str:
        calls.append(name)
        return real_version(name)

    monkeypatch.setattr(version_module.importlib.metadata, "version", counting_version)

    first = version_module.get_mloda_version()
    second = version_module.get_mloda_version()

    assert first == second
    assert version_module._mloda_version_cache == first
    assert len(calls) == 1, f"expected metadata lookup to be memoized, got {len(calls)} lookups"


def test_get_mloda_version_falls_back_when_distribution_missing(
    reset_version_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mloda.core.version as version_module

    def raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version_module.importlib.metadata, "version", raise_not_found)

    assert version_module.get_mloda_version() == "0.0.0"


@pytest.mark.parametrize("module_name", ["mloda.user", "mloda.provider", "mloda.steward"])
def test_facade_modules_expose_version(module_name: str) -> None:
    import importlib

    module: Any = importlib.import_module(module_name)

    assert hasattr(module, "__version__"), f"{module_name} must expose a module-level __version__"
    assert isinstance(module.__version__, str)
    assert module.__version__ == importlib.metadata.version("mloda")


def test_base_feature_group_version_uses_helper() -> None:
    from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion
    from mloda.core.version import get_mloda_version

    assert BaseFeatureGroupVersion.mloda_version() == get_mloda_version()
    assert BaseFeatureGroupVersion.mloda_version() == importlib.metadata.version("mloda")


def test_mloda_stays_a_namespace_package() -> None:
    """Regression guard: adding __version__ to facades must not introduce mloda/__init__.py."""
    import mloda

    assert mloda.__file__ is None, f"mloda must stay a PEP 420 namespace package, got __file__={mloda.__file__}"
    assert type(mloda.__path__).__name__ == "_NamespacePath"
