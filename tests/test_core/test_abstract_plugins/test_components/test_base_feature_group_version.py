import importlib.metadata
import inspect
from typing import Any

import pytest

import mloda.core.version as version_module
from mloda.provider import BaseFeatureGroupVersion, FeatureGroup
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1


class TestBaseFeatureGroupVersion:
    def test_version_composite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch importlib.metadata.version to a known value and reset the
        # mloda_version memo so the patched value is actually recomputed even
        # when an earlier test in the same worker already warmed the cache.
        # monkeypatch teardown restores both the version function and the
        # pre-test cache value, so no "1.2.3" poisoning leaks to other tests.
        monkeypatch.setattr(importlib.metadata, "version", lambda pkg: "1.2.3")
        monkeypatch.setattr(version_module, "_mloda_version_cache", None)

        composite = BaseTestFeatureGroup1.version()
        # Expected format: "1.2.3-{module_name}-{hash}"
        expected_prefix = f"1.2.3-{BaseTestFeatureGroup1.__module__}-"
        assert composite.startswith(expected_prefix), (
            f"Composite version should start with '{expected_prefix}', got '{composite}'"
        )

        # Split the composite string into parts.
        parts = composite.split("-")
        assert len(parts) == 3, "Composite version should have three parts separated by '-'"

        # Check that the hash part is 64 hex characters (SHA-256 produces 64 hex digits).
        hash_val = parts[2]
        assert len(hash_val) == 64, "Hash length should be 64 characters"
        # Verify that the hash is valid hexadecimal.
        int(hash_val, 16)

    def test_invalid_target_class_for_hash(self) -> None:
        # Calling BaseFeatureGroupVersion.class_source_hash with a class not inheriting from FeatureGroup should raise a ValueError.
        with pytest.raises(ValueError):
            BaseFeatureGroupVersion.class_source_hash(str)


def _define_same_name_class_variant_a() -> type[FeatureGroup]:
    class RedefinedCacheProbeFeatureGroup(FeatureGroup):
        """Variant A of a redefined class, body deliberately distinct."""

    return RedefinedCacheProbeFeatureGroup


def _define_same_name_class_variant_b() -> type[FeatureGroup]:
    class RedefinedCacheProbeFeatureGroup(FeatureGroup):
        """Variant B of a redefined class, body deliberately different from variant A."""

    return RedefinedCacheProbeFeatureGroup


class TestClassSourceHashCaching:
    """Caching contract for BaseFeatureGroupVersion.class_source_hash.

    Within one process, the source of a given class OBJECT is read at most
    once; later calls return the cached hash. Different class objects are
    cached independently, including a redefined class with the same name.
    """

    def _install_counting_getsource(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Wraps inspect.getsource with a call recorder.

        The module under test does ``import inspect`` and resolves
        ``inspect.getsource`` at call time, so patching the shared ``inspect``
        module attribute intercepts its source reads.
        """
        calls: list[object] = []
        real_getsource = inspect.getsource

        def counting_getsource(obj: Any) -> str:
            calls.append(obj)
            return real_getsource(obj)

        monkeypatch.setattr(inspect, "getsource", counting_getsource)
        return calls

    def test_class_source_hash_is_cached_per_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._install_counting_getsource(monkeypatch)

        class CachedProbeFeatureGroup(FeatureGroup):
            """Local feature group used to observe how often its source is read."""

        first = BaseFeatureGroupVersion.class_source_hash(CachedProbeFeatureGroup)
        second = BaseFeatureGroupVersion.class_source_hash(CachedProbeFeatureGroup)

        assert first == second
        reads = [obj for obj in calls if obj is CachedProbeFeatureGroup]
        assert len(reads) <= 1, (
            f"Source of CachedProbeFeatureGroup was read {len(reads)} times across two "
            "class_source_hash calls; the second call must be served from the per-class cache."
        )

    def test_class_source_hash_distinct_classes_hashed_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._install_counting_getsource(monkeypatch)

        class IndependentProbeFeatureGroupOne(FeatureGroup):
            """First independent probe class with its own body."""

        class IndependentProbeFeatureGroupTwo(FeatureGroup):
            """Second independent probe class with a deliberately different body."""

        hash_one = BaseFeatureGroupVersion.class_source_hash(IndependentProbeFeatureGroupOne)
        hash_two = BaseFeatureGroupVersion.class_source_hash(IndependentProbeFeatureGroupTwo)

        assert hash_one != hash_two
        reads_one = [obj for obj in calls if obj is IndependentProbeFeatureGroupOne]
        reads_two = [obj for obj in calls if obj is IndependentProbeFeatureGroupTwo]
        assert len(reads_one) == 1, "First class must trigger exactly one source read"
        assert len(reads_two) == 1, "Second class must trigger its own source read"

    def test_class_source_hash_redefined_class_gets_fresh_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._install_counting_getsource(monkeypatch)

        first_class = _define_same_name_class_variant_a()
        second_class = _define_same_name_class_variant_b()
        assert first_class.__name__ == second_class.__name__
        assert first_class is not second_class

        hash_first = BaseFeatureGroupVersion.class_source_hash(first_class)
        hash_first_again = BaseFeatureGroupVersion.class_source_hash(first_class)
        assert hash_first == hash_first_again
        first_reads = [obj for obj in calls if obj is first_class]
        assert len(first_reads) == 1, (
            f"Source of the first class object was read {len(first_reads)} times across two calls; "
            "the second call must hit the cache."
        )

        hash_second = BaseFeatureGroupVersion.class_source_hash(second_class)
        assert hash_second != hash_first, (
            "A new class object with the same name must be hashed fresh, not served from the old object's cache entry."
        )
        second_reads = [obj for obj in calls if obj is second_class]
        assert len(second_reads) == 1, "The redefined class object must trigger its own source read"


class TestMlodaVersionMemoization:
    """Memoization contract for BaseFeatureGroupVersion.mloda_version.

    Within one process, the underlying importlib.metadata lookup runs at most
    once; all subsequent mloda_version calls return the memoized string. Docs
    enumeration calls mloda_version once per FeatureGroup subclass per
    enumeration, and the repeated metadata parsing is a hot path.
    """

    def test_mloda_version_metadata_lookup_is_memoized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Earlier tests in the same process may already have populated the
        # memo; reset the module-level cache so this test observes the first
        # lookup. raising=False creates the attribute if it does not exist yet.
        monkeypatch.setattr(version_module, "_mloda_version_cache", None, raising=False)

        calls: list[str] = []
        real_metadata_version = importlib.metadata.version

        def counting_metadata_version(distribution_name: str) -> str:
            calls.append(distribution_name)
            return real_metadata_version(distribution_name)

        # The module under test does ``import importlib.metadata`` and
        # resolves ``importlib.metadata.version`` at call time, so patching
        # the shared module attribute intercepts its metadata lookups.
        monkeypatch.setattr(importlib.metadata, "version", counting_metadata_version)

        first = BaseFeatureGroupVersion.mloda_version()
        second = BaseFeatureGroupVersion.mloda_version()
        third = BaseFeatureGroupVersion.mloda_version()

        assert isinstance(first, str)
        assert first != ""
        assert first == second == third
        assert len(calls) == 1, (
            f"importlib.metadata lookup ran {len(calls)} times across three mloda_version calls; "
            "it must run at most once per process, with later calls served from the memo."
        )
