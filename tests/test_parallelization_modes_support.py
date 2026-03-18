"""Failing tests for the supported_parallelization_modes feature.

These tests define requirements for:
1. SqliteFramework.supported_parallelization_modes() returns {SYNC} only
2. DuckDBFramework.supported_parallelization_modes() returns {SYNC, THREADING} only
3. ComputeFramework.supported_parallelization_modes() base class returns all three modes
4. SetupComputeFramework filters out frameworks incompatible with requested parallelization_modes
5. FeatureGroupStep.get_parallelization_mode() delegates to compute_framework.supported_parallelization_modes()
"""

from unittest.mock import MagicMock

import pytest

from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.api.prepare.setup_compute_framework import SetupComputeFramework
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework


class TestComputeFrameworkSupportedParallelizationModes:
    """Tests for ComputeFramework.supported_parallelization_modes() base class method."""

    def test_base_class_returns_all_three_modes(self) -> None:
        """ComputeFramework base class must return all three parallelization modes."""
        result = ComputeFramework.supported_parallelization_modes()
        assert result == {
            ParallelizationMode.SYNC,
            ParallelizationMode.THREADING,
            ParallelizationMode.MULTIPROCESSING,
        }

    def test_base_class_returns_a_set(self) -> None:
        """ComputeFramework.supported_parallelization_modes() must return a set."""
        result = ComputeFramework.supported_parallelization_modes()
        assert isinstance(result, set)


class TestSqliteFrameworkSupportedParallelizationModes:
    """Tests for SqliteFramework.supported_parallelization_modes()."""

    def test_sqlite_returns_sync_only(self) -> None:
        """SqliteFramework must only support SYNC mode (no threading, no multiprocessing)."""
        result = SqliteFramework.supported_parallelization_modes()
        assert result == {ParallelizationMode.SYNC}

    def test_sqlite_does_not_support_threading(self) -> None:
        """SqliteFramework must not include THREADING in supported modes."""
        result = SqliteFramework.supported_parallelization_modes()
        assert ParallelizationMode.THREADING not in result

    def test_sqlite_does_not_support_multiprocessing(self) -> None:
        """SqliteFramework must not include MULTIPROCESSING in supported modes."""
        result = SqliteFramework.supported_parallelization_modes()
        assert ParallelizationMode.MULTIPROCESSING not in result

    def test_sqlite_returns_a_set(self) -> None:
        """SqliteFramework.supported_parallelization_modes() must return a set."""
        result = SqliteFramework.supported_parallelization_modes()
        assert isinstance(result, set)


class TestDuckDBFrameworkSupportedParallelizationModes:
    """Tests for DuckDBFramework.supported_parallelization_modes()."""

    def test_duckdb_returns_sync_and_threading(self) -> None:
        """DuckDBFramework must support SYNC only (connections are not thread-safe)."""
        result = DuckDBFramework.supported_parallelization_modes()
        assert result == {ParallelizationMode.SYNC}

    def test_duckdb_does_not_support_multiprocessing(self) -> None:
        """DuckDBFramework must not include MULTIPROCESSING in supported modes."""
        result = DuckDBFramework.supported_parallelization_modes()
        assert ParallelizationMode.MULTIPROCESSING not in result

    def test_duckdb_supports_sync(self) -> None:
        """DuckDBFramework must include SYNC in supported modes."""
        result = DuckDBFramework.supported_parallelization_modes()
        assert ParallelizationMode.SYNC in result

    def test_duckdb_does_not_support_threading(self) -> None:
        """DuckDBFramework must not include THREADING (connections are not thread-safe)."""
        result = DuckDBFramework.supported_parallelization_modes()
        assert ParallelizationMode.THREADING not in result

    def test_duckdb_returns_a_set(self) -> None:
        """DuckDBFramework.supported_parallelization_modes() must return a set."""
        result = DuckDBFramework.supported_parallelization_modes()
        assert isinstance(result, set)


class TestSetupComputeFrameworkParallelizationModeFiltering:
    """Tests for SetupComputeFramework filtering by parallelization_modes."""

    def test_multiprocessing_mode_excludes_sqlite_framework(self) -> None:
        """Passing parallelization_modes={MULTIPROCESSING} with user_compute_frameworks={SqliteFramework}
        must raise ValueError because SqliteFramework only supports SYNC.

        After filtering, no compatible framework remains, which must trigger a ValueError.
        """
        features = Features([Feature("test_feature")])
        with pytest.raises(ValueError):
            SetupComputeFramework(
                user_compute_frameworks={SqliteFramework},
                features=features,
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            )

    def test_sync_mode_includes_sqlite_framework(self) -> None:
        """Passing parallelization_modes={SYNC} with user_compute_frameworks={SqliteFramework}
        must succeed because SqliteFramework supports SYNC.
        """
        features = Features([Feature("test_feature")])
        setup = SetupComputeFramework(
            user_compute_frameworks={SqliteFramework},
            features=features,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        assert SqliteFramework in setup.compute_frameworks

    def test_multiprocessing_mode_excludes_duckdb_framework(self) -> None:
        """Passing parallelization_modes={MULTIPROCESSING} with user_compute_frameworks={DuckDBFramework}
        must raise ValueError because DuckDBFramework only supports SYNC and THREADING.
        """
        features = Features([Feature("test_feature")])
        with pytest.raises(ValueError):
            SetupComputeFramework(
                user_compute_frameworks={DuckDBFramework},
                features=features,
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            )

    def test_none_parallelization_modes_does_not_filter(self) -> None:
        """When parallelization_modes=None (default), no filtering by mode should occur.
        SqliteFramework must still be included.
        """
        features = Features([Feature("test_feature")])
        setup = SetupComputeFramework(
            user_compute_frameworks={SqliteFramework},
            features=features,
        )
        assert SqliteFramework in setup.compute_frameworks


class TestFeatureGroupStepGetParallelizationMode:
    """Tests for FeatureGroupStep.get_parallelization_mode() delegation."""

    def _make_minimal_feature_set(self) -> FeatureSet:
        feature = Feature("test_feature")
        feature_set = FeatureSet()
        feature_set.add(feature)
        return feature_set

    def test_get_parallelization_mode_delegates_to_sqlite_framework(self) -> None:
        """FeatureGroupStep.get_parallelization_mode() with SqliteFramework must return {SYNC}."""
        mock_feature_group = MagicMock()
        feature_set = self._make_minimal_feature_set()

        step = FeatureGroupStep(
            feature_group=mock_feature_group,
            features=feature_set,
            required_uuids=set(),
            compute_framework=SqliteFramework,
        )

        result = step.get_parallelization_mode()
        assert result == {ParallelizationMode.SYNC}

    def test_get_parallelization_mode_delegates_to_duckdb_framework(self) -> None:
        """FeatureGroupStep.get_parallelization_mode() with DuckDBFramework must return {SYNC}."""
        mock_feature_group = MagicMock()
        feature_set = self._make_minimal_feature_set()

        step = FeatureGroupStep(
            feature_group=mock_feature_group,
            features=feature_set,
            required_uuids=set(),
            compute_framework=DuckDBFramework,
        )

        result = step.get_parallelization_mode()
        assert result == {ParallelizationMode.SYNC}
