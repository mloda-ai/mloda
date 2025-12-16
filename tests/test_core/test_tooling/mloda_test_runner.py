"""
Unified test runner infrastructure for API integration tests.

This module provides consistent patterns for:
- Running with different parallelization modes (SYNC, THREADING, MULTIPROCESSING)
- Flight server management and cleanup
- Result and artifact extraction

Usage:
    from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_ALL

    @PARALLELIZATION_MODES_ALL
    class TestMyFeature:
        def test_something(self, modes, flight_server):
            result = MlodaTestRunner.run_api(features, parallelization_modes=modes, flight_server=flight_server)
            assert result.artifacts == expected
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type

import pytest

from mloda.user import Features
from mloda.user import Link
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda import ComputeFramework
from mloda.steward import Extender
import mloda
from mloda.user import GlobalFilter
from mloda.core.core.engine import Engine
from mloda.core.runtime.flight.flight_server import FlightServer
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


@dataclass
class RunResult:
    """Container for test run results."""

    results: List[Any] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    runner: Optional[ExecutionOrchestrator] = None


class MlodaTestRunner:
    """
    Unified test runner for API integration tests.

    Provides consistent patterns for:
    - Running with different parallelization modes
    - Flight server management and cleanup
    - Result and artifact extraction

    Methods:
        run_api: Full access to results and artifacts (recommended for most tests)
        run_api_simple: Quick runs when only results are needed
        run_engine: Full control over execution via Engine + ExecutionOrchestrator
    """

    @staticmethod
    def run_api(
        features: Features,
        compute_frameworks: Optional[Set[Type[ComputeFramework]]] = None,
        parallelization_modes: Optional[Set[ParallelizationMode]] = None,
        flight_server: Any = None,
        function_extender: Optional[Set[Extender]] = None,
        links: Optional[Set[Link]] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[Dict[str, Any]] = None,
        cleanup_flight_server: bool = True,
        plugin_collector: Optional[PluginCollector] = None,
        strict_type_enforcement: bool = False,
    ) -> RunResult:
        """
        Run API with the given configuration.

        This is the recommended method for most integration tests.
        Uses API instance + _batch_run() for full access to results and artifacts.

        Args:
            features: The feature set to compute
            compute_frameworks: Set of compute frameworks to use (default: {PyArrowTable})
            parallelization_modes: Set of parallelization modes (default: {SYNC})
            flight_server: Flight server instance for MULTIPROCESSING mode
            function_extender: Optional function extenders
            links: Optional links for joins
            global_filter: Optional global filter
            api_data: Optional API data dictionary
            cleanup_flight_server: Whether to assert flight server cleanup (default: True)
            plugin_collector: Optional plugin collector for feature group filtering
            strict_type_enforcement: If True, enforce strict type matching for typed features

        Returns:
            RunResult containing results, artifacts, and optionally the runner
        """
        if compute_frameworks is None:
            compute_frameworks = {PyArrowTable}
        if parallelization_modes is None:
            parallelization_modes = {ParallelizationMode.SYNC}

        api = mloda.API(
            features,
            compute_frameworks,
            links=links,
            global_filter=global_filter,
            api_data=api_data,
            plugin_collector=plugin_collector,
            strict_type_enforcement=strict_type_enforcement,
        )
        api._batch_run(parallelization_modes, flight_server, function_extender)

        results = api.get_result()
        artifacts = api.get_artifacts()

        if cleanup_flight_server:
            MlodaTestRunner.assert_flight_server_clean(parallelization_modes, flight_server)

        return RunResult(results=results, artifacts=artifacts)

    @staticmethod
    def run_api_simple(
        features: Features,
        compute_frameworks: Optional[Set[Type[ComputeFramework]]] = None,
        parallelization_modes: Optional[Set[ParallelizationMode]] = None,
        flight_server: Any = None,
        function_extender: Optional[Set[Extender]] = None,
    ) -> List[Any]:
        """
        Simplified runner using mloda.run_all().

        Use when you only need results (no artifacts).

        Args:
            features: The feature set to compute
            compute_frameworks: Set of compute frameworks to use (default: {PyArrowTable})
            parallelization_modes: Set of parallelization modes (default: {SYNC})
            flight_server: Flight server instance for MULTIPROCESSING mode
            function_extender: Optional function extenders

        Returns:
            List of result data
        """
        if compute_frameworks is None:
            compute_frameworks = {PyArrowTable}
        if parallelization_modes is None:
            parallelization_modes = {ParallelizationMode.SYNC}

        return mloda.run_all(
            features,
            compute_frameworks,
            None,  # links
            None,  # data_access_collection
            parallelization_modes,
            flight_server,
            function_extender,
        )

    @staticmethod
    def run_engine(
        features: Features,
        compute_frameworks: Optional[Set[Type[ComputeFramework]]] = None,
        parallelization_modes: Optional[Set[ParallelizationMode]] = None,
        flight_server: Any = None,
        function_extender: Optional[Set[Extender]] = None,
        links: Optional[Set[Link]] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionOrchestrator:
        """
        Run using Engine + ExecutionOrchestrator for full control over execution.

        Use when you need access to the ExecutionOrchestrator internals (execution plan, CFW collection, etc.).

        Args:
            features: The feature set to compute
            compute_frameworks: Set of compute frameworks to use (default: {PyArrowTable})
            parallelization_modes: Set of parallelization modes (default: {SYNC})
            flight_server: Flight server instance for MULTIPROCESSING mode
            function_extender: Optional function extenders
            links: Optional links for joins
            global_filter: Optional global filter
            api_data: Optional API data dictionary

        Returns:
            ExecutionOrchestrator instance after execution
        """
        if compute_frameworks is None:
            compute_frameworks = {PyArrowTable}
        if parallelization_modes is None:
            parallelization_modes = {ParallelizationMode.SYNC}

        engine = Engine(features, compute_frameworks, links, global_filter=global_filter)

        use_flight = ParallelizationMode.MULTIPROCESSING in parallelization_modes
        runner = engine.compute(flight_server if use_flight else None)

        try:
            runner.__enter__(parallelization_modes, function_extender, api_data)
            runner.compute()
            runner.__exit__(None, None, None)
        finally:
            try:
                runner.manager.shutdown()
            except Exception:  # nosec
                pass

        return runner

    @staticmethod
    def assert_flight_server_clean(
        parallelization_modes: Set[ParallelizationMode],
        flight_server: Any,
        timeout: float = 1.0,
    ) -> None:
        """
        Assert that all datasets have been cleaned up from the flight server.

        Uses polling with timeout to handle async cleanup race conditions.

        Args:
            parallelization_modes: The parallelization modes used in the test
            flight_server: The flight server instance
            timeout: Maximum time to wait for cleanup (default: 1.0 seconds)

        Raises:
            AssertionError: If datasets remain on the flight server after timeout
        """
        if ParallelizationMode.MULTIPROCESSING not in parallelization_modes:
            return
        if flight_server is None:
            return

        start = time.time()
        while time.time() - start < timeout:
            flight_infos = FlightServer.list_flight_infos(flight_server.location)
            if len(flight_infos) == 0:
                return
            time.sleep(0.02)

        # Final check with assertion
        flight_infos = FlightServer.list_flight_infos(flight_server.location)
        assert len(flight_infos) == 0, f"Flight server still has {len(flight_infos)} datasets"


# Shared pytest parametrization for parallelization modes
PARALLELIZATION_MODES_SYNC_THREADING = pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
    ],
)

PARALLELIZATION_MODES_ALL = pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
        ({ParallelizationMode.MULTIPROCESSING}),
    ],
)
