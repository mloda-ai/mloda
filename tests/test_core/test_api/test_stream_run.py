"""Unit tests for the stream_run instance method on mlodaAPI.

These tests verify:
1. mlodaAPI has a stream_run method that is callable
2. stream_run returns a generator (types.GeneratorType), not a list

All tests are expected to FAIL because:
- mlodaAPI does not have a stream_run method yet
"""

import types

import pytest

from mloda.user import Feature, Features, ParallelizationMode, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class TestStreamRunImport:
    """Tests that stream_run exists as a callable method on mlodaAPI."""

    def test_stream_run_method_exists_on_mloda_api(self) -> None:
        """mlodaAPI must have a stream_run attribute and it must be callable."""
        assert hasattr(mlodaAPI, "stream_run"), "mlodaAPI must have a stream_run attribute"
        assert callable(getattr(mlodaAPI, "stream_run")), "mlodaAPI.stream_run must be callable"


class TestStreamRunReturnsGenerator:
    """Tests that stream_run returns a generator, not a list."""

    def test_stream_run_returns_generator_type(self) -> None:
        """Calling stream_run on an mlodaAPI instance must return a Generator."""
        session = mlodaAPI(
            Features([Feature(name="EngineRunnerTest1", initial_requested_data=True)]),
            compute_frameworks={PyArrowTable},
        )
        result = session.stream_run(parallelization_modes={ParallelizationMode.SYNC})

        assert isinstance(result, types.GeneratorType), f"stream_run must return a generator, got {type(result)}"

    def test_stream_run_is_not_a_list(self) -> None:
        """stream_run must NOT return a list (distinguishing it from run)."""
        session = mlodaAPI(
            Features([Feature(name="EngineRunnerTest1", initial_requested_data=True)]),
            compute_frameworks={PyArrowTable},
        )
        result = session.stream_run(parallelization_modes={ParallelizationMode.SYNC})

        assert not isinstance(result, list), "stream_run must return a generator, not a list"
