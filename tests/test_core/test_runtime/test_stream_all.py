"""Unit tests for the stream_all standalone function.

These tests verify:
1. stream_all can be imported from mloda.core.api.streaming_request
2. stream_all returns a generator (types.GeneratorType)
3. stream_all can be imported from mloda.user (user-facing module)

All tests are expected to FAIL because:
- mloda.core.api.streaming_request module does not exist yet
- stream_all is not exported from mloda.user
"""

import types

import pytest


class TestStreamAllImport:
    """Tests that stream_all is importable from the correct modules."""

    def test_stream_all_importable_from_streaming_request(self) -> None:
        """stream_all must be importable from mloda.core.api.streaming_request."""
        from mloda.core.api.streaming_request import stream_all

        assert callable(stream_all)

    def test_stream_all_importable_from_user_module(self) -> None:
        """stream_all must be importable from mloda.user as a user-facing API."""
        from mloda.user import stream_all

        assert callable(stream_all)


class TestStreamAllReturnsGenerator:
    """Tests that stream_all returns a generator, not a list."""

    def test_stream_all_returns_generator_type(self) -> None:
        """Calling stream_all with minimal valid arguments must return a Generator."""
        from mloda.core.api.streaming_request import stream_all
        from mloda.user import Feature, Features, ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        features = Features([Feature(name="EngineRunnerTest1", initial_requested_data=True)])
        result = stream_all(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert isinstance(result, types.GeneratorType), f"stream_all must return a generator, got {type(result)}"

    def test_stream_all_is_not_a_list(self) -> None:
        """stream_all must NOT return a list (distinguishing it from run_all)."""
        from mloda.core.api.streaming_request import stream_all
        from mloda.user import Feature, Features, ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        features = Features([Feature(name="EngineRunnerTest1", initial_requested_data=True)])
        result = stream_all(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert not isinstance(result, list), "stream_all must return a generator, not a list"
