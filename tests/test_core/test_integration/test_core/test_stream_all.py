# mypy: disable-error-code="type-arg"
"""Integration tests for the stream_all standalone function."""

from typing import Any, Dict, FrozenSet, List, Set, Tuple

import pyarrow as pa
import pytest

from mloda.user import Feature, Features, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


COMPUTE_FRAMEWORKS: Set[Any] = {PyArrowTable}
PARALLELIZATION_MODES: Set[ParallelizationMode] = {ParallelizationMode.SYNC}


def _get_features(feature_list: List[str]) -> Features:
    """Helper to create a Features collection from a list of feature name strings."""
    return Features([Feature(name=f_name, initial_requested_data=True) for f_name in feature_list])


def _run_all_results(feature_list: List[str]) -> List[Any]:
    """Run features using the batch run_all API and return the result list."""
    from mloda.user import mloda

    features = _get_features(feature_list)
    return mloda.run_all(
        features,
        compute_frameworks=COMPUTE_FRAMEWORKS,
        parallelization_modes=PARALLELIZATION_MODES,
    )


def _stream_all_results(feature_list: List[str]) -> List[Any]:
    """Run features using stream_all and collect all yielded results into a list."""
    from mloda.core.api.streaming_request import stream_all

    features = _get_features(feature_list)
    return list(
        stream_all(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
            parallelization_modes=PARALLELIZATION_MODES,
        )
    )


class TestStreamAllSingleFeature:
    """stream_all with a single independent feature group."""

    def test_stream_all_single_feature_yields_result(self) -> None:
        """stream_all with EngineRunnerTest1 must yield exactly one result."""
        from mloda.core.api.streaming_request import stream_all

        features = _get_features(["EngineRunnerTest1"])
        results = list(
            stream_all(
                features,
                compute_frameworks=COMPUTE_FRAMEWORKS,
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], pa.Table)
        assert results[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}


class TestStreamAllDependentFeature:
    """stream_all with a feature group that depends on another."""

    def test_stream_all_dependent_feature_yields_correct_result(self) -> None:
        """stream_all with EngineRunnerTest2 must yield the computed dependent result."""
        from mloda.core.api.streaming_request import stream_all

        features = _get_features(["EngineRunnerTest2"])
        results = list(
            stream_all(
                features,
                compute_frameworks=COMPUTE_FRAMEWORKS,
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results) == 1
        assert results[0].to_pydict() == {"EngineRunnerTest2": [2, 4, 6]}


class TestStreamAllMultipleFeatures:
    """stream_all with multiple feature groups requested together."""

    def test_stream_all_multiple_features_yields_all_results(self) -> None:
        """stream_all with two features must yield results for both."""
        from mloda.core.api.streaming_request import stream_all

        features = _get_features(["EngineRunnerTest2", "EngineRunnerTest1"])
        results = list(
            stream_all(
                features,
                compute_frameworks=COMPUTE_FRAMEWORKS,
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results) == 2

        result_dicts = [r.to_pydict() for r in results]
        result_keys = set()
        for d in result_dicts:
            result_keys.update(d.keys())

        assert "EngineRunnerTest1" in result_keys
        assert "EngineRunnerTest2" in result_keys


class TestStreamAllMatchesRunAll:
    """stream_all collected into a list must match run_all output."""

    def test_stream_all_matches_run_all_single_feature(self) -> None:
        """list(stream_all(...)) must produce the same data as run_all(...) for a single feature."""
        from mloda.core.api.streaming_request import stream_all

        feature_list = ["EngineRunnerTest1"]

        run_all_result = _run_all_results(feature_list)
        stream_all_result = _stream_all_results(feature_list)

        assert len(stream_all_result) == len(run_all_result)

        for stream_table, batch_table in zip(stream_all_result, run_all_result):
            assert stream_table.to_pydict() == batch_table.to_pydict()

    def test_stream_all_matches_run_all_multiple_features(self) -> None:
        """list(stream_all(...)) must produce the same data as run_all(...) for multiple features."""
        from mloda.core.api.streaming_request import stream_all

        feature_list = ["EngineRunnerTest2", "EngineRunnerTest1"]

        run_all_result = _run_all_results(feature_list)
        stream_all_result = _stream_all_results(feature_list)

        assert len(stream_all_result) == len(run_all_result)

        # Convert both to sets of frozensets for order-independent comparison,
        # since computation order is not guaranteed.
        def to_comparable(results: List[Any]) -> Set[frozenset]:
            comparable = set()
            for table in results:
                d = table.to_pydict()
                items = frozenset((k, tuple(v)) for k, v in d.items())
                comparable.add(items)
            return comparable

        assert to_comparable(stream_all_result) == to_comparable(run_all_result)
