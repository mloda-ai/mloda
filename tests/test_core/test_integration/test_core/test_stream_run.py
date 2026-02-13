# mypy: disable-error-code="type-arg"
"""Integration tests for the stream_run instance method on mlodaAPI."""

from typing import Any, List, Set

import pyarrow as pa
import pytest

from mloda.user import Feature, Features, ParallelizationMode, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


COMPUTE_FRAMEWORKS: Set[Any] = {PyArrowTable}
PARALLELIZATION_MODES: Set[ParallelizationMode] = {ParallelizationMode.SYNC}


def _get_features(feature_list: List[str]) -> Features:
    """Helper to create a Features collection from a list of feature name strings."""
    return Features([Feature(name=f_name, initial_requested_data=True) for f_name in feature_list])


def _run_results(feature_list: List[str]) -> List[Any]:
    """Run features using the two-phase prepare + run API and return the result list."""
    features = _get_features(feature_list)
    session = mlodaAPI.prepare(
        features,
        compute_frameworks=COMPUTE_FRAMEWORKS,
    )
    return session.run(
        parallelization_modes=PARALLELIZATION_MODES,
    )


def _stream_run_results(feature_list: List[str]) -> List[Any]:
    """Run features using prepare + stream_run and collect all yielded results into a list."""
    features = _get_features(feature_list)
    session = mlodaAPI.prepare(
        features,
        compute_frameworks=COMPUTE_FRAMEWORKS,
    )
    return list(
        session.stream_run(
            parallelization_modes=PARALLELIZATION_MODES,
        )
    )


class TestStreamRunSingleFeature:
    """stream_run with a single independent feature group."""

    def test_stream_run_single_feature_yields_result(self) -> None:
        """stream_run with EngineRunnerTest1 must yield exactly one result."""
        features = _get_features(["EngineRunnerTest1"])
        session = mlodaAPI.prepare(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
        )
        results = list(
            session.stream_run(
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], pa.Table)
        assert results[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}


class TestStreamRunDependentFeature:
    """stream_run with a feature group that depends on another."""

    def test_stream_run_dependent_feature_yields_correct_result(self) -> None:
        """stream_run with EngineRunnerTest2 must yield the computed dependent result."""
        features = _get_features(["EngineRunnerTest2"])
        session = mlodaAPI.prepare(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
        )
        results = list(
            session.stream_run(
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results) == 1
        assert results[0].to_pydict() == {"EngineRunnerTest2": [2, 4, 6]}


class TestStreamRunMultipleFeatures:
    """stream_run with multiple feature groups requested together."""

    def test_stream_run_multiple_features_yields_all_results(self) -> None:
        """stream_run with two features must yield results for both."""
        features = _get_features(["EngineRunnerTest2", "EngineRunnerTest1"])
        session = mlodaAPI.prepare(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
        )
        results = list(
            session.stream_run(
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


class TestStreamRunMatchesRun:
    """stream_run collected into a list must match run output."""

    def test_stream_run_matches_run_single_feature(self) -> None:
        """list(stream_run()) must produce the same data as run() for a single feature."""
        feature_list = ["EngineRunnerTest1"]

        run_result = _run_results(feature_list)
        stream_run_result = _stream_run_results(feature_list)

        assert len(stream_run_result) == len(run_result)

        for stream_table, batch_table in zip(stream_run_result, run_result):
            assert stream_table.to_pydict() == batch_table.to_pydict()

    def test_stream_run_matches_run_multiple_features(self) -> None:
        """list(stream_run()) must produce the same data as run() for multiple features."""
        feature_list = ["EngineRunnerTest2", "EngineRunnerTest1"]

        run_result = _run_results(feature_list)
        stream_run_result = _stream_run_results(feature_list)

        assert len(stream_run_result) == len(run_result)

        def to_comparable(results: List[Any]) -> Set[frozenset]:
            comparable = set()
            for table in results:
                d = table.to_pydict()
                items = frozenset((k, tuple(v)) for k, v in d.items())
                comparable.add(items)
            return comparable

        assert to_comparable(stream_run_result) == to_comparable(run_result)


class TestStreamRunPlanReuse:
    """stream_run must work when called multiple times on the same session."""

    def test_stream_run_reuse_session(self) -> None:
        """Calling prepare() once and stream_run() twice must produce correct results both times."""
        features = _get_features(["EngineRunnerTest1"])
        session = mlodaAPI.prepare(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
        )

        results_first = list(
            session.stream_run(
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )
        results_second = list(
            session.stream_run(
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results_first) == 1
        assert len(results_second) == 1
        assert results_first[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}
        assert results_second[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}


class TestStreamRunRunnerAssignment:
    """stream_run must set the runner attribute on the session after execution."""

    def test_runner_set_after_stream_run(self) -> None:
        """After consuming the full generator from stream_run(), session.runner must not be None."""
        features = _get_features(["EngineRunnerTest1"])
        session = mlodaAPI.prepare(
            features,
            compute_frameworks=COMPUTE_FRAMEWORKS,
        )

        results = list(
            session.stream_run(
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert session.runner is not None, "session.runner must be set after stream_run() completes"
