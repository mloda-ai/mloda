# mypy: disable-error-code="type-arg"
"""Integration tests for mlodaAPI.stream_all."""

from typing import Any, List, Set

import pyarrow as pa

from mloda.user import Feature, Features, ParallelizationMode, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


COMPUTE_FRAMEWORKS: Set[Any] = {PyArrowTable}
PARALLELIZATION_MODES: Set[ParallelizationMode] = {ParallelizationMode.SYNC}


def _get_features(feature_list: List[str]) -> Features:
    return Features([Feature(name=f_name, initial_requested_data=True) for f_name in feature_list])


def _run_all_results(feature_list: List[str]) -> List[Any]:
    return mlodaAPI.run_all(
        _get_features(feature_list),
        compute_frameworks=COMPUTE_FRAMEWORKS,
        parallelization_modes=PARALLELIZATION_MODES,
    )


def _stream_all_results(feature_list: List[str]) -> List[Any]:
    return list(
        mlodaAPI.stream_all(
            _get_features(feature_list),
            compute_frameworks=COMPUTE_FRAMEWORKS,
            parallelization_modes=PARALLELIZATION_MODES,
        )
    )


class TestStreamAllSingleFeature:
    def test_single_feature_yields_result(self) -> None:
        results = _stream_all_results(["EngineRunnerTest1"])

        assert len(results) == 1
        assert isinstance(results[0], pa.Table)
        assert results[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}


class TestStreamAllDependentFeature:
    def test_dependent_feature_yields_correct_result(self) -> None:
        results = _stream_all_results(["EngineRunnerTest2"])

        assert len(results) == 1
        assert results[0].to_pydict() == {"EngineRunnerTest2": [2, 4, 6]}


class TestStreamAllMultipleFeatures:
    def test_multiple_features_yields_all_results(self) -> None:
        results = _stream_all_results(["EngineRunnerTest2", "EngineRunnerTest1"])

        assert len(results) == 2
        result_keys = {k for r in results for k in r.to_pydict()}
        assert result_keys == {"EngineRunnerTest1", "EngineRunnerTest2"}


class TestStreamAllMatchesRunAll:
    def test_matches_run_all_single_feature(self) -> None:
        feature_list = ["EngineRunnerTest1"]
        run_result = _run_all_results(feature_list)
        stream_result = _stream_all_results(feature_list)

        assert len(stream_result) == len(run_result)
        for s, r in zip(stream_result, run_result):
            assert s.to_pydict() == r.to_pydict()

    def test_matches_run_all_multiple_features(self) -> None:
        feature_list = ["EngineRunnerTest2", "EngineRunnerTest1"]
        run_result = _run_all_results(feature_list)
        stream_result = _stream_all_results(feature_list)

        assert len(stream_result) == len(run_result)

        def to_comparable(results: List[Any]) -> Set[frozenset]:
            return {frozenset((k, tuple(v)) for k, v in t.to_pydict().items()) for t in results}

        assert to_comparable(stream_result) == to_comparable(run_result)


class TestStreamAllViaAlias:
    def test_mloda_alias_stream_all(self) -> None:
        from mloda.user import mloda

        results = list(
            mloda.stream_all(
                _get_features(["EngineRunnerTest1"]),
                compute_frameworks=COMPUTE_FRAMEWORKS,
                parallelization_modes=PARALLELIZATION_MODES,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], pa.Table)
        assert results[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}
