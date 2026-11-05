# mypy: disable-error-code="type-arg"
"""stream_run exposes artifacts and failed_rows on the stream after an early consumer exit.

Breaking out or stopping after one ``next()`` must leave ``stream.artifacts`` and
``stream.failed_rows`` readable, as after a full iteration.
"""

from typing import Any

import pyarrow as pa
import pytest

# stream_run plans eagerly, so the EngineRunnerTest* feature groups must be
# registered before prepare() is called.
import tests.test_core.test_integration.test_core.test_runner_one_compute_framework  # noqa: F401
from mloda.user import Feature, Features, ParallelizationMode, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

COMPUTE_FRAMEWORKS: list[Any] = [PyArrowTable]
PARALLELIZATION_MODES: set[ParallelizationMode] = {ParallelizationMode.SYNC}


def _session_with_features(feature_list: list[str]) -> mlodaAPI:
    """Prepare a session with the given features."""
    return mlodaAPI.prepare(
        Features([Feature(name=f, initial_requested_data=True) for f in feature_list]),
        compute_frameworks=COMPUTE_FRAMEWORKS,
    )


class TestStreamRunEarlyExit:
    """The stream exposes artifacts and failed_rows after any consumer exit style."""

    @pytest.mark.parametrize("exit_style", ["single_next", "break", "next_then_close", "full_iteration"])
    def test_state_readable_and_session_keeps_no_runner(self, exit_style: str) -> None:
        session = _session_with_features(["EngineRunnerTest1", "EngineRunnerTest2"])
        stream = session.stream_run(parallelization_modes=PARALLELIZATION_MODES)

        if exit_style == "single_next":
            assert isinstance(next(stream), pa.Table)
        elif exit_style == "break":
            for _result in stream:
                break
        elif exit_style == "next_then_close":
            next(stream)
            stream.close()
        else:
            results = list(stream)
            assert len(results) == 2
            assert all(isinstance(r, pa.Table) for r in results)

        assert isinstance(stream.artifacts, dict)
        assert stream.failed_rows == ()
        assert not hasattr(session, "runner")

    def test_stream_all_break_then_close_exposes_state(self) -> None:
        """Closing a stream_all stream after an early break drives teardown without raising."""
        stream = mlodaAPI.stream_all(
            Features(
                [
                    Feature(name="EngineRunnerTest1", initial_requested_data=True),
                    Feature(name="EngineRunnerTest2", initial_requested_data=True),
                ]
            ),
            compute_frameworks=COMPUTE_FRAMEWORKS,
            parallelization_modes=PARALLELIZATION_MODES,
        )

        for _result in stream:
            break

        stream.close()
        assert isinstance(stream.artifacts, dict)
        assert stream.failed_rows == ()
