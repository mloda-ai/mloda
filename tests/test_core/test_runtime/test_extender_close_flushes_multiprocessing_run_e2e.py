"""E2E: a buffering-sink Extender's close() must run inside a real spawned MULTIPROCESSING
worker before that worker exits, so events "flushed" only at close() time are not lost when
the worker is torn down. GitHub issue #1439.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_SENTINEL_CONTENT = "flushed-by-close"


class _MultiprocessingOnlyCloseFlushFramework(PythonDictFramework):
    """Restricted to MULTIPROCESSING only, so its FeatureGroupStep always runs in a spawned worker."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.MULTIPROCESSING}


class _CloseFlushFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"close_flush_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_MultiprocessingOnlyCloseFlushFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"close_flush_e2e_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_CloseFlushFeatureGroup})


class _BufferingSinkExtender(Extender):
    """Buffers nothing until close(), which writes the sentinel to output_path.

    output_path is set in __init__, before the extender is pickled into the worker; the
    write itself must happen from the worker's own close() call.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def close(self) -> None:
        self._output_path.write_text(_SENTINEL_CONTENT)


@pytest.mark.timeout(30)
class TestExtenderCloseFlushesBufferedEventsInAMultiprocessingWorker:
    def test_sentinel_file_is_written_by_close_inside_the_spawned_worker(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        output_path = tmp_path / "close_flush_sentinel.txt"
        probe = _BufferingSinkExtender(output_path)

        session = mloda.prepare(
            [Feature(name="close_flush_e2e_col")],
            compute_frameworks=["_MultiprocessingOnlyCloseFlushFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
        )

        session.run(
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            function_extender={probe},
            flight_server=flight_server,
        )

        assert output_path.exists(), (
            "close() was never called inside the worker before it exited: buffered events would be lost"
        )
        assert output_path.read_text() == _SENTINEL_CONTENT
