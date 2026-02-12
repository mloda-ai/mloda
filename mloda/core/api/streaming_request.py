from typing import Any, Dict, Generator, Optional, Set, Type, Union

from mloda.core.api.request import mlodaAPI
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.filter.global_filter import GlobalFilter


def stream_all(
    features: Union[Features, list[Union[Feature, str]]],
    compute_frameworks: Union[Set[Type[ComputeFramework]], Optional[list[str]]] = None,
    links: Optional[Set[Link]] = None,
    data_access_collection: Optional[DataAccessCollection] = None,
    parallelization_modes: Set[ParallelizationMode] = {ParallelizationMode.SYNC},
    flight_server: Optional[Any] = None,
    function_extender: Optional[Set[Extender]] = None,
    global_filter: Optional[GlobalFilter] = None,
    api_data: Optional[Dict[str, Dict[str, Any]]] = None,
    plugin_collector: Optional[PluginCollector] = None,
    copy_features: Optional[bool] = True,
    strict_type_enforcement: bool = False,
    column_ordering: Optional[str] = None,
) -> Generator[Any, None, None]:
    """Stream results at feature-group granularity.

    Works like ``run_all`` but yields each result as soon as its feature group
    finishes, instead of waiting for every group to complete.  Each yielded
    value is a fully materialized compute-framework object (e.g. a
    ``pa.Table``).  ``list(stream_all(...))`` produces the same results as
    ``run_all(...)``.

    This function does **not** provide row-by-row or partial-result streaming
    within a single feature group.

    Args:
        features: Features to compute (same as ``run_all``).
        compute_frameworks: Optional set of compute frameworks to use.
        links: Optional join links between datasets.
        data_access_collection: Optional data sources.
        parallelization_modes: Execution modes (default: ``{SYNC}``).
        flight_server: Optional flight server for multiprocessing.
        function_extender: Optional function extenders.
        global_filter: Optional global data filter.
        api_data: Optional per-feature-group API data.
        plugin_collector: Optional plugin collector.
        copy_features: Whether to deep-copy the feature definitions.
        strict_type_enforcement: Enable strict data-type checking.
        column_ordering: Column ordering mode for results.

    Yields:
        One complete result per feature group, in the order groups finish.
    """
    api = mlodaAPI(
        features,
        compute_frameworks,
        links,
        data_access_collection,
        global_filter,
        api_data=api_data,
        plugin_collector=plugin_collector,
        copy_features=copy_features,
        strict_type_enforcement=strict_type_enforcement,
        column_ordering=column_ordering,
    )
    api._setup_engine_runner(parallelization_modes, flight_server)
    if api.runner is None:
        raise ValueError("ExecutionOrchestrator initialization failed.")
    try:
        api._enter_runner_context(parallelization_modes, function_extender, api.api_data)
        for _step_uuid, result in api.runner.compute_stream():
            yield result
    finally:
        api._exit_runner_context()
