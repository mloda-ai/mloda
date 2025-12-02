from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.core.engine import Engine
from mloda_core.api.prepare.setup_compute_framework import SetupComputeFramework
from mloda_core.filter.global_filter import GlobalFilter
from mloda_core.runtime.run import Runner
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.function_extender import WrapperFunctionExtender
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.link import Link


class mlodaAPI:
    def __init__(
        self,
        requested_features: Union[Features, list[Union[Feature, str]]],
        compute_frameworks: Union[Set[Type[ComputeFrameWork]], Optional[list[str]]] = None,
        links: Optional[Set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[Dict[str, Dict[str, Any]]] = None,
        plugin_collector: Optional[PlugInCollector] = None,
        copy_features: Optional[bool] = True,
    ) -> None:
        # The features object is potentially changed during the run, so we need to deepcopy it by default, so that follow up runs with the same object are not affected.
        # Set copy_features=False to disable deep copying for use cases where features contain non-copyable objects.
        _requested_features = deepcopy(requested_features) if copy_features else requested_features

        # Handle api_data: create ApiInputDataCollection if api_data provided
        api_input_data_collection: Optional[ApiInputDataCollection] = None
        if api_data is not None and len(api_data) > 0:
            api_input_data_collection = ApiInputDataCollection()
            for key_name, key_data in api_data.items():
                api_input_data_collection.setup_key_class(key_name, list(key_data.keys()))

        self.features = self._process_features(_requested_features, api_input_data_collection)
        self.compute_framework = SetupComputeFramework(compute_frameworks, self.features).compute_frameworks
        self.links = links
        self.data_access_collection = data_access_collection
        self.global_filter = global_filter
        self.api_input_data_collection = api_input_data_collection
        self.api_data = api_data
        self.plugin_collector = plugin_collector

        self.runner: None | Runner = None
        self.engine: None | Engine = None

        self.engine = self._create_engine()

    def _process_features(
        self,
        requested_features: Union[Features, list[Union[Feature, str]]],
        api_input_data_collection: Optional[ApiInputDataCollection],
    ) -> Features:
        """Processes the requested features, ensuring they are in the correct format and adding API input data."""
        features = requested_features if isinstance(requested_features, Features) else Features(requested_features)

        for feature in features:
            feature.initial_requested_data = True
            self._add_api_input_data(feature, api_input_data_collection)

        return features

    @staticmethod
    def run_all(
        features: Union[Features, list[Union[Feature, str]]],
        compute_frameworks: Union[Set[Type[ComputeFrameWork]], Optional[list[str]]] = None,
        links: Optional[Set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[Dict[str, Dict[str, Any]]] = None,
        plugin_collector: Optional[PlugInCollector] = None,
        copy_features: Optional[bool] = True,
    ) -> List[Any]:
        """
        Run feature computation in one step.

        Args:
            features: Features to compute.
            compute_frameworks: Compute frameworks to use.
            links: Links between feature groups.
            data_access_collection: Data access configuration.
            parallelization_modes: Parallelization modes.
            flight_server: Flight server for distributed processing.
            function_extender: Function extenders.
            global_filter: Global filter configuration.
            api_data: Runtime API data as {"KeyName": {"column": [values]}}.
                Auto-creates ApiInputDataCollection internally.
            plugin_collector: Plugin collector.
            copy_features: Whether to deep copy features (default True).

        Returns:
            List of computed results.

        Example:
            result = mlodaAPI.run_all(
                features,
                api_data={"UserQuery": {"row_index": [0], "query": ["hello"]}}
            )
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
        )
        return api._execute_batch_run(parallelization_modes, flight_server, function_extender)

    def _execute_batch_run(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
    ) -> List[Any]:
        """Encapsulates the batch run execution flow."""
        self._batch_run(parallelization_modes, flight_server, function_extender)
        return self.get_result()

    def _batch_run(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Sets up the engine runner and runs the engine computation."""
        # Use stored api_data if not explicitly provided
        _api_data = api_data if api_data is not None else self.api_data
        self._setup_engine_runner(parallelization_modes, flight_server)
        self._run_engine_computation(parallelization_modes, function_extender, _api_data)

    def _run_engine_computation(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Runs the engine computation within a context manager."""
        if not isinstance(self.runner, Runner):
            raise ValueError("You need to run setup_engine_runner beforehand.")

        try:
            self._enter_runner_context(parallelization_modes, function_extender, api_data)
            self.runner.compute()
        finally:
            self._exit_runner_context()

    def _enter_runner_context(
        self,
        parallelization_modes: Set[ParallelizationModes],
        function_extender: Optional[Set[WrapperFunctionExtender]],
        api_data: Optional[Dict[str, Any]],
    ) -> None:
        """Enters the runner context."""
        if self.runner is None:
            raise ValueError("You need to run setup_engine_runner beforehand.")

        self.runner.__enter__(parallelization_modes, function_extender, api_data)

    def _exit_runner_context(self) -> None:
        """Exits the runner context, shutting down the runner manager."""
        if self.runner is None:
            raise ValueError("You need to run setup_engine_runner beforehand.")

        self.runner.__exit__(None, None, None)
        self._shutdown_runner_manager()

    def _shutdown_runner_manager(self) -> None:
        """Shuts down the runner manager, handling potential exceptions."""
        try:
            if self.runner is None:
                return

            self.runner.manager.shutdown()
        except Exception:  # nosec
            pass

    def _create_engine(self) -> Engine:
        engine = Engine(
            self.features,
            self.compute_framework,
            self.links,
            self.data_access_collection,
            self.global_filter,
            self.api_input_data_collection,
            self.plugin_collector,
        )
        if not isinstance(engine, Engine):
            raise ValueError("Engine initialization failed.")
        return engine

    def _setup_engine_runner(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        flight_server: Optional[Any] = None,
    ) -> None:
        """Sets up the engine runner based on parallelization mode."""
        if self.engine is None:
            raise ValueError("You need to run setup_engine beforehand.")

        self.runner = (
            self.engine.compute(flight_server)
            if ParallelizationModes.MULTIPROCESSING in parallelization_modes
            else self.engine.compute()
        )

        if not isinstance(self.runner, Runner):
            raise ValueError("Runner initialization failed.")

    def get_result(self) -> List[Any]:
        if self.runner is None:
            raise ValueError("You need to run any run function beforehand.")
        return self.runner.get_result()

    def get_artifacts(self) -> Dict[str, Any]:
        if self.runner is None:
            raise ValueError("You need to run any run function beforehand.")
        return self.runner.get_artifacts()

    def _add_api_input_data(
        self, feature: Feature, api_input_data_collection: Optional[ApiInputDataCollection]
    ) -> None:
        """Adds API input data to the feature options if available."""
        if api_input_data_collection:
            api_input_data_column_names = api_input_data_collection.get_column_names()
            if len(api_input_data_column_names.data) == 0:
                raise ValueError("No entry names found in ApiInputDataCollection.")
            feature.options.add("ApiInputData", api_input_data_collection.get_column_names())
