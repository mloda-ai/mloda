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
        api_input_data_collection: Optional[ApiInputDataCollection] = None,
        plugin_collector: Optional[PlugInCollector] = None,
    ) -> None:
        if not isinstance(requested_features, Features):
            features = Features(requested_features)
        else:
            features = requested_features

        for feature in features:
            feature.initial_requested_data = True
            self.add_api_input_data(feature, api_input_data_collection)

        self.compute_framework = SetupComputeFramework(compute_frameworks, features).compute_frameworks
        self.links = links
        self.features = features
        self.data_access_collection = data_access_collection

        self.runner: None | Runner = None
        self.engine: None | Engine = None

        self.global_filter = global_filter
        self.api_input_data_collection = api_input_data_collection
        self.plugin_collector = plugin_collector

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
        api_input_data_collection: Optional[ApiInputDataCollection] = None,
        api_data: Optional[Dict[str, Any]] = None,
        plugin_collector: Optional[PlugInCollector] = None,
    ) -> List[Any]:
        """
        This step runs setup engine, batch run and get result in one go.
        """
        api = mlodaAPI(
            features,
            compute_frameworks,
            links,
            data_access_collection,
            global_filter,
            api_input_data_collection,
            plugin_collector,
        )
        api.batch_run(parallelization_modes, flight_server, function_extender, api_data)
        return api.get_result()

    def batch_run(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.setup_engine()
        self.setup_engine_runner(parallelization_modes, flight_server)
        self.run_engine_computation(parallelization_modes, function_extender, api_data)

    def run_engine_computation(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(self.runner, Runner):
            raise ValueError("You need to run setup_engine_runner beforehand.")

        try:
            self.runner.__enter__(parallelization_modes, function_extender, api_data)
            self.runner.compute()
            self.runner.__exit__(None, None, None)
        finally:
            try:
                self.runner.manager.shutdown()
            except Exception:  # nosec
                pass

    def setup_engine(self) -> None:
        self.engine = Engine(
            self.features,
            self.compute_framework,
            self.links,
            self.data_access_collection,
            self.global_filter,
            self.api_input_data_collection,
            self.plugin_collector,
        )
        if not isinstance(self.engine, Engine):
            raise ValueError("Engine initialization failed.")

    def setup_engine_runner(
        self,
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        flight_server: Optional[Any] = None,
    ) -> None:
        if self.engine is None:
            raise ValueError("You need to run setup_engine beforehand.")

        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            self.runner = self.engine.compute(flight_server)
        else:
            self.runner = self.engine.compute()

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

    def add_api_input_data(self, feature: Feature, api_input_data_collection: Optional[ApiInputDataCollection]) -> None:
        if api_input_data_collection:
            api_input_data_column_names = api_input_data_collection.get_column_names()
            if len(api_input_data_column_names.data) == 0:
                raise ValueError("No entry names found in ApiInputDataCollection.")
            feature.options.add("ApiInputData", api_input_data_collection.get_column_names())
