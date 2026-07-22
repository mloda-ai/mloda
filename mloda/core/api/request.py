from copy import deepcopy
from typing import Any, Generator, Optional

from mloda.core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector

# Explicit alias re-exports Engine under no_implicit_reexport so tests can patch mloda.core.api.request.Engine.
from mloda.core.core.engine import Engine as Engine
from mloda.core.api.plan_info import PlanStep, build_plan_steps
from mloda.core.prepare.identify_feature_group import (
    ComputeFrameworkPinError,
    FeatureResolutionError,
    ResolutionDiagnosis,
    ResolutionRecord,
)
from mloda.core.api.run_result import ResultStream, RunResult
from mloda.core.api.prepare.setup_compute_framework import SetupComputeFramework
from mloda.core.prepare.accessible_plugins import (
    EnvironmentPreconditionError,
    FrameworkDeclarationError,
    RedefinitionConflictError,
    filter_extenders_by_strict_mode,
)
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


class SetupConfigurationError(ValueError):
    """Invalid mlodaAPI setup argument, raised before any feature resolution runs."""


class mlodaAPI:
    """Main API for executing mloda feature requests.

    Two usage patterns:
        Batch: ``mlodaAPI.run_all(features, ...)`` — configure and execute in one call.
        Two-phase: ``session = mlodaAPI.prepare(features, ...)`` then ``session.run(api_data=...)``
        — reuse the same session with different data (e.g. realtime inference).

    For JSON-based feature configuration, see ``load_features_from_config()``.
    """

    def __init__(
        self,
        requested_features: Features | list[Feature | str],
        compute_frameworks: set[type[ComputeFramework]] | Optional[list[str]] = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        plugin_collector: Optional[PluginCollector] = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: Optional[str] = None,
        parallelization_modes: Optional[set[ParallelizationMode]] = None,
    ) -> None:
        # Setup boundary: any invalid request argument surfaces as the typed error before planning.
        try:
            if column_ordering is not None and column_ordering not in ("alphabetical", "request_order"):
                raise SetupConfigurationError(
                    f"column_ordering must be None, 'alphabetical', or 'request_order', got '{column_ordering}'"
                )
            self.column_ordering = column_ordering
            # The features object is potentially changed during the run, so we need to deepcopy it by default, so that follow up runs with the same object are not affected.
            # Set copy_features=False to disable deep copying for use cases where features contain non-copyable objects.
            _requested_features = deepcopy(requested_features) if copy_features else requested_features

            # Handle api_data: create ApiInputDataCollection if api_data provided
            api_input_data_collection: Optional[ApiInputDataCollection] = None
            if api_data is not None and len(api_data) > 0:
                api_input_data_collection = ApiInputDataCollection()
                for key_name, key_data in api_data.items():
                    api_input_data_collection.setup_key_class(key_name, list(key_data.keys()))

            self.strict_type_enforcement = strict_type_enforcement
            self.features = self._process_features(_requested_features, api_input_data_collection)
            self.compute_framework = SetupComputeFramework(
                compute_frameworks, self.features, parallelization_modes=parallelization_modes
            ).compute_frameworks
            self.links = links
            self.data_access_collection = data_access_collection
            self.global_filter = global_filter
            self.api_input_data_collection = api_input_data_collection
            self.api_data = api_data
            self.plugin_collector = plugin_collector
        except SetupConfigurationError:
            raise
        except ValueError as error:
            raise SetupConfigurationError(str(error)) from error

        self.runner: None | ExecutionOrchestrator = None
        self.engine: None | Engine = None

        self.engine = self._create_engine()

    def _process_features(
        self,
        requested_features: Features | list[Feature | str],
        api_input_data_collection: Optional[ApiInputDataCollection],
    ) -> Features:
        """Processes the requested features, ensuring they are in the correct format and adding API input data."""
        features = requested_features if isinstance(requested_features, Features) else Features(requested_features)

        for feature in features:
            feature.initial_requested_data = True
            self._add_api_input_data(feature, api_input_data_collection)
            # Propagate strict_type_enforcement to typed features only
            if self.strict_type_enforcement and feature.data_type is not None:
                feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)

        return features

    @classmethod
    def run_all(
        cls,
        features: Features | list[Feature | str],
        compute_frameworks: set[type[ComputeFramework]] | Optional[list[str]] = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[set[Extender]] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        plugin_collector: Optional[PluginCollector] = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: Optional[str] = None,
    ) -> RunResult:
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
            api_data: Runtime API data as ``{"KeyName": {"column": [values]}}``.
                KeyName is an arbitrary label grouping related columns into
                one input source (e.g. ``"CustomerData"``). During feature
                resolution, columns listed under a KeyName are matched to
                requested features by name. Multiple KeyNames allow passing
                independent datasets in a single call.
            plugin_collector: Plugin collector.
            copy_features: Whether to deep copy features (default True).
            strict_type_enforcement: If True, enforce strict type matching for typed features.

        Returns:
            ``RunResult``, a list of computed results, one per feature group in
            execution plan order. Each element is a compute-framework object
            (e.g. ``pd.DataFrame``, ``pa.Table``) containing only the columns
            for the requested features resolved by that group. When all
            requested features resolve to a single group, the list has one
            element. ``result.plan`` exposes the resolved plan of this run.

        Example:
            result = mloda.run_all(
                features,
                api_data={"UserQuery": {"row_index": [0], "query": ["hello"]}}
            )
        """
        session = cls.prepare(
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
            parallelization_modes=parallelization_modes,
        )
        results = session.run(
            api_data=api_data,
            parallelization_modes=parallelization_modes,
            flight_server=flight_server,
            function_extender=function_extender,
        )
        return RunResult(results, session.resolved_plan())

    @classmethod
    def stream_all(
        cls,
        features: Features | list[Feature | str],
        compute_frameworks: set[type[ComputeFramework]] | Optional[list[str]] = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[set[Extender]] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        plugin_collector: Optional[PluginCollector] = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: Optional[str] = None,
    ) -> ResultStream:
        """Stream results at feature-group granularity.

        Like ``run_all`` but yields each feature group's result as it completes.
        ``list(stream_all(...))`` equals ``run_all(...)``. Planning happens eagerly
        at the call; the returned ``ResultStream`` exposes ``plan`` before iteration.

        Returns:
            ``ResultStream`` yielding one complete result per feature group.
        """
        session = cls.prepare(
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
            parallelization_modes=parallelization_modes,
        )
        # Planning is eager in prepare, so the plan snapshot is available before iteration.
        return ResultStream(
            session.stream_run(
                api_data=api_data,
                parallelization_modes=parallelization_modes,
                flight_server=flight_server,
                function_extender=function_extender,
            ),
            session.resolved_plan(),
        )

    @classmethod
    def prepare(
        cls,
        features: Features | list[Feature | str],
        compute_frameworks: set[type[ComputeFramework]] | Optional[list[str]] = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        plugin_collector: Optional[PluginCollector] = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: Optional[str] = None,
        parallelization_modes: Optional[set[ParallelizationMode]] = None,
    ) -> "mlodaAPI":
        """Build an execution plan without running it.

        Returns a configured mlodaAPI session. Call ``session.run()`` to execute,
        optionally passing fresh ``api_data`` each time.
        """
        return cls(
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
            parallelization_modes=parallelization_modes,
        )

    @classmethod
    def explain(
        cls,
        features: Features | list[Feature | str],
        *,
        compute_frameworks: set[type[ComputeFramework]] | Optional[list[str]] = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        plugin_collector: Optional[PluginCollector] = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: Optional[str] = None,
        parallelization_modes: Optional[set[ParallelizationMode]] = None,
    ) -> list[PlanStep]:
        """Resolve the execution plan without executing it.

        Same as ``prepare(...).resolved_plan()``: no feature is computed. The plan is re-resolved
        from scratch, so this answers "what would this request resolve to", not "what did a previous
        ``run_all`` execute". To mirror a ``run_all`` resolution, pass the same
        ``parallelization_modes``: ``run_all`` defaults to ``{ParallelizationMode.SYNC}`` while
        ``prepare``/``explain`` default to None, and ``SetupComputeFramework`` filters compute
        frameworks by mode.

        Every parameter after ``features`` is keyword-only.
        """
        session = cls.prepare(
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
            parallelization_modes=parallelization_modes,
        )
        return session.resolved_plan()

    @classmethod
    def diagnose(
        cls,
        features: Features | list[Feature | str],
        *,
        compute_frameworks: set[type[ComputeFramework]] | Optional[list[str]] = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
        global_filter: Optional[GlobalFilter] = None,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        plugin_collector: Optional[PluginCollector] = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: Optional[str] = None,
        parallelization_modes: Optional[set[ParallelizationMode]] = None,
    ) -> ResolutionDiagnosis:
        """Non-raising whole-request resolution preflight.

        Runs the same eager planning as prepare() but projects the outcome instead of raising: on success
        records equals resolution_report() with complete True; on a resolution failure records holds the
        features resolved before the failing one (from the error payload, capped at PARTIAL_RECORDS_CAP on
        huge requests) plus that feature's EvaluationResult and rendered message. A SetupConfigurationError
        (any invalid request argument caught during session setup, before planning, for example an invalid
        column_ordering or an unknown compute framework name) yields only the message. Environment-build
        failures (EnvironmentPreconditionError, RedefinitionConflictError, FrameworkDeclarationError) and
        compute-framework pin misuse (ComputeFrameworkPinError) are likewise projected into the diagnosis
        instead of raising; any other error propagates. Every parameter after features is keyword-only.
        """
        try:
            session = cls(
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
                parallelization_modes=parallelization_modes,
            )
        except FeatureResolutionError as error:
            return ResolutionDiagnosis(
                records=list(error.partial_records),
                complete=False,
                feature_name=error.feature_name,
                failed_result=deepcopy(error.result),
                message=str(error),
            )
        except (
            ComputeFrameworkPinError,
            EnvironmentPreconditionError,
            RedefinitionConflictError,
            FrameworkDeclarationError,
        ) as error:
            return ResolutionDiagnosis(records=[], complete=False, message=str(error))
        except SetupConfigurationError as error:
            return ResolutionDiagnosis(records=[], complete=False, message=str(error))
        return ResolutionDiagnosis(records=session.resolution_report(), complete=True)

    def resolved_plan(self) -> list[PlanStep]:
        """Return the resolved execution plan of this session as ``PlanStep`` records.

        Available after ``prepare()`` and unchanged by ``run()``.
        """
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        return build_plan_steps(self.engine.execution_planner)

    def resolution_report(self) -> list[ResolutionRecord]:
        """Return this session's per-feature resolution records, captured during planning.

        Available after ``prepare()`` and unchanged by ``run()``. Mirrors ``resolved_plan()``.
        """
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        return deepcopy(self.engine.resolution_records)

    def run(
        self,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[set[Extender]] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """Execute the prepared session and return results.

        Can be called multiple times on the same session with different ``api_data``
        and/or ``artifacts`` to re-run the execution plan against new inputs.

        Args:
            api_data: Fresh runtime data, replacing any data from prepare().
            artifacts: Artifact dict from a previous run's ``get_artifacts()``.
                When provided, feature groups with matching artifact names
                switch to load mode for this run, enabling train-then-predict
                workflows without re-preparing.
        """
        runner = self._batch_run(
            parallelization_modes, flight_server, function_extender, api_data=api_data, artifacts=artifacts
        )
        self.runner = runner
        return self.get_result()

    def stream_run(
        self,
        api_data: Optional[dict[str, dict[str, Any]]] = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[set[Extender]] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Execute the prepared session and yield each feature group's result as it completes."""
        _api_data = api_data if api_data is not None else self.api_data
        runner = self._setup_engine_runner(parallelization_modes, flight_server)
        try:
            self._enter_runner_context(runner, parallelization_modes, function_extender, _api_data, artifacts=artifacts)
            for _step_uuid, result in runner.compute_stream():
                yield result
        finally:
            self._exit_runner_context(runner)
        self.runner = runner

    def _batch_run(
        self,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Optional[Any] = None,
        function_extender: Optional[set[Extender]] = None,
        api_data: Optional[dict[str, Any]] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> ExecutionOrchestrator:
        """Sets up the engine runner and runs the engine computation."""
        # Use stored api_data if not explicitly provided
        _api_data = api_data if api_data is not None else self.api_data
        runner = self._setup_engine_runner(parallelization_modes, flight_server)
        self._run_engine_computation(runner, parallelization_modes, function_extender, _api_data, artifacts=artifacts)
        self.runner = runner
        return runner

    def _run_engine_computation(
        self,
        runner: ExecutionOrchestrator,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        function_extender: Optional[set[Extender]] = None,
        api_data: Optional[dict[str, Any]] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> None:
        """Runs the engine computation within a context manager."""
        if not isinstance(runner, ExecutionOrchestrator):
            raise ValueError("Internal error: execution orchestrator not initialized. This is likely a bug in mloda.")

        try:
            self._enter_runner_context(runner, parallelization_modes, function_extender, api_data, artifacts=artifacts)
            runner.compute()
        finally:
            self._exit_runner_context(runner)

    def _enter_runner_context(
        self,
        runner: ExecutionOrchestrator,
        parallelization_modes: set[ParallelizationMode],
        function_extender: Optional[set[Extender]],
        api_data: Optional[dict[str, Any]],
        artifacts: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enters the runner context with strict-mode-filtered extenders."""
        function_extender = filter_extenders_by_strict_mode(function_extender, self.plugin_collector)
        runner.__enter__(parallelization_modes, function_extender, api_data, artifacts)

    def _exit_runner_context(self, runner: ExecutionOrchestrator) -> None:
        """Exits the runner context."""
        runner.__exit__(None, None, None)

    def _create_engine(self) -> Engine:
        engine = Engine(
            self.features,
            self.compute_framework,
            self.links,
            self.data_access_collection,
            self.global_filter,
            self.api_input_data_collection,
            self.plugin_collector,
            column_ordering=self.column_ordering,
        )
        if not isinstance(engine, Engine):
            raise ValueError("Engine initialization failed.")
        return engine

    def _setup_engine_runner(
        self,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Optional[Any] = None,
    ) -> ExecutionOrchestrator:
        """Sets up the engine runner based on parallelization mode."""
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")

        runner = (
            self.engine.compute(flight_server)
            if ParallelizationMode.MULTIPROCESSING in parallelization_modes
            else self.engine.compute()
        )

        if not isinstance(runner, ExecutionOrchestrator):
            raise ValueError("ExecutionOrchestrator initialization failed.")

        return runner

    def get_result(self) -> list[Any]:
        """Return the computed run results; raises if no run function has executed yet."""
        if self.runner is None:
            raise ValueError("You need to run any run function beforehand.")
        return self.runner.get_result()

    def get_artifacts(self) -> dict[str, Any]:
        """Return the artifacts produced by the run; raises if no run function has executed yet."""
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
            feature.options.add_to_group("ApiInputData", api_input_data_collection.get_column_names())
