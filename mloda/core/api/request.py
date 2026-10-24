import contextlib
import contextvars
from copy import copy, deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Generator, cast
from uuid import UUID

from mloda.core.abstract_plugins.components.credential_scrub import redact_exception, registered_secrets_scope
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourcePointerError
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector

# Explicit alias re-exports Engine under no_implicit_reexport so tests can patch mloda.core.api.request.Engine.
from mloda.core.core.engine import Engine as Engine
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.api.plan_info import PlanStep, build_plan_steps, source_of
from mloda.core.api.plan_lock import plan_content_hash, plan_structure_hash
from mloda.core.prepare.identify_feature_group import (
    ComputeFrameworkPinError,
    FeatureResolutionError,
)
from mloda.core.prepare.resolution_types import (
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
from mloda.core.abstract_plugins.function_extender import (
    Extender,
    ExtenderHook,
    GateBypassError,
    LifecycleOutcome,
    call_contained_hook,
    call_run_complete_hook,
    call_run_start_hook,
    error_outcome,
    note_mloda_id,
    reject_old_run_complete_signature,
)
from mloda.core.abstract_plugins.plan_context import PlanContext, PlanOrigin
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.verified_context import current_verified_context
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.runtime.run_id import generate_run_id


_PLAN_ORIGIN: contextvars.ContextVar[PlanOrigin | None] = contextvars.ContextVar("_PLAN_ORIGIN", default=None)


@contextlib.contextmanager
def _plan_origin(origin: PlanOrigin | None) -> Generator[None, None, None]:
    token = _PLAN_ORIGIN.set(origin)
    try:
        yield
    finally:
        _PLAN_ORIGIN.reset(token)


class _CredentialRedaction:
    """Mask secret literals in logs and in any exception escaping the block (a class: see registered_secrets_scope)."""

    def __init__(self, literals: frozenset[str]) -> None:
        self._literals = literals
        self._scope = registered_secrets_scope(literals)

    def __enter__(self) -> None:
        self._scope.__enter__()

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            if exc is not None:
                redact_exception(exc, self._literals)
        finally:
            self._scope.__exit__(exc_type, exc, tb)


_PLAN_TIME_HOOKS = (ExtenderHook.FEATURE_GROUP_MATCHED, ExtenderHook.INPUT_DATA_MATCH)


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
        requested_features: Features | Sequence[Feature | str],
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None = None,
        links: set[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        global_filter: GlobalFilter | None = None,
        api_data: dict[str, dict[str, Any]] | None = None,
        plugin_collector: PluginCollector | None = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: str | None = None,
        parallelization_modes: set[ParallelizationMode] | None = None,
        function_extender: set[Extender] | None = None,
        output_framework: str | type[ComputeFramework] | None = None,
    ) -> None:
        self.runner: None | ExecutionOrchestrator = None
        self.engine: None | Engine = None
        self.function_extender = function_extender
        self.plugin_collector = plugin_collector
        filtered = filter_extenders_by_strict_mode(function_extender, plugin_collector)
        self._extenders: set[Extender] = set(filtered) if filtered is not None else set()
        reject_old_run_complete_signature(self._extenders)
        self._plan_steps: tuple[PlanStep, ...] | None = None
        verified = current_verified_context()
        self.plan_context = PlanContext(
            plan_id=generate_run_id(),
            tenant_id=verified.tenant_id if verified else None,
            project_id=verified.project_id if verified else None,
            principal=verified.principal if verified else None,
            created_at=datetime.now(UTC),
            origin=_PLAN_ORIGIN.get() or "prepare",
        )
        self.plan_id = self.plan_context.plan_id

        with _plan_origin(None):
            call_contained_hook(self._extenders, "on_plan_start", self.plan_context)
            plan_outcome = LifecycleOutcome("succeeded")
            try:
                self._plan(
                    requested_features,
                    compute_frameworks,
                    links,
                    data_access_collection,
                    global_filter,
                    api_data,
                    copy_features,
                    strict_type_enforcement,
                    column_ordering,
                    parallelization_modes,
                    output_framework,
                )
                if self._extenders:
                    self._stamp_structure_hash()
            except BaseException as error:
                plan_outcome = error_outcome(error)
                if self._extenders:
                    note_mloda_id(error, "plan_id", self.plan_context.plan_id)
                raise
            finally:
                call_contained_hook(self._extenders, "on_plan_complete", self.plan_context, plan_outcome)

    def _stamp_structure_hash(self) -> None:
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        self._plan_steps = tuple(self.resolved_plan())
        self.plan_context = replace(
            self.plan_context,
            structure_hash=plan_structure_hash(self._plan_steps),
            content_hash=plan_content_hash(self._plan_steps),
        )
        self.engine.plan_context = self.plan_context

    def _plan(
        self,
        requested_features: Features | Sequence[Feature | str],
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None,
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None,
        global_filter: GlobalFilter | None,
        api_data: dict[str, dict[str, Any]] | None,
        copy_features: bool,
        strict_type_enforcement: bool,
        column_ordering: str | None,
        parallelization_modes: set[ParallelizationMode] | None,
        output_framework: str | type[ComputeFramework] | None,
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

            data_access_collection, self._connection_handles = self._bind_feature_connections(
                _requested_features, data_access_collection
            )

            self._api_data_columns = self._api_data_spec(api_data)
            if self._api_data_columns:
                data_access_collection = (
                    DataAccessCollection() if data_access_collection is None else data_access_collection
                ).with_api_data_columns(self._api_data_columns)

            self.strict_type_enforcement = strict_type_enforcement
            self.features = self._process_features(_requested_features)
            setup_compute_framework = SetupComputeFramework(
                compute_frameworks,
                self.features,
                parallelization_modes=parallelization_modes,
                output_framework=output_framework,
            )
            self.compute_framework = setup_compute_framework.compute_frameworks
            self.framework_preference = setup_compute_framework.framework_preference
            self.links = links
            self.data_access_collection = data_access_collection
            self.global_filter = global_filter
            self.api_data = api_data
            self.output_framework = setup_compute_framework.output_framework
        except SetupConfigurationError:
            raise
        except ValueError as error:
            raise SetupConfigurationError(str(error)) from error

        with self._credential_redaction():
            self.engine = self._create_engine()

    @staticmethod
    def _api_data_spec(api_data: dict[str, dict[str, Any]] | None) -> dict[str, tuple[str, ...]]:
        """Column names per api_data set; a column may belong to one set only."""
        spec: dict[str, tuple[str, ...]] = {}
        column_owner: dict[str, str] = {}
        for key_name, key_data in (api_data or {}).items():
            if not isinstance(key_data, Mapping):
                raise ValueError(
                    f"api_data set '{key_name}' must be a mapping of column to values, got {type(key_data).__name__}."
                )
            for column in key_data:
                if column in column_owner:
                    raise ValueError(
                        f"Column '{column}' appears in api_data sets '{column_owner[column]}' and "
                        f"'{key_name}'; column names must be unique across sets."
                    )
                column_owner[column] = key_name
            spec[key_name] = tuple(key_data)
        return spec

    def _validate_run_api_data(self, api_data: dict[str, Any] | None) -> None:
        """Refuse run data whose sets or column names differ from the ones prepared."""
        if api_data is None:
            return
        prepared = self._api_data_columns
        for key, values in api_data.items():
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"api_data set '{key}' must be a mapping of column to values, got {type(values).__name__}."
                )
        if set(api_data) != set(prepared):
            raise ValueError(
                f"api_data sets {sorted(map(str, api_data))} differ from the prepared sets "
                f"{sorted(map(str, prepared))}: missing {sorted(map(str, set(prepared) - set(api_data)))}, "
                f"unexpected {sorted(map(str, set(api_data) - set(prepared)))}."
            )
        for key, columns in prepared.items():
            if set(api_data[key]) != set(columns):
                raise ValueError(
                    f"api_data set '{key}' differs from the prepared columns: "
                    f"missing {sorted(map(str, set(columns) - set(api_data[key])))}, "
                    f"unexpected {sorted(map(str, set(api_data[key]) - set(columns)))}."
                )

    def _secret_literals(self) -> frozenset[str]:
        dac = getattr(self, "data_access_collection", None)
        return frozenset() if dac is None else dac.secret_literals()

    def _credential_redaction(self, literals: frozenset[str] | None = None) -> "_CredentialRedaction":
        return _CredentialRedaction(self._secret_literals() if literals is None else literals)

    @staticmethod
    def _bind_feature_connections(
        requested_features: Features | Sequence[Feature | str],
        data_access_collection: DataAccessCollection | None,
    ) -> tuple[DataAccessCollection | None, dict[int, str]]:
        """Register each Feature(connection=) on a copy of the collection; return it and the id-to-handle map."""
        entries = requested_features.collection if isinstance(requested_features, Features) else requested_features
        bound = [f for f in entries if isinstance(f, Feature) and f.connection is not None]
        if not bound:
            return data_access_collection, {}
        dac = DataAccessCollection() if data_access_collection is None else data_access_collection.copy()
        handles: dict[int, str] = {}
        for feature in bound:
            connection = feature.connection
            if id(connection) not in handles:
                named = [h for h, c in dac.connections.items() if c is connection and h not in dac.auto_handles]
                if named:
                    handles[id(connection)] = named[0]
                else:
                    index = len(handles)
                    while f"_feature_connection_{index}" in dac.handles():
                        index += 1
                    handles[id(connection)] = f"_feature_connection_{index}"
                    dac.bind_auto_connection(connection, handles[id(connection)])
            feature.options = copy(feature.options)
            feature.options.set("connection_handle", handles[id(connection)])
        return dac, handles

    def _process_features(
        self,
        requested_features: Features | Sequence[Feature | str],
    ) -> Features:
        """Processes the requested features, ensuring they are in the correct format."""
        features = requested_features if isinstance(requested_features, Features) else Features(requested_features)

        for feature in features:
            feature.initial_requested_data = True
            # Framework stamps below are never the feature's own declaration.
            feature.options.lock_own_keys()
            # Propagate strict_type_enforcement to typed features only
            if self.strict_type_enforcement and feature.data_type is not None:
                feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)

        return features

    @classmethod
    def run_all(
        cls,
        features: Features | Sequence[Feature | str],
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None = None,
        links: set[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Any | None = None,
        function_extender: set[Extender] | None = None,
        global_filter: GlobalFilter | None = None,
        api_data: dict[str, dict[str, Any]] | None = None,
        plugin_collector: PluginCollector | None = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: str | None = None,
        carrier: dict[str, str] | None = None,
        child_bootstrap: Callable[[], None] | None = None,
        graceful_shutdown_timeout: float = 2.0,
        output_framework: str | type[ComputeFramework] | None = None,
    ) -> RunResult:
        """
        Run feature computation in one step.

        Args:
            features: Features to compute.
            compute_frameworks: Ordered list of compute frameworks to use; cheapest plan first, then list order.
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
                independent datasets in a single call. Each set returns its own
                result frame (order not guaranteed); a column name shared by two
                sets raises ``SetupConfigurationError``.
            plugin_collector: Plugin collector.
            copy_features: Whether to deep copy features (default True).
            strict_type_enforcement: If True, enforce strict type matching for typed features.
            carrier: Opaque W3C trace-context carrier dict, forwarded to every ``HookContext``.
            child_bootstrap: Picklable, no-argument callable run once in a spawned
                MULTIPROCESSING worker before it processes its first command.
            graceful_shutdown_timeout: Seconds a MULTIPROCESSING worker gets to run its
                extenders' close() before being terminated, shared across every extender in
                that worker. Also the longest a cancelled THREADING run waits for step threads.

        Returns:
            ``RunResult``, a list of computed results, one per feature group, in ``result.plan``
            order. Each element is a compute-framework object
            (e.g. ``pd.DataFrame``, ``pa.Table``) containing only the columns
            for the requested features resolved by that group. When all
            requested features resolve to a single group, the list has one
            element. ``result.plan`` exposes the resolved plan of this run, and
            ``result.frames()`` pairs each result with the ``PlanStep`` that produced it.
            ``result.run_id``/``result.plan_id`` are the run's ``HookContext`` ids.

        Example:
            result = mloda.run_all(
                features,
                api_data={"UserQuery": {"row_index": [0], "query": ["hello"]}}
            )
        """
        with _plan_origin("run_all"):
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
                function_extender=function_extender,
                output_framework=output_framework,
            )
        return session.run(
            api_data=api_data,
            parallelization_modes=parallelization_modes,
            flight_server=flight_server,
            carrier=carrier,
            child_bootstrap=child_bootstrap,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
        )

    @classmethod
    def stream_all(
        cls,
        features: Features | Sequence[Feature | str],
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None = None,
        links: set[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Any | None = None,
        function_extender: set[Extender] | None = None,
        global_filter: GlobalFilter | None = None,
        api_data: dict[str, dict[str, Any]] | None = None,
        plugin_collector: PluginCollector | None = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: str | None = None,
        carrier: dict[str, str] | None = None,
        child_bootstrap: Callable[[], None] | None = None,
        graceful_shutdown_timeout: float = 2.0,
        output_framework: str | type[ComputeFramework] | None = None,
    ) -> ResultStream:
        """Stream results at feature-group granularity.

        Like ``run_all`` but yields each feature group's result as it completes.
        ``list(stream_all(...))`` carries the same elements as ``run_all(...)``. Planning happens
        eagerly at the call; the returned ``ResultStream`` exposes ``plan`` before iteration.
        ``carrier``/``child_bootstrap``/``graceful_shutdown_timeout`` behave as in ``run_all``. Unlike
        ``run_all``, it yields in completion order, which can differ from ``run_all``'s plan order under
        THREADING/MULTIPROCESSING.

        Returns:
            ``ResultStream`` yielding one complete result per feature group; ``stream.frames()`` pairs each
            result with the ``PlanStep`` that produced it. ``stream.run_id``/``stream.plan_id`` are the run's
            ``HookContext`` ids, set before iteration.
        """
        with _plan_origin("stream_all"):
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
                function_extender=function_extender,
                output_framework=output_framework,
            )
        return session.stream_run(
            api_data=api_data,
            parallelization_modes=parallelization_modes,
            flight_server=flight_server,
            carrier=carrier,
            child_bootstrap=child_bootstrap,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
        )

    @classmethod
    def prepare(
        cls,
        features: Features | Sequence[Feature | str],
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None = None,
        links: set[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        global_filter: GlobalFilter | None = None,
        api_data: dict[str, dict[str, Any]] | None = None,
        plugin_collector: PluginCollector | None = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: str | None = None,
        parallelization_modes: set[ParallelizationMode] | None = None,
        function_extender: set[Extender] | None = None,
        output_framework: str | type[ComputeFramework] | None = None,
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
            function_extender=function_extender,
            output_framework=output_framework,
        )

    @classmethod
    def explain(
        cls,
        features: Features | Sequence[Feature | str],
        *,
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None = None,
        links: set[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        global_filter: GlobalFilter | None = None,
        api_data: dict[str, dict[str, Any]] | None = None,
        plugin_collector: PluginCollector | None = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: str | None = None,
        parallelization_modes: set[ParallelizationMode] | None = None,
        function_extender: set[Extender] | None = None,
        output_framework: str | type[ComputeFramework] | None = None,
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
        with _plan_origin("explain"):
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
                function_extender=function_extender,
                output_framework=output_framework,
            )
        return session.resolved_plan()

    @classmethod
    def diagnose(
        cls,
        features: Features | Sequence[Feature | str],
        *,
        compute_frameworks: Sequence[str | type[ComputeFramework]] | None = None,
        links: set[Link] | None = None,
        data_access_collection: DataAccessCollection | None = None,
        global_filter: GlobalFilter | None = None,
        api_data: dict[str, dict[str, Any]] | None = None,
        plugin_collector: PluginCollector | None = None,
        copy_features: bool = True,
        strict_type_enforcement: bool = False,
        column_ordering: str | None = None,
        parallelization_modes: set[ParallelizationMode] | None = None,
        function_extender: set[Extender] | None = None,
        output_framework: str | type[ComputeFramework] | None = None,
    ) -> ResolutionDiagnosis:
        """Whole-request resolution preflight that does not raise on resolution or setup failures.

        Runs the same eager planning as prepare() but projects the outcome instead of raising: on success
        records equals resolution_report() with complete True; on a resolution failure records holds the
        features resolved before the failing one (from the error payload, capped at PARTIAL_RECORDS_CAP on
        huge requests) plus that feature's EvaluationResult and rendered message. A SetupConfigurationError
        (any invalid request argument caught during session setup, before planning, for example an invalid
        column_ordering or an unknown compute framework name) yields only the message. Environment-build
        failures (EnvironmentPreconditionError, RedefinitionConflictError, FrameworkDeclarationError),
        compute-framework pin misuse (ComputeFrameworkPinError) and SourcePointerError are likewise projected into
        the diagnosis instead of raising (an environment failure found after planning, a missing connection, also
        carries the resolved records); any other error propagates, including an exception a breaking extender
        raises unless it is one of the types above. Every parameter after features is keyword-only.
        """
        try:
            with _plan_origin("diagnose"):
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
                    function_extender=function_extender,
                    output_framework=output_framework,
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
            SourcePointerError,
            RedefinitionConflictError,
            FrameworkDeclarationError,
        ) as error:
            return ResolutionDiagnosis(records=[], complete=False, message=str(error))
        except EnvironmentPreconditionError as error:
            return ResolutionDiagnosis(records=list(error.partial_records), complete=False, message=str(error))
        except SetupConfigurationError as error:
            return ResolutionDiagnosis(records=[], complete=False, message=str(error))
        return ResolutionDiagnosis(records=session.resolution_report(), complete=True)

    def resolved_plan(self) -> list[PlanStep]:
        """Return the resolved execution plan of this session as ``PlanStep`` records.

        Available after ``prepare()`` and unchanged by ``run()``. Steps are in dependency order,
        independent ones sorted by content.
        """
        with self._credential_redaction():
            execution_plan = self._execution_plan()
            engine = cast(Engine, self.engine)
            return build_plan_steps(
                execution_plan,
                execution_plan.resolved_join_plan,
                specialized_from=engine.specialized_from,
                output_framework=engine.output_framework,
                auto_handles=(
                    () if engine.data_access_collection is None else engine.data_access_collection.auto_handles
                ),
            )

    def describe_columns(self, step: PlanStep) -> dict[str, DataType | None]:
        """Read the step's source columns from this session, so the PlanStep stays credential-free.

        Opens the source outside a run (no input-data-load hooks).
        """
        with self._credential_redaction():
            group, match, _ = source_of(self._execution_plan(), step)
            return group.describe_columns(match)

    def count_rows(self, step: PlanStep) -> int | None:
        """Count the step's source rows from this session, so the PlanStep stays credential-free.

        Opens the source outside a run (no input-data-load hooks).
        """
        with self._credential_redaction():
            group, match, compute_framework = source_of(self._execution_plan(), step)
            return group.count_rows(match, compute_framework)

    def _execution_plan(self) -> ExecutionPlan:
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        return self.engine.execution_planner

    def resolution_report(self) -> list[ResolutionRecord]:
        """Return this session's per-feature resolution records, captured during planning.

        Available after ``prepare()`` and unchanged by ``run()``. Mirrors ``resolved_plan()``.
        """
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        return deepcopy(self.engine.resolution_records)

    def _build_run_context(
        self,
        carrier: dict[str, str] | None,
        child_bootstrap: Callable[[], None] | None,
        graceful_shutdown_timeout: float = 2.0,
    ) -> RunContext:
        """Derive this run's context from the engine's plan-time base."""
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        identity = current_verified_context() or self.plan_context
        return replace(
            self.engine.run_context,
            run_id=generate_run_id(),
            started_at=datetime.now(UTC),
            carrier=carrier,
            child_bootstrap=child_bootstrap,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
            tenant_id=identity.tenant_id,
            project_id=identity.project_id,
            principal=identity.principal,
            structure_hash=self.plan_context.structure_hash,
        )

    def run(
        self,
        api_data: dict[str, dict[str, Any]] | None = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Any | None = None,
        artifacts: dict[str, Any] | None = None,
        carrier: dict[str, str] | None = None,
        child_bootstrap: Callable[[], None] | None = None,
        graceful_shutdown_timeout: float = 2.0,
    ) -> RunResult:
        """Execute the prepared session and return a ``RunResult`` (plan order, with ``run_id``/``plan_id``).

        Can be called multiple times on the same session with different ``api_data``
        and/or ``artifacts`` to re-run the execution plan against new inputs.

        Args:
            api_data: Fresh runtime data, replacing any data from prepare().
            artifacts: Artifact dict from a previous run's ``get_artifacts()``.
                When provided, feature groups with matching artifact names
                switch to load mode for this run, enabling train-then-predict
                workflows without re-preparing.
            carrier/child_bootstrap: See ``run_all``. Like ``run_id`` (minted per
                call), ``carrier`` may differ on each ``run()`` call.
            graceful_shutdown_timeout: Seconds a MULTIPROCESSING worker gets to run its
                extenders' ``close()`` before being terminated, shared across every extender in
                that worker. Also the longest a cancelled THREADING run waits for step threads.
        """
        run_context = self._build_run_context(carrier, child_bootstrap, graceful_shutdown_timeout)
        plan = list(self._plan_snapshot())
        with self._credential_redaction():
            runner = self._batch_run(
                parallelization_modes, flight_server, api_data=api_data, artifacts=artifacts, run_context=run_context
            )
            items = self._plan_ordered_items(runner, plan)
        return RunResult(
            [r for _, r in items],
            plan,
            items,
            run_id=run_context.run_id,
            plan_id=self.plan_id,
            connection_handles=self._connection_handles,
        )

    def stream_run(
        self,
        api_data: dict[str, dict[str, Any]] | None = None,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Any | None = None,
        artifacts: dict[str, Any] | None = None,
        carrier: dict[str, str] | None = None,
        child_bootstrap: Callable[[], None] | None = None,
        graceful_shutdown_timeout: float = 2.0,
    ) -> ResultStream:
        """Execute the prepared session and stream each feature group's result as it completes.

        Returns a ``ResultStream`` carrying the run's ``run_id``/``plan_id``. carrier/child_bootstrap and the
        active verified_context() scope are read here, at call time, not at first iteration: a function whose
        body itself contains ``yield`` only starts executing on first iteration, which would defer that read
        past the caller's intended scope.
        """
        run_context = self._build_run_context(carrier, child_bootstrap, graceful_shutdown_timeout)
        plan = list(self._plan_snapshot())
        stream = self._start_stream(api_data, parallelization_modes, flight_server, artifacts, run_context)
        return ResultStream(stream, plan, run_id=run_context.run_id, plan_id=self.plan_id)

    def _start_stream(
        self,
        api_data: dict[str, dict[str, Any]] | None,
        parallelization_modes: set[ParallelizationMode],
        flight_server: Any | None,
        artifacts: dict[str, Any] | None,
        run_context: RunContext,
    ) -> Generator[Any, None, None]:
        """Callers build ``run_context`` at call time; no yield here, so a refusal at run start raises at the call."""
        _api_data = api_data if api_data is not None else self.api_data
        self._validate_run_api_data(_api_data)
        stream = self._stream_run_results(flight_server, parallelization_modes, _api_data, artifacts, run_context)
        # Primes the generator so a refusal at run start raises here, not on first iteration.
        next(stream)
        return stream

    def _start_run(self, run_context: RunContext) -> None:
        """Refuse an unguarded identity change, then fire on_run_start."""
        if not self._extenders:
            return
        plan = self.plan_context
        identity_changed = (run_context.tenant_id, run_context.project_id, run_context.principal) != (
            plan.tenant_id,
            plan.project_id,
            plan.principal,
        )
        if identity_changed:
            gates = [e for e in self._extenders if e.never_fall_back and any(h in e.wraps() for h in _PLAN_TIME_HOOKS)]
            if gates and not any(type(gate).on_run_start is not Extender.on_run_start for gate in gates):
                raise GateBypassError(
                    "The run identity differs from the plan identity and no never_fall_back extender "
                    "overrides on_run_start to check it."
                )
        call_run_start_hook(self._extenders, run_context, plan, self._plan_snapshot())

    def _plan_snapshot(self) -> tuple[PlanStep, ...]:
        if self._plan_steps is None:
            self._plan_steps = tuple(self.resolved_plan())
        return self._plan_steps

    def _complete_run(self, run_context: RunContext, outcome: LifecycleOutcome) -> None:
        call_run_complete_hook(self._extenders, run_context, outcome)

    def _stream_run_results(
        self,
        flight_server: Any | None,
        parallelization_modes: set[ParallelizationMode],
        api_data: dict[str, dict[str, Any]] | None,
        artifacts: dict[str, Any] | None,
        run_context: RunContext,
    ) -> Generator[Any, None, None]:
        """Deferred half of ``stream_run``: the priming yield separates run start from computation."""
        outcome = LifecycleOutcome("succeeded")
        literals = self._secret_literals()
        try:
            with self._credential_redaction(literals):
                self.runner = None
                self._start_run(run_context)
                runner = self._setup_engine_runner(parallelization_modes, flight_server)
                # Assigned before any yield so get_result()/get_artifacts() work after an early exit.
                self.runner = runner
            yield None
            try:
                with self._credential_redaction(literals):
                    self._enter_runner_context(
                        runner,
                        parallelization_modes,
                        api_data,
                        artifacts=artifacts,
                        run_context=run_context,
                    )
                stream = runner.compute_stream()
                try:
                    while True:
                        # The scope never spans the yield, so it cannot leak into the caller between items.
                        with self._credential_redaction(literals):
                            item = next(stream, None)
                        if item is None:
                            break
                        yield item
                finally:
                    with self._credential_redaction(literals):
                        stream.close()
            finally:
                with self._credential_redaction(literals):
                    self._exit_runner_context(runner)
        except GeneratorExit:
            outcome = LifecycleOutcome("cancelled")
            raise
        except BaseException as error:
            outcome = error_outcome(error)
            if self._extenders:
                note_mloda_id(error, "run_id", run_context.run_id)
            raise
        finally:
            self._complete_run(run_context, outcome)

    def _batch_run(
        self,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Any | None = None,
        api_data: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
    ) -> ExecutionOrchestrator:
        """Sets up the engine runner and runs the engine computation."""
        # Use stored api_data if not explicitly provided
        _api_data = api_data if api_data is not None else self.api_data
        self._validate_run_api_data(_api_data)
        if run_context is None:
            run_context = self._build_run_context(None, None)
        outcome = LifecycleOutcome("succeeded")
        try:
            self.runner = None
            self._start_run(run_context)
            runner = self._setup_engine_runner(parallelization_modes, flight_server)
            # Assigned before computing so a failed run still exposes its runner, never a prior run's.
            self.runner = runner
            self._run_engine_computation(
                runner,
                parallelization_modes,
                _api_data,
                artifacts=artifacts,
                run_context=run_context,
            )
            return runner
        except BaseException as error:
            outcome = error_outcome(error)
            if self._extenders:
                note_mloda_id(error, "run_id", run_context.run_id)
            raise
        finally:
            self._complete_run(run_context, outcome)

    def _run_engine_computation(
        self,
        runner: ExecutionOrchestrator,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        api_data: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        """Runs the engine computation within a context manager."""
        if not isinstance(runner, ExecutionOrchestrator):
            raise ValueError("Internal error: execution orchestrator not initialized. This is likely a bug in mloda.")

        try:
            self._enter_runner_context(
                runner,
                parallelization_modes,
                api_data,
                artifacts=artifacts,
                run_context=run_context,
            )
            runner.compute()
        finally:
            self._exit_runner_context(runner)

    def _enter_runner_context(
        self,
        runner: ExecutionOrchestrator,
        parallelization_modes: set[ParallelizationMode],
        api_data: dict[str, Any] | None,
        artifacts: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        """Passes the runner the engine's prepare-time extender set."""
        if self.engine is None:
            raise ValueError("Internal error: engine not initialized. This is likely a bug in mloda.")
        function_extender = self.engine.function_extender or None
        if run_context is None:
            run_context = self._build_run_context(None, None)
        runner.__enter__(parallelization_modes, function_extender, api_data, artifacts, run_context)

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
            self.plugin_collector,
            column_ordering=self.column_ordering,
            function_extender=self._extenders,
            plan_context=self.plan_context,
            framework_preference=self.framework_preference,
            output_framework=self.output_framework,
        )
        if not isinstance(engine, Engine):
            raise ValueError("Engine initialization failed.")
        return engine

    def _setup_engine_runner(
        self,
        parallelization_modes: set[ParallelizationMode] = {ParallelizationMode.SYNC},
        flight_server: Any | None = None,
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
        """Return the computed run results in plan order; raises if no run function has executed yet."""
        if self.runner is None:
            raise ValueError("You need to run any run function beforehand.")
        items = self._plan_ordered_items(self.runner, list(self._plan_snapshot()))
        return [r for _, r in items]

    def _plan_ordered_items(self, runner: Any, plan: list[PlanStep]) -> list[tuple[UUID, Any]]:
        position = {step.step_uuid: index for index, step in enumerate(plan)}
        return sorted(runner.get_result_items(), key=lambda item: position.get(item[0], len(position)))

    def get_artifacts(self) -> dict[str, Any]:
        """Return the artifacts produced by the run; raises if no run function has executed yet."""
        if self.runner is None:
            raise ValueError("You need to run any run function beforehand.")
        return self.runner.get_artifacts()
