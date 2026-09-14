import functools
from collections.abc import Callable, Iterable
from multiprocessing.managers import BaseManager
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.runtime.parent_death_watchdog import start_parent_death_watchdog

import logging


logger = logging.getLogger(__name__)


def _watchdog_then_initializer(initializer: Callable[..., object] | None, initargs: Iterable[Any]) -> None:
    start_parent_death_watchdog()
    if initializer is not None:
        initializer(*initargs)


class MyManager(BaseManager):
    def start(self, initializer: Callable[..., object] | None = None, initargs: Iterable[Any] = ()) -> None:
        # BaseManager._run_server() runs initializer inside the freshly spawned manager server
        # process, so the watchdog must always run there; a caller-supplied initializer is
        # chained after it rather than replaced. A module-level function plus functools.partial
        # is used, not a closure, because BaseManager.start() pickles the initializer to send it
        # to the spawned server process, and closures are not picklable.
        super().start(functools.partial(_watchdog_then_initializer, initializer, initargs), ())


class CfwManager:
    """
    Manages Compute Frameworks (CFWs) and related data.

    This class handles the registration, merging, and retrieval of Compute Frameworks,
    along with managing multiprocessing resources and error tracking.  It aims to
    centralize and simplify the management of CFWs within the mloda core.

    Does not carry extenders: it is a cross-process object, and crossing that proxy strips
    extender state; workers get extenders on the pickled ComputeFramework they own.
    """

    def __init__(
        self,
        parallelization_modes: set[ParallelizationMode],
    ) -> None:
        """
        Initializes the CfwManager.

        Args:
            parallelization_modes: The set of parallelization modes to use.
        """
        self.parallelization_modes = parallelization_modes

        self.compute_frameworks: dict[
            UUID, tuple[str, set[UUID]]
        ] = {}  # cfw uuid -> (cfw class name, children_if_root)
        self.cfw_merge_relation: dict[UUID, tuple[UUID, str]] = {}  # merge relation

        self.location: str | None = None  # multiprocessing location
        self.error = False  # multiprocessing error flag
        self.msg: Any = None
        self.exc_info: Any = None
        self.exception: Any = None

        self.uuid_flyway_datasets: dict[UUID, set[UUID]] = {}

        self.artifact_to_save: dict[str, Any] = {}

        self.runtime_artifacts: dict[str, Any] | None = None

        self.api_data: dict[str, Any] | None = None

        self.run_context: RunContext = RunContext()

    def add_uuid_flyway_datasets(self, cf_uuid: UUID, object_ids: set[UUID]) -> None:
        """Associates a set of Flyway dataset UUIDs with a Compute Framework UUID."""
        self.uuid_flyway_datasets[cf_uuid] = object_ids

    def get_uuid_flyway_datasets(self, cf_uuid: UUID) -> set[UUID] | None:
        """Retrieves the set of Flyway dataset UUIDs associated with a Compute Framework UUID."""
        return self.uuid_flyway_datasets.get(cf_uuid, None)

    def get_cfw_uuid(
        self,
        cf_class_name: str,
        feature_uuid: UUID,
    ) -> UUID | None:
        """
        Retrieves the UUID of a Compute Framework based on its class name and a feature UUID.

        Usually, the feature UUID is a parent of the current feature; it also matches a cfw's own
        registered key directly, since a TFS-created cfw is keyed by its own step uuid rather than
        listed in anyone's children_if_root. Among matches, the narrowest children_if_root wins (a
        proper subset of every other match's), since a hop back into a framework re-registers the
        same lineage more narrowly; unrelated routes whose children_if_root sets aren't nested keep
        the first one registered, same as before. A genuine tie between two or more matches with the
        EXACT SAME children_if_root set is not this method's job to break: it returns None rather
        than pick one via incidental iteration/registration order, unless a later, strictly narrower
        match resolves it.

        Args:
            cf_class_name: The class name of the Compute Framework.
            feature_uuid: The UUID of the feature.

        Returns:
            The UUID of the Compute Framework, or None if not found or tied.
        """
        best_match: UUID | None = None
        best_children_if_root: set[UUID] | None = None
        tied = False
        for cfw_uuid, value in self.compute_frameworks.items():
            cls_name, children_if_root = value
            if cf_class_name != cls_name:
                continue
            if cfw_uuid != feature_uuid and feature_uuid not in children_if_root:
                continue
            if best_children_if_root is None:
                best_match, best_children_if_root, tied = cfw_uuid, children_if_root, False
            elif children_if_root == best_children_if_root:
                tied = True
            elif children_if_root <= best_children_if_root:
                best_match, best_children_if_root, tied = cfw_uuid, children_if_root, False

        if best_match is None or tied:
            return None

        return self.find_leftmost(best_match, cf_class_name)

    def get_cfw_uuid_as_registered(self, cf_class_name: str, feature_uuid: UUID) -> UUID | None:
        """Same lookup as `get_cfw_uuid`, but without canonicalizing through `find_leftmost`/
        `cfw_merge_relation`, which a join already executed would re-point at a different cfw.

        Applies the same narrowest-children_if_root-wins tie-break as `get_cfw_uuid` (a genuine tie,
        identical children_if_root, returns None rather than picking a winner by iteration order):
        a same-framework JoinStep's source side can span more than one FeatureGroupStep (a
        subclass-clustered case-override hop), and add_tfs's same-framework JoinStep branch tags
        EVERY matching FeatureGroupStep's own children_if_root with the join's link uuid, not just
        one, so this lookup can face the same multi-match shape `get_cfw_uuid` does.
        """
        best_match: UUID | None = None
        best_children_if_root: set[UUID] | None = None
        tied = False
        for cfw_uuid, value in self.compute_frameworks.items():
            cls_name, children_if_root = value
            if cf_class_name != cls_name or feature_uuid not in children_if_root:
                continue
            if best_children_if_root is None:
                best_match, best_children_if_root, tied = cfw_uuid, children_if_root, False
            elif children_if_root == best_children_if_root:
                tied = True
            elif children_if_root <= best_children_if_root:
                best_match, best_children_if_root, tied = cfw_uuid, children_if_root, False

        if tied:
            return None
        return best_match

    def get_unique_cfw_uuid(self, cf_class_name: str, tfs_ids: set[UUID]) -> UUID | None:
        """
        Resolves a set of tfs_ids to at most one distinct Compute Framework UUID.

        Each tfs_id is checked directly against `self.compute_frameworks` first: a tfs_id that IS
        itself some cfw's own registered key for cf_class_name is a strictly stronger signal than
        `get_cfw_uuid`'s membership-or-tie resolution, so it is resolved via `find_leftmost` alone,
        without going through `get_cfw_uuid`. Several tfs_ids each independently matching their own
        cfw this way is the ordinary shape of a consumer reaching several independent hops into the
        same framework, even after a JoinStep merge re-points each into a different destination, so
        it never raises: it is treated the same as no resolution at all, deferring to the caller's
        own-feature-uuid fallback. Tfs_ids that aren't themselves a registered key fall back to
        `get_cfw_uuid`'s children_if_root-membership resolution. Raises only when several tfs_ids
        resolve, via genuine membership, to more than one distinct cfw (ambiguous).
        """
        membership_resolved: set[UUID] = set()
        own_key_resolved: set[UUID] = set()
        for tfs_id in tfs_ids:
            own_entry = self.compute_frameworks.get(tfs_id)
            if own_entry is not None and own_entry[0] == cf_class_name:
                own_key_resolved.add(self.find_leftmost(tfs_id, cf_class_name))
                continue
            resolved = self.get_cfw_uuid(cf_class_name, tfs_id)
            if resolved is not None:
                membership_resolved.add(resolved)

        if len(membership_resolved) > 1:
            raise ValueError(
                internal_invariant_error(
                    "step.tfs_ids resolved to more than one distinct compute framework: ambiguous.",
                    f"cf_class_name={cf_class_name}, resolved cfw_uuids={membership_resolved}, tfs_ids={tfs_ids}",
                )
            )
        if len(membership_resolved) == 1:
            return next(iter(membership_resolved))
        if len(own_key_resolved) == 1:
            return next(iter(own_key_resolved))
        return None

    def resolve_cfw_uuid_by_tfs_ids(
        self, cf_class_name: str, tfs_ids: set[UUID], own_feature_uuid: UUID | None
    ) -> UUID | None:
        """Resolves a step's tfs_ids to at most one cfw uuid, the shared fallback chain behind
        `ComputeFrameworkExecutor.prepare_execute_step`, `ComputeFrameworkExecutor.get_cfw`, and
        `ExecutionOrchestrator._cfw_to_occupy`.

        Tries `get_unique_cfw_uuid(tfs_ids)` first. A resolution that is still literally one of the
        queried tfs_ids only proves that hop's own freshly created cfw exists, not that the caller
        should read from it over an already-established cfw a redundant hop leaves unused, so it is
        cross-checked against `own_feature_uuid` (e.g. a chained join's final destination) and
        overridden when that resolves. When `get_unique_cfw_uuid` resolves nothing at all (several
        tfs_ids are each independently some cfw's own key - an ordinary shape, see
        `get_unique_cfw_uuid`'s own docstring), `own_feature_uuid` is tried first, then each tfs_id
        candidate directly, since any one of them beats treating the step as having no cfw at all.
        """
        resolved_uuid = self.get_unique_cfw_uuid(cf_class_name, tfs_ids)

        if resolved_uuid is not None and resolved_uuid in tfs_ids and own_feature_uuid is not None:
            by_feature_uuid = self.get_cfw_uuid(cf_class_name, own_feature_uuid)
            if by_feature_uuid is not None:
                resolved_uuid = by_feature_uuid

        if resolved_uuid is None:
            if own_feature_uuid is not None:
                resolved_uuid = self.get_cfw_uuid(cf_class_name, own_feature_uuid)
            if resolved_uuid is None:
                for candidate_uuid in tfs_ids:
                    resolved_uuid = self.get_cfw_uuid(cf_class_name, candidate_uuid)
                    if resolved_uuid is not None:
                        break

        return resolved_uuid

    def add_to_merge_relation(self, left_uuid: UUID, right_uuid: UUID, cls_name: str) -> None:
        """
        Adds a merge relation between two Compute Framework UUIDs.

        Calling this a second time for a pair already merged, with either uuid on either
        side, can create a cycle in cfw_merge_relation, which find_leftmost will detect and raise on.

        Args:
            left_uuid: The UUID of the left Compute Framework.
            right_uuid: The UUID of the right Compute Framework.
            cls_name: The class name of the Compute Framework.
        """
        self.cfw_merge_relation[right_uuid] = (left_uuid, cls_name)

        if left_uuid not in self.cfw_merge_relation:
            self.cfw_merge_relation[left_uuid] = (left_uuid, cls_name)

    def find_leftmost(self, uuid: UUID, cls_name: str) -> UUID:
        """
        Finds the leftmost UUID in a merge relation chain.

        Args:
            uuid: The starting UUID.
            cls_name: The class name of the Compute Framework.

        Returns:
            The leftmost UUID in the chain.

        Raises:
            ValueError: If cfw_merge_relation contains a cycle.
        """
        if uuid not in self.cfw_merge_relation:
            return uuid

        start_uuid = uuid
        leftmost_uuid = uuid
        visited = {uuid}

        while self.cfw_merge_relation[uuid][0] != uuid:
            uuid = self.cfw_merge_relation[uuid][0]

            if uuid in visited:
                raise ValueError(
                    internal_invariant_error(
                        "cfw_merge_relation contains a cycle while resolving leftmost uuid.",
                        f"start_uuid={start_uuid}, cls_name={cls_name}, visited={visited}",
                    )
                )
            visited.add(uuid)

            if self.cfw_merge_relation[uuid][1] == cls_name:
                leftmost_uuid = uuid
        return leftmost_uuid

    def add_cfw_to_compute_frameworks(self, uuid: UUID, cls_name: str, children_if_root: set[UUID]) -> None:
        """
        Adds a Compute Framework to the registered frameworks.

        Args:
            uuid: The UUID of the Compute Framework.
            cls_name: The class name of the Compute Framework.
            children_if_root: The set of child UUIDs if the CFW is a root..
        """
        if self.compute_frameworks.get(uuid):
            raise ValueError(f"UUID {uuid} already exists in compute_frameworks")
        self.compute_frameworks[uuid] = (cls_name, children_if_root)

    def get_initialized_compute_framework_uuid(self, cf_class: type[ComputeFramework], feature_uuid: UUID) -> UUID:
        """
        Retrieves the UUID of an initialized Compute Framework.

        Args:
            cf_class: The class of the Compute Framework.
            feature_uuid: The UUID of the feature.

        Returns:
            The UUID of the Compute Framework.
        """
        cfw_uuid = self.get_cfw_uuid(cf_class.get_class_name(), feature_uuid)

        if cfw_uuid is None:
            raise ValueError("No compute framework registered.")
        return cfw_uuid

    def set_location(self, location: str) -> None:
        """Sets the location for multiprocessing."""
        if not self.location:
            self.location = location

    def get_location(self) -> str | None:
        """Retrieves the location for multiprocessing."""
        return self.location

    def get_parallelization_modes(self) -> set[ParallelizationMode]:
        """Retrieves the set of parallelization modes."""
        return self.parallelization_modes

    def set_error(self, msg: Any, exc_info: Any, exception: Any = None) -> None:
        """Sets an error message and exception information."""
        self.error = True
        self.msg = msg
        self.exc_info = exc_info
        self.exception = exception

    def get_error(self) -> bool:
        """Retrieves the error flag."""
        return self.error

    def take_error_exception(self) -> Any:
        """Hands over the original exception object, if captured, and drops the register's reference."""
        # error and msg stay set on purpose, so a later flag read still raises the typed fallback.
        exception = self.exception
        self.exception = None
        return exception

    def get_error_msg(self) -> Any:
        """Retrieves the error message."""
        return self.msg

    def get_error_exc_info(self) -> Any:
        """Retrieves the exception information."""
        return self.exc_info

    def set_artifact_to_save(self, artifact_name: str, artifact: Any) -> None:
        """
        Saves an artifact or meta-information to the artifact_to_save dictionary.

        Args:
            artifact_name: The name of the artifact.
            artifact: The artifact to save.
        """
        if artifact_name in self.artifact_to_save:
            raise ValueError(f"Artifact name {artifact_name} already exists.")

        self.artifact_to_save[artifact_name] = artifact

    def get_artifacts(self) -> dict[str, Any]:
        """Retrieves the dictionary of saved artifacts."""
        return self.artifact_to_save

    def set_runtime_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Sets runtime artifacts passed to run() for load-mode resolution."""
        self.runtime_artifacts = artifacts

    def get_runtime_artifacts(self) -> dict[str, Any] | None:
        """Retrieves runtime artifacts, or None if not provided."""
        return self.runtime_artifacts

    def set_api_data(self, api_data: dict[str, Any]) -> None:
        """Sets the API data."""
        self.api_data = api_data

    def get_api_data_by_name(self, key: str) -> Any | None:
        """
        Retrieves API data by name.

        Args:
            key: The name of the API data.

        Returns:
            The API data, or None if not found.
        """
        if self.api_data is None:
            raise ValueError("No api data set.")

        api_data = self.api_data.get(key, None)

        if api_data is None:
            raise ValueError(f"Api data with key {key} not found.")

        return api_data

    def set_run_context(self, run_context: RunContext) -> None:
        """Sets the run context every compute framework of this run receives."""
        self.run_context = run_context

    def get_run_context(self) -> RunContext:
        """Retrieves the run context."""
        return self.run_context
