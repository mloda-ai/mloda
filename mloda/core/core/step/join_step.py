from typing import Optional, Any
from uuid import UUID, uuid4
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import ExtenderHook, _invoke_extender
from mloda.core.abstract_plugins.hook_context import HookContext, instrument
from mloda.core.core.cfw_manager import CfwManager

from mloda.core.core.step.abstract_step import Step
from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.runtime.flight.flight_server import FlightServer


class JoinStep(Step):
    def __init__(
        self,
        link: Link,
        destination_framework: type[ComputeFramework],
        source_framework: type[ComputeFramework],
        required_uuids: set[UUID],
        destination_framework_uuids: set[UUID],
        source_framework_uuids: set[UUID],
        swap_merge_sides: bool = False,
        token: Optional[UUID] = None,
    ) -> None:
        self.link = link
        self.swap_merge_sides = swap_merge_sides
        self.destination_framework = destination_framework
        self.source_framework = source_framework
        self.required_uuids = required_uuids
        self.destination_framework_uuids = destination_framework_uuids
        self.source_framework_uuids = source_framework_uuids
        self.uuid = token if token is not None else uuid4()
        self.step_is_done = False

    def get_uuids(self) -> set[UUID]:
        """Only this step's uuid is a completion token; the link uuid is shared by both orientations."""
        return {self.uuid}

    def _merge_data(self, cfw: ComputeFramework, from_cfw_data: Any) -> None:
        """Merges data from another ComputeFramework into the current one."""
        extender = cfw.get_function_extender(ExtenderHook.JOIN)
        if extender is None:
            self._do_merge_data(cfw, from_cfw_data)
            return

        context = HookContext(
            hook=ExtenderHook.JOIN,
            feature_group_class="",
            feature_group_version="",
            plugin_version=None,
            feature_names=(),
            input_features=None,
            compute_framework_name=cfw.get_class_name(),
            join_type=self.link.jointype.value,
            join_keys=self._join_keys(),
            run_id=cfw.run_id,
            carrier=cfw.carrier,
            worker_index=cfw.worker_index,
        )
        with context.activate():
            _invoke_extender(extender, instrument(context, self._do_merge_data), cfw, from_cfw_data)

    def _join_keys(self) -> Optional[tuple[str, ...]]:
        """Pairs each left column with its corresponding right column; None for APPEND/UNION, which merge without keys."""
        if self.link.jointype in (JoinType.APPEND, JoinType.UNION):
            return None
        return tuple(f"{left}={right}" for left, right in zip(self.link.left_index.index, self.link.right_index.index))

    def _do_merge_data(self, cfw: ComputeFramework, from_cfw_data: Any) -> None:
        merge_engine_class = cfw.merge_engine()
        framework_connection = cfw.get_framework_connection_object()
        merge_engine_instance = merge_engine_class(framework_connection)

        # Link indices are bound to the feature groups, so the left group's data must stay the left argument.
        if self.swap_merge_sides:
            cfw.data = merge_engine_instance.merge(from_cfw_data, cfw.data, self.link)
        else:
            cfw.data = merge_engine_instance.merge(cfw.data, from_cfw_data, self.link)
        cfw.set_column_names()

    def _upload_data_if_needed(self, cfw: ComputeFramework, cfw_register: CfwManager) -> None:
        """Uploads the merged data to Flyway if a location is configured."""
        if self.location:
            if cfw_register.get_uuid_flyway_datasets(cfw.uuid):
                cfw.upload_finished_data(self.location)

    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFramework,
        from_cfw: Optional[ComputeFramework | UUID] = None,
        data: Optional[Any] = None,
    ) -> Optional[Any]:
        self.location = cfw_register.get_location()

        if from_cfw is None:
            raise ValueError("From_cfw should not be none for join step.")
        from_cfw_data, from_cfw_uuid = self.get_data(from_cfw, cfw)

        self._merge_data(cfw, from_cfw_data)

        cfw_register.add_to_merge_relation(cfw.uuid, from_cfw_uuid, cls_name=cfw.get_class_name())

        self._upload_data_if_needed(cfw, cfw_register)

        return None

    def get_data(self, from_cfw: UUID | ComputeFramework, cfw: ComputeFramework) -> Any:
        """
        This method is used to get the data from the compute framework.
        If we are using multiprocessing, we use flightserver to transport the data.

        If we are not using multiprocessing, we just get the data from the compute framework.
        """
        if self.location and isinstance(from_cfw, UUID):
            transformer = ComputeFrameworkTransformer()

            data = FlightServer.download_table(self.location, str(from_cfw))
            data = cfw.convert_flight_server_data_back(data, transformer)
            return data, from_cfw
        if isinstance(from_cfw, UUID):
            raise ValueError("From_cfw is a UUID, but we are not using flightserver.")
        return from_cfw.get_data(), from_cfw.uuid

    def matched(self, other_framework: type[ComputeFramework], uuid: UUID) -> Optional[UUID]:
        """
        If matched, return the uuid of the join step.
        """

        if uuid not in self.destination_framework_uuids and uuid not in self.source_framework_uuids:
            return None

        if other_framework == self.destination_framework:
            return self.uuid

        if other_framework == self.source_framework:
            return self.uuid
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JoinStep):
            return False
        return self.uuid == other.uuid

    def __hash__(self) -> int:
        return hash(self.uuid)
