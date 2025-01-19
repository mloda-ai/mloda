from typing import Optional, Set, Type, Any, Union
from uuid import UUID, uuid4
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.core.cfw_manager import CfwManager

from mloda_core.core.step.abstract_step import Step
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.runtime.flight.flight_server import FlightServer


class JoinStep(Step):
    def __init__(
        self,
        link: Link,
        left_framework: Type[ComputeFrameWork],
        right_framework: Type[ComputeFrameWork],
        required_uuids: Set[UUID],
        left_framework_uuids: Set[UUID],
        right_framework_uuids: Set[UUID],
    ) -> None:
        self.link = link
        self.left_framework = left_framework
        self.right_framework = right_framework
        self.required_uuids = required_uuids
        self.left_framework_uuids = left_framework_uuids
        self.right_framework_uuids = right_framework_uuids
        self.uuid = uuid4()
        self.step_is_done = False

    def get_uuids(self) -> Set[UUID]:
        return {self.uuid, self.link.uuid}

    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFrameWork,
        from_cfw: Optional[ComputeFrameWork] = None,
        data: Optional[Any] = None,
    ) -> Optional[Any]:
        self.location = cfw_register.get_location()

        if from_cfw is None:
            raise ValueError("From_cfw should not be none for join step.")
        from_cfw_data, from_cfw_uuid = self.get_data(from_cfw, cfw)

        merge_engine = cfw.merge_engine()
        cfw.data = merge_engine().merge(
            cfw.data, from_cfw_data, self.link.jointype, self.link.left_index, self.link.right_index
        )

        cfw_register.add_to_merge_relation(cfw.uuid, from_cfw_uuid, cls_name=cfw.get_class_name())

        if self.location:
            # check if dataset was uploaded before -> then overwrite
            if cfw_register.get_uuid_flyway_datasets(cfw.uuid):
                cfw.upload_finished_data(self.location)

        return None

    def get_data(self, from_cfw: Union[UUID, ComputeFrameWork], cfw: ComputeFrameWork) -> Any:
        """
        This method is used to get the data from the compute framework.
        If we are using multiprocessing, we use flightserver to transport the data.

        If we are not using multiprocessing, we just get the data from the compute framework.
        """
        if self.location and isinstance(from_cfw, UUID):
            data = FlightServer.download_table(self.location, str(from_cfw))
            data = cfw.convert_flyserver_data_back(data)
            return data, from_cfw
        if isinstance(from_cfw, UUID):
            raise ValueError("From_cfw is a UUID, but we are not using flightserver.")
        return from_cfw.get_data(), from_cfw.uuid

    def matched(self, other_framework: Type[ComputeFrameWork], uuid: UUID) -> Optional[UUID]:
        """
        If matched, return the uuid of the join step.
        """

        if uuid not in self.required_uuids:
            return None

        if other_framework == self.left_framework:
            return self.uuid

        if other_framework == self.right_framework:
            return self.uuid
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JoinStep):
            return False
        return self.uuid == other.uuid

    def __hash__(self) -> int:
        return hash(self.uuid)
