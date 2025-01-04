from typing import Any, Optional, Set, Type, Union
from uuid import UUID, uuid4
from mloda_core.abstract_plugins.components.cfw_transformer import ComputeFrameworkTransformMap
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.core.cfw_manager import CfwManager
from mloda_core.core.step.abstract_step import Step
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.runtime.flight.flight_server import FlightServer


class TransformFrameworkStep(Step):
    def __init__(
        self,
        from_framework: Type[ComputeFrameWork],
        to_framework: Type[ComputeFrameWork],
        required_uuids: Set[UUID],
        from_feature_group: Type[AbstractFeatureGroup],
        to_feature_group: Type[AbstractFeatureGroup],
        link_id: Optional[UUID] = None,
        right_framework_uuids: Set[UUID] = set(),
    ) -> None:
        self.from_framework = from_framework
        self.to_framework = to_framework
        self.required_uuids = required_uuids
        self.uuid = uuid4()
        self.from_feature_group = from_feature_group
        self.to_feature_group = to_feature_group
        self.link_id = link_id

        # This variable is only set, if the TFS was requested by a joinstep.
        self.right_framework_uuid: Optional[UUID] = None
        if right_framework_uuids is not None and len(right_framework_uuids) > 0:
            self.right_framework_uuid = next(iter(right_framework_uuids))

        self.step_is_done = False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TransformFrameworkStep):
            return False
        return (
            self.from_framework == other.from_framework
            and self.to_framework == other.to_framework
            and self.from_feature_group == other.from_feature_group
            and self.to_feature_group == other.to_feature_group
        )

    def __hash__(self) -> int:
        return hash((self.from_framework, self.to_framework, self.from_feature_group, self.to_feature_group))

    def get_uuids(self) -> Set[UUID]:
        return {self.uuid}

    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFrameWork,
        from_cfw: Optional[Union[ComputeFrameWork, UUID]] = None,
        data: Optional[Any] = None,
    ) -> Optional[Any]:
        self.location = cfw_register.get_location()

        if from_cfw is None:
            raise ValueError("From_cfw is None in transform_framework_step. This should not happen.")

        data = self.get_data(from_cfw)
        column_names = self.get_column_names(cfw_register, from_cfw)
        data = self.transform(cfw, data, column_names)

        cfw.set_data(data)

        if self.location:
            cfw.upload_finished_data(self.location)
            return data
        return None

    def get_column_names(self, cfw_register: CfwManager, from_cfw: Union[ComputeFrameWork, UUID]) -> Set[str]:
        if self.location and isinstance(from_cfw, UUID):
            return cfw_register.get_column_names(from_cfw)

        if isinstance(from_cfw, UUID):
            raise ValueError("From_cfw is a UUID, but we are not using flightserver.")

        return from_cfw.get_column_names()

    def get_data(self, cfw: Union[ComputeFrameWork, UUID]) -> Any:
        """
        This method is used to get the data from the compute framework.
        If we are using multiprocessing, we use flightserver to transport the data.

        If we are not using multiprocessing, we just get the data from the compute framework.
        """
        if isinstance(cfw, UUID) and self.location:
            data = FlightServer.download_table(self.location, str(cfw))
            return data

        if isinstance(cfw, UUID):
            raise ValueError("From_cfw is a UUID, but we are not using flightserver.")

        return cfw.get_data()

    def set_data(self, cfw: ComputeFrameWork, data: Any) -> None:
        cfw.set_data(data)

    def transform(self, cfw: ComputeFrameWork, data: Any, feature_names: Set[str]) -> Any:
        transform_map = self.set_transform_map()
        return cfw.transform(data, feature_names, transform_map)

    def set_transform_map(self) -> ComputeFrameworkTransformMap:
        # get expected data framework
        _from_concrete_framework = self.from_framework.expected_data_framework()
        _to_concrete_framework = self.to_framework.expected_data_framework()

        if _from_concrete_framework == _to_concrete_framework:
            return ComputeFrameworkTransformMap()

        # get transformation function for expected data framework if exists
        from_func, from_parameters = self.from_framework.get_framework_transform_functions(
            from_other=False, other=_to_concrete_framework
        )

        to_func, to_parameters = self.to_framework.get_framework_transform_functions(
            from_other=True, other=_from_concrete_framework
        )

        if from_func is None and to_func is None:
            raise ValueError(f"Transformation from {_from_concrete_framework} to {_to_concrete_framework} not found.")

        if from_func and to_func:
            raise ValueError(
                f"Two framework transformations found for {_from_concrete_framework} to {_to_concrete_framework}."
            )

        if from_func:
            parameters = from_parameters
        else:
            parameters = to_parameters

        if parameters.get("defensive"):
            raise ValueError(
                f"Transformation parameter from {_from_concrete_framework} to {_to_concrete_framework} not found."
            )

        cfw_transform_map = ComputeFrameworkTransformMap()
        cfw_transform_map.add_transformation(
            _from_concrete_framework,
            _to_concrete_framework,
            from_func or to_func,
            parameters,
        )

        return cfw_transform_map
