from typing import Any, Dict, Type
from enum import Enum, auto


class CFWTransformEnums(Enum):
    FUNCTION = auto()
    PARAMETERS = auto()


class ComputeFrameworkTransformMap:
    def __init__(self) -> None:
        self.transform_map: Dict[Any, Any] = {}

    def validate_input(
        self,
        source_compute_frame_work: Type[Any],
        target_compute_frame_work: Type[Any],
        parameters: Dict[str, Any],
    ) -> None:
        if source_compute_frame_work not in self.transform_map:
            raise ValueError(
                f"Source compute framework {source_compute_frame_work} not found in transform map. Thus, we cannot transform to {target_compute_frame_work}."
            )

        if target_compute_frame_work not in self.transform_map[source_compute_frame_work]:
            raise ValueError(
                f"Target compute framework {target_compute_frame_work} not found in transform map for source compute framework {source_compute_frame_work}."
            )

        for parameter, value in parameters.items():
            if self.transform_map[source_compute_frame_work][target_compute_frame_work].get("parameter") is None:
                raise ValueError(
                    f"Parameter {parameter} is not allowed for transformation from {source_compute_frame_work} to {target_compute_frame_work}."
                )

    def validate_output(
        self, source_compute_frame_work: Type[Any], target_compute_frame_work: Type[Any], result_data: Any
    ) -> None:
        if not isinstance(result_data, target_compute_frame_work):
            raise ValueError(
                f"Transformation from {source_compute_frame_work} to {target_compute_frame_work} failed as result is of type {type(result_data)}."
            )

    def transform(
        self,
        data: Any,
        source_compute_frame_work: Type[Any],
        target_compute_frame_work: Type[Any],
        parameters: Dict[str, Any],
    ) -> Any:
        self.validate_input(source_compute_frame_work, target_compute_frame_work, parameters)

        func = self.transform_map[source_compute_frame_work][target_compute_frame_work][CFWTransformEnums.FUNCTION]
        result_data = func(data=data, **parameters)

        self.validate_output(source_compute_frame_work, target_compute_frame_work, result_data)

        return result_data

    def add_transformation(
        self,
        source_compute_frame_work: Any,
        target_cfw: Type[Any],
        function: Any,
        allowed_parameters: Dict[str, Any] = {},
    ) -> None:
        if source_compute_frame_work not in self.transform_map:
            self.transform_map[source_compute_frame_work] = {}

        self.transform_map[source_compute_frame_work][target_cfw] = {
            CFWTransformEnums.FUNCTION: function,
            CFWTransformEnums.PARAMETERS: allowed_parameters,
        }
