from typing import Optional, Set, Type, Union
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.utils import get_all_subclasses


class SetupComputeFramework:
    """A class to create the compute framework and do basic validation."""

    def __init__(
        self, user_compute_frameworks: Union[Set[Type[ComputeFrameWork]], Optional[list[str]]], features: Features
    ) -> None:
        available_compute_frameworks = get_all_subclasses(ComputeFrameWork)

        if user_compute_frameworks:
            if isinstance(user_compute_frameworks, list):
                user_set_compute_frameworks: set[str | Type[ComputeFrameWork]] = set(user_compute_frameworks)
            else:
                user_set_compute_frameworks = user_compute_frameworks  # type: ignore

            available_compute_frameworks = self.filter_user_set_in_available_sub_classes(
                user_set_compute_frameworks, available_compute_frameworks
            )

        self.validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
            features, available_compute_frameworks
        )

        self.compute_frameworks = available_compute_frameworks

    def validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
        self, features: Features, available_compute_frameworks: set[type[ComputeFrameWork]]
    ) -> None:
        for feature in features.collection:
            if feature.compute_frameworks and not any(
                cf in available_compute_frameworks for cf in feature.compute_frameworks
            ):
                raise ValueError(
                    f"Feature {feature.name} has compute frameworks {feature.compute_frameworks} not in {available_compute_frameworks}."
                )

    def filter_user_set_in_available_sub_classes(
        self,
        api_request_compute_frameworks: set[str | Type[ComputeFrameWork]],
        sub_classes: set[type[ComputeFrameWork]],
    ) -> set[type[ComputeFrameWork]]:
        compute_frameworks = set()
        compute_frameworks = {
            sub
            for sub in sub_classes
            if sub.get_class_name() in api_request_compute_frameworks or sub in api_request_compute_frameworks
        }

        if not compute_frameworks:
            raise ValueError(
                f"No given compute frameworks {api_request_compute_frameworks} found in available compute frameworks: {sub_classes}."
            )
        return compute_frameworks
