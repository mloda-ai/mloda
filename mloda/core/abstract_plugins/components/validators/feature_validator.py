from typing import Optional, Set, Type
from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class FeatureValidator:
    @staticmethod
    def validate_and_resolve_compute_framework(
        framework_name: str, available_frameworks: Set[Type[ComputeFramework]], source: str = "parameter"
    ) -> Type[ComputeFramework]:
        for subclass in available_frameworks:
            if framework_name == subclass.get_class_name():
                return subclass
        raise ValueError(f"Compute framework via {source} {framework_name} not found.")

    @staticmethod
    def validate_compute_frameworks_resolved(
        compute_frameworks: Optional[Set[Type[ComputeFramework]]], feature_name: str
    ) -> None:
        if compute_frameworks is None:
            raise ValueError(
                f"Feature {feature_name} does not have any compute framework. "
                "This function can only be called when the frameworks were resolved."
            )
