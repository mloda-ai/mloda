from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class FeatureValidator:
    @staticmethod
    def validate_and_resolve_compute_framework(
        framework_name: str, available_frameworks: set[type[ComputeFramework]], source: str = "parameter"
    ) -> type[ComputeFramework]:
        for subclass in available_frameworks:
            if framework_name == subclass.get_class_name():
                return subclass
        available = sorted(subclass.get_class_name() for subclass in available_frameworks)
        if source == "options":
            raise ValueError(
                "Compute framework via options not found: no available compute framework has that class name; "
                f"available: {available}"
            )
        raise ValueError(f"Compute framework via {source} {framework_name} not found; available: {available}")

    @staticmethod
    def validate_compute_frameworks_resolved(
        compute_frameworks: set[type[ComputeFramework]] | None, feature_name: str
    ) -> None:
        if not compute_frameworks:
            raise ValueError(
                f"Feature {feature_name} does not have any compute framework. "
                "This function can only be called when the frameworks were resolved."
            )
