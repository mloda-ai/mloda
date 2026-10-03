from collections.abc import Sequence
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class SetupComputeFramework:
    """A class to create the compute framework and do basic validation."""

    def __init__(
        self,
        user_compute_frameworks: Sequence[str | type[ComputeFramework]] | None,
        features: Features,
        parallelization_modes: set[ParallelizationMode] | None = None,
    ) -> None:
        available_compute_frameworks = get_all_subclasses(ComputeFramework)

        if user_compute_frameworks is not None and (
            not isinstance(user_compute_frameworks, Sequence) or isinstance(user_compute_frameworks, (str, bytes))
        ):
            raise ValueError(
                "compute_frameworks must be an ordered list, the first entry preferred, "
                f'for example ["PolarsDataFrame", "PandasDataFrame"], got {type(user_compute_frameworks).__name__}.'
            )

        preference: dict[type[ComputeFramework], int] = {}
        if user_compute_frameworks:
            matched = self.filter_user_set_in_available_sub_classes(
                user_compute_frameworks, available_compute_frameworks
            )
            # Same-named classes share the index of the first entry naming them.
            preference = {s: self._position(s, user_compute_frameworks) for s in matched}
            available_compute_frameworks = matched

        self.framework_preference = preference

        if parallelization_modes is not None:
            available_compute_frameworks = self._filter_by_parallelization_modes(
                available_compute_frameworks, parallelization_modes
            )
            if not available_compute_frameworks:
                raise ValueError(
                    f"No compute frameworks support the requested parallelization modes: {parallelization_modes}."
                )

        self.validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
            features, available_compute_frameworks
        )

        self.compute_frameworks = available_compute_frameworks

    def validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
        self, features: Features, available_compute_frameworks: set[type[ComputeFramework]]
    ) -> None:
        for feature in features.collection:
            if feature.compute_frameworks and not any(
                cf in available_compute_frameworks for cf in feature.compute_frameworks
            ):
                raise ValueError(
                    f"Feature {feature.name} has compute frameworks {feature.compute_frameworks} not in {available_compute_frameworks}."
                )

    def _filter_by_parallelization_modes(
        self,
        compute_frameworks: set[type[ComputeFramework]],
        parallelization_modes: set[ParallelizationMode],
    ) -> set[type[ComputeFramework]]:
        return {
            cfw
            for cfw in compute_frameworks
            if cfw.supported_parallelization_modes().intersection(parallelization_modes)
        }

    def filter_user_set_in_available_sub_classes(
        self,
        api_request_compute_frameworks: Sequence[str | type[ComputeFramework]],
        sub_classes: set[type[ComputeFramework]],
    ) -> set[type[ComputeFramework]]:
        unmatched = [
            entry
            for entry in api_request_compute_frameworks
            if not any(sub is entry or sub.get_class_name() == entry for sub in sub_classes)
        ]
        if unmatched:
            available_names = sorted(cls.get_class_name() for cls in sub_classes)
            plugin_loader_hint = " Did you call PluginLoader.all()?" if not sub_classes else ""
            raise ValueError(
                f"No given compute frameworks {unmatched} found in "
                f"available compute frameworks: {available_names}.{plugin_loader_hint}"
            )
        count = len(api_request_compute_frameworks)
        return {sub for sub in sub_classes if self._position(sub, api_request_compute_frameworks) < count}

    @staticmethod
    def _position(sub: type[ComputeFramework], entries: Sequence[str | type[ComputeFramework]]) -> int:
        """Index of the first entry naming sub, else len(entries)."""
        for index, entry in enumerate(entries):
            if sub is entry or sub.get_class_name() == entry:
                return index
        return len(entries)
