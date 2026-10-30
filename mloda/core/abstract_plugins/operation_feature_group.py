"""OperationFeatureGroup: one group per operation, served by one Implementation per ComputeFramework."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework, compute_framework_subclasses
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.implementation import Implementation


class OperationFeatureGroup(FeatureGroup):
    """A FeatureGroup whose compute frameworks are the ones its Implementations serve."""

    @classmethod
    def select_implementation(
        cls, compute_framework: type[ComputeFramework], implementations: Iterable[type["Implementation[Any]"]]
    ) -> "type[Implementation[Any]] | None":
        """The implementation serving this group and framework: group MRO outer, framework MRO inner."""
        from mloda.core.abstract_plugins.implementation import serving_implementation

        if not compute_framework.is_available():
            return None
        return serving_implementation(cls, compute_framework, implementations)

    @classmethod
    def _implemented_frameworks(cls) -> set[type[ComputeFramework]] | None:
        from mloda.core.abstract_plugins.implementation import current_implementations, serving_implementation

        implementations = current_implementations()
        return {
            framework
            for framework in compute_framework_subclasses()
            if serving_implementation(cls, framework, implementations) is not None
        }

    @classmethod
    def _implemented_subtype_matrix(cls) -> dict[str, frozenset[str]]:
        from mloda.core.abstract_plugins.implementation import current_implementations, serving_implementation

        implementations = current_implementations()
        universe = cls.subtype_universe()
        matrix: dict[str, frozenset[str]] = {}
        for framework in cls.compute_framework_definition():
            impl = serving_implementation(cls, framework, implementations)
            supported = None if impl is None else impl.supported
            matrix[framework.get_class_name()] = (
                universe if supported is None else universe & supported
            ) & cls.supported_subtypes(framework)
        return matrix

    @classmethod
    def _overrides(cls, method_name: str) -> bool:
        """True when cls replaces the OperationFeatureGroup classmethod."""
        own = getattr(getattr(cls, method_name), "__func__", None)
        base = getattr(getattr(OperationFeatureGroup, method_name), "__func__", None)
        return own is not base

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
        *,
        part: str | None = None,
    ) -> bool:
        """The serving implementation's supported subtypes and accepts decide, after the SUBTYPES gate."""
        from mloda.core.abstract_plugins.implementation import current_implementations, serving_implementation

        impl = serving_implementation(cls, compute_framework, current_implementations())
        if impl is None or not super().supports_compute_framework(feature_name, options, compute_framework, part=part):
            return False
        if impl.supported is not None:
            canonical = impl.canonical_subtype(cls, str(feature_name), options)
            if canonical is not None and canonical not in impl.supported:
                return False
        return impl.accepts(str(feature_name), options, part=part)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        implementation = features.implementation
        if implementation is None:
            raise ValueError(f"{cls.__name__}.calculate_feature found no implementation on the feature set.")
        marker = features._load_marker
        framework = marker.framework if marker is not None else None
        return implementation.calculate(cls, data, features, framework)
