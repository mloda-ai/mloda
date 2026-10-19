"""Implementation: a plugin that serves one (FormatFeatureGroup, ComputeFramework) pair with its own calculate."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.base_feature_group_version import (
    BaseFeatureGroupVersion,
    ThirdPartyVersionMode,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.input_data.format_feature_group import FormatFeatureGroup


def _is_concrete(cls: Any) -> bool:
    return not getattr(cls.__dict__.get("calculate", cls.calculate), "__isabstractmethod__", False)


def _names_a_pair(cls: Any) -> bool:
    group = getattr(cls, "feature_group", None)
    framework = getattr(cls, "compute_framework", None)
    return (
        isinstance(group, type)
        and issubclass(group, FormatFeatureGroup)
        and isinstance(framework, type)
        and issubclass(framework, ComputeFramework)
    )


class Implementation(ABC):
    """Names its pair with feature_group and compute_framework; a format group calls calculate(group, match, features)."""

    feature_group: ClassVar[type[FeatureGroup]]
    compute_framework: ClassVar[type[ComputeFramework]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not _is_concrete(cls):
            return
        group = getattr(cls, "feature_group", None)
        framework = getattr(cls, "compute_framework", None)
        if not (isinstance(group, type) and issubclass(group, FormatFeatureGroup)):
            raise TypeError(f"{cls.__name__} must set feature_group to a FormatFeatureGroup subclass.")
        if not (isinstance(framework, type) and issubclass(framework, ComputeFramework)):
            raise TypeError(f"{cls.__name__} must set compute_framework to a ComputeFramework subclass.")

    @classmethod
    def version(cls) -> str:
        """Version: mloda version, module name and the implementation hash of the code this class runs."""
        return BaseFeatureGroupVersion.version(cls)

    @classmethod
    def version_third_party_mode(cls) -> ThirdPartyVersionMode:
        """Whether version()'s hash reacts to reachable third-party dependency versions; INCLUDE by default."""
        return ThirdPartyVersionMode.INCLUDE

    @classmethod
    @abstractmethod
    def calculate(cls, *args: Any, **kwargs: Any) -> Any:
        """Load the source; a format group passes (group, match, features)."""


def loaded_implementations() -> list[type[Implementation]]:
    """Concrete loaded Implementation classes naming a valid pair (a rejected subclass lingers until collected)."""
    return [impl for impl in get_all_subclasses(Implementation) if _is_concrete(impl) and _names_a_pair(impl)]
