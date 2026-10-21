"""Implementation: a plugin that serves one (Format or Operation FeatureGroup, ComputeFramework) pair."""

import inspect
import types
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.base_feature_group_version import (
    BaseFeatureGroupVersion,
    ThirdPartyVersionMode,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.input_data.format_feature_group import FormatFeatureGroup
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup


def _is_concrete(cls: Any) -> bool:
    """No abstract member left; checked by hand because ABCMeta sets the flag after __init_subclass__."""
    return not any(getattr(inspect.getattr_static(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))


def _serves_groups(group: Any) -> bool:
    return isinstance(group, type) and issubclass(group, (FormatFeatureGroup, OperationFeatureGroup))


def _names_a_pair(cls: Any) -> bool:
    group = getattr(cls, "feature_group", None)
    framework = getattr(cls, "compute_framework", None)
    return _serves_groups(group) and isinstance(framework, type) and issubclass(framework, ComputeFramework)


class Implementation(ABC):
    """Names its pair with feature_group and compute_framework; a format group calls calculate(group, match, features)."""

    feature_group: ClassVar[type[FeatureGroup]]
    compute_framework: ClassVar[type[ComputeFramework]]
    supported: ClassVar[frozenset[str] | None] = None
    divergent: ClassVar[Mapping[str, str]] = types.MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not _is_concrete(cls):
            return
        group = getattr(cls, "feature_group", None)
        framework = getattr(cls, "compute_framework", None)
        if not _serves_groups(group):
            raise TypeError(
                f"{cls.__name__} must set feature_group to a FormatFeatureGroup or OperationFeatureGroup subclass."
            )
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
    def accepts(cls, feature_name: str, options: Options) -> bool:
        """Whether this implementation can compute the concrete feature; True by default."""
        return True

    @classmethod
    def canonical_subtype(cls, group: type[FeatureGroup], feature_name: str, options: Options) -> str | None:
        """The feature's canonical subtype when the group declares it, else None."""
        subtype = group.resolve_subtype(feature_name, options)
        if subtype is None:
            return None
        canonical = group.canonical_subtype(subtype)
        return canonical if canonical in group.subtype_universe() else None

    @classmethod
    def divergence(cls, group: type[FeatureGroup], feature_name: str, options: Options) -> str | None:
        """The declared reason this implementation differs from the reference for the feature, else None."""
        canonical = cls.canonical_subtype(group, feature_name, options)
        return None if canonical is None else cls.divergent.get(canonical)

    @classmethod
    @abstractmethod
    def calculate(cls, *args: Any, **kwargs: Any) -> Any:
        """Load the source; a format group passes (group, match, features)."""


def serving_implementation(
    group: type[FeatureGroup],
    compute_framework: type[ComputeFramework],
    implementations: Iterable[type[Implementation]],
    stop_after: str | None = None,
) -> type[Implementation] | None:
    """The most derived implementation for group and framework (group MRO outer, framework MRO inner)."""
    by_pair: dict[tuple[type, type], list[type[Implementation]]] = {}
    for impl in implementations:
        by_pair.setdefault((impl.feature_group, impl.compute_framework), []).append(impl)
    for klass in group.__mro__:
        for framework_class in compute_framework.__mro__:
            candidates = by_pair.get((klass, framework_class))
            if candidates:
                return next(c for c in candidates if not any(o is not c and issubclass(o, c) for o in candidates))
        if stop_after is not None and stop_after in klass.__dict__:
            return None
    return None


def loaded_implementations() -> list[type[Implementation]]:
    """Concrete loaded Implementation classes naming a valid pair (a rejected subclass lingers until collected)."""
    return [impl for impl in get_all_subclasses(Implementation) if _is_concrete(impl) and _names_a_pair(impl)]


_ACCESSIBLE: ContextVar[frozenset[type[Implementation]] | None] = ContextVar(
    "mloda_accessible_implementations", default=None
)


@contextmanager
def accessible_implementations(implementations: Iterable[type[Implementation]]) -> Iterator[None]:
    """Make the run's accessible implementations current for capability checks inside the block."""
    token = _ACCESSIBLE.set(frozenset(implementations))
    try:
        yield
    finally:
        _ACCESSIBLE.reset(token)


def current_implementations() -> Iterable[type[Implementation]]:
    """The accessible implementations of the run being planned, else every loaded one."""
    accessible = _ACCESSIBLE.get()
    return loaded_implementations() if accessible is None else accessible
