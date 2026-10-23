"""Implementation: a plugin that serves one (Source or Operation FeatureGroup, ComputeFramework) pair."""

import inspect
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Generic, TypeVar

from mloda.core.abstract_plugins.components.base_feature_group_version import (
    BaseFeatureGroupVersion,
    ThirdPartyVersionMode,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup

G = TypeVar("G", bound=FeatureGroup)


def _is_concrete(cls: Any) -> bool:
    """No abstract member left; checked by hand because ABCMeta sets the flag after __init_subclass__."""
    return not any(getattr(inspect.getattr_static(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))


def _serves_groups(group: Any) -> bool:
    return isinstance(group, type) and issubclass(group, (SourceFeatureGroup, OperationFeatureGroup))


def _declared_group(cls: Any) -> type | None:
    """The concrete G of the nearest Implementation[G] base, else None (a TypeVar is not resolved through a generic subclass)."""
    for klass in cls.__mro__:
        for alias in klass.__dict__.get("__orig_bases__", ()):
            if typing.get_origin(alias) is Implementation:
                arg = typing.get_args(alias)[0]
                return arg if isinstance(arg, type) and arg is not Any else None
    return None


def _matches_declared_group(cls: Any, group: Any) -> bool:
    declared = _declared_group(cls)
    return declared is None or issubclass(group, declared)


def _names_a_pair(cls: Any) -> bool:
    group = getattr(cls, "feature_group", None)
    framework = getattr(cls, "compute_framework", None)
    return (
        _serves_groups(group)
        and _matches_declared_group(cls, group)
        and isinstance(framework, type)
        and issubclass(framework, ComputeFramework)
    )


class Implementation(ABC, Generic[G]):
    """Names its pair with feature_group and compute_framework and computes it in calculate."""

    feature_group: ClassVar[type[FeatureGroup]]
    compute_framework: ClassVar[type[ComputeFramework]]
    supported: ClassVar[frozenset[str] | None] = None
    divergent: ClassVar[Mapping[str, str]] = types.MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not _is_concrete(cls):
            return
        group: Any = getattr(cls, "feature_group", None)
        framework = getattr(cls, "compute_framework", None)
        if not _serves_groups(group):
            raise TypeError(
                f"{cls.__name__} must set feature_group to a SourceFeatureGroup or OperationFeatureGroup subclass."
            )
        declared = _declared_group(cls)
        if declared is not None and not issubclass(group, declared):
            raise TypeError(
                f"{cls.__name__}.feature_group must be {declared.__name__} or a subclass "
                f"(Implementation[{declared.__name__}]), got {group.__name__}."
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
    def calculate(cls, group: type[G], data: Any, features: FeatureSet, framework: ComputeFramework | None) -> Any:
        """Compute the features; data is the input frame for an operation group, the SourceMatch for a source group."""


def serving_implementation(
    group: type[FeatureGroup],
    compute_framework: type[ComputeFramework],
    implementations: Iterable[type[Implementation[Any]]],
    stop_after: str | None = None,
) -> type[Implementation[Any]] | None:
    """The most derived implementation for group and framework (group MRO outer, framework MRO inner)."""
    by_pair: dict[tuple[type, type], list[type[Implementation[Any]]]] = {}
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


def loaded_implementations() -> list[type[Implementation[Any]]]:
    """Concrete loaded Implementation classes naming a valid pair (a rejected subclass lingers until collected)."""
    return [impl for impl in get_all_subclasses(Implementation) if _is_concrete(impl) and _names_a_pair(impl)]


_ACCESSIBLE: ContextVar[frozenset[type[Implementation[Any]]] | None] = ContextVar(
    "mloda_accessible_implementations", default=None
)


@contextmanager
def accessible_implementations(implementations: Iterable[type[Implementation[Any]]]) -> Iterator[None]:
    """Make the run's accessible implementations current for capability checks inside the block."""
    token = _ACCESSIBLE.set(frozenset(implementations))
    try:
        yield
    finally:
        _ACCESSIBLE.reset(token)


def current_implementations() -> Iterable[type[Implementation[Any]]]:
    """The accessible implementations of the run being planned, else every loaded one."""
    accessible = _ACCESSIBLE.get()
    return loaded_implementations() if accessible is None else accessible
