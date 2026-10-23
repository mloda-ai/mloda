"""Plan-time pass turning Feature(implementation=...) pins into per-step allowed compute frameworks."""

from collections import defaultdict
from collections.abc import Collection
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations, serving_implementation
from mloda.core.abstract_plugins.components.input_data.format_feature_group import FormatFeatureGroup
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation_pin_error import ImplementationPinError
from mloda.core.prepare.graph.graph import Graph

Framework = type[ComputeFramework]


def _qualified(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class ResolveImplementationPins:
    """Resolves pins on requested features over their transitive inputs; never mutates a Feature."""

    def __init__(self, graph: Graph, accessible: Collection[type[Implementation[Any]]]) -> None:
        self.graph = graph
        self.accessible = list(accessible)
        self.loaded = loaded_implementations()
        self.inputs: dict[UUID, set[UUID]] = defaultdict(set)
        for parent, child in graph.edges:
            self.inputs[child].add(parent)

    def resolve(self) -> tuple[dict[UUID, frozenset[Framework]], set[UUID]]:
        """The allowed frameworks per pinned step uuid, and the pinned uuids."""
        pinned_class: dict[UUID, type[Implementation[Any]]] = {}
        seen: list[tuple[UUID, type[Implementation[Any]], str]] = []
        for uuid, node in self.graph.nodes.items():
            pin = node.feature.implementation_pin
            if pin is None:
                continue
            for step, impl in self._assign(uuid, pin):
                feature = self.graph.nodes[step].feature
                served = self._allowed(step, impl)
                for other, other_impl, other_request in seen:
                    twin = self.graph.nodes[other].feature
                    if (
                        twin.name == feature.name
                        and twin.options == feature.options
                        and self._allowed(other, other_impl) != served
                    ):
                        raise ImplementationPinError(
                            f"Step {feature.name} is pinned to {other_impl.__name__} by request {other_request} "
                            f"and to {impl.__name__} by request {node.feature.name}."
                        )
                seen.append((step, impl, str(node.feature.name)))
                pinned_class[step] = impl
        allowed = {uuid: self._allowed(uuid, impl) for uuid, impl in pinned_class.items()}
        return allowed, set(allowed)

    def _chain(self, uuid: UUID) -> set[UUID]:
        seen = {uuid}
        todo = [uuid]
        while todo:
            for parent in self.inputs.get(todo.pop(), ()):
                if parent not in seen:
                    seen.add(parent)
                    todo.append(parent)
        return seen

    def _ancestors(self, uuid: UUID) -> set[UUID]:
        return self._chain(uuid) - {uuid}

    def _lookup(self, name: str, request: str) -> type[Implementation[Any]]:
        matches = [c for c in self.loaded if name in (c.__name__, _qualified(c))]
        if not matches:
            raise ImplementationPinError(
                f"Step {request} pins implementation {name}, which is not a loaded Implementation."
            )
        if len(matches) > 1:
            raise ImplementationPinError(
                f"Step {request} pins implementation {name}, which names several loaded classes "
                f"({', '.join(sorted(_qualified(c) for c in matches))}); pin by module:qualname or by the class."
            )
        impl = matches[0]
        if impl not in self.accessible:
            raise ImplementationPinError(
                f"Step {request} pins implementation {impl.__name__}, which is not accessible "
                "(disabled, denied by policy, unregistered, or its compute framework is not enabled)."
            )
        return impl

    def _assign(self, uuid: UUID, pin: tuple[str, ...]) -> list[tuple[UUID, type[Implementation[Any]]]]:
        request = str(self.graph.nodes[uuid].feature.name)
        chain = self._chain(uuid)
        names_by_group: dict[type, list[type[Implementation[Any]]]] = defaultdict(list)
        for name in pin:
            impl = self._lookup(name, request)
            names_by_group[impl.feature_group].append(impl)
        assigned: list[tuple[UUID, type[Implementation[Any]]]] = []
        for group, impls in names_by_group.items():
            steps = [s for s in chain if issubclass(self.graph.nodes[s].feature_group_class, group)]
            if not steps:
                raise ImplementationPinError(
                    f"Step {request} pins implementation {impls[0].__name__}, which keys no feature group of its chain."
                )
            if len(impls) == 1:
                assigned.extend((s, impls[0]) for s in steps)
                continue
            if len(impls) != len(steps):
                raise ImplementationPinError(
                    f"Step {request} pins {len(impls)} implementations ({', '.join(i.__name__ for i in impls)}) "
                    f"for {group.__name__}, but its chain has {len(steps)} such steps."
                )
            ordered = sorted(steps, key=lambda s: len(self._ancestors(s) & set(steps)))
            for earlier, later in zip(ordered, ordered[1:]):
                if earlier not in self._ancestors(later):
                    raise ImplementationPinError(
                        f"Step {request} pins {', '.join(i.__name__ for i in impls)} for {group.__name__}, "
                        "but two of its steps are not ordered by input ancestry."
                    )
            assigned.extend(zip(ordered, impls))
        return assigned

    def _serving(self, group: type[FeatureGroup], framework: Framework) -> type[Implementation[Any]] | None:
        stop_after = "load_neutral" if issubclass(group, FormatFeatureGroup) else None
        return serving_implementation(group, framework, self.accessible, stop_after=stop_after)

    def _allowed(self, uuid: UUID, impl: type[Implementation[Any]]) -> frozenset[Framework]:
        node = self.graph.nodes[uuid]
        feature = node.feature
        group = node.feature_group_class
        serves = {
            fw
            for fw in feature.compute_frameworks or ()
            if (served := self._serving(group, fw)) is not None and issubclass(served, impl)
        }
        if not serves:
            reason = (
                "conflicts with the compute_framework of the feature"
                if feature.framework_pinned
                else "is served by no compute framework enabled for this run"
            )
            raise ImplementationPinError(f"Step {feature.name} pins implementation {impl.__name__}, which {reason}.")
        return frozenset(serves)
