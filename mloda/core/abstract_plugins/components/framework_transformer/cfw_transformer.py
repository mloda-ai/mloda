import logging
import weakref
from collections import deque
from typing import Any, ClassVar, Optional

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.utils import get_all_subclasses

logger = logging.getLogger(__name__)


class ComputeFrameworkTransformer:
    """
    Manages transformations between different compute frameworks.

    This class maintains a registry of available transformers and provides
    methods to add new transformers. It auto-loads transformer files once per
    process, gated on ``_auto_load_done`` and the disabled-groups set.

    To suppress auto-loading of transformer files:
        PluginLoader.disable_auto_load("compute_framework")

    Note: "compute_framework" is the group key checked by initilize_transformer.
    Disabling it prevents any transformer files from being auto-loaded.

    The transformer registry is a mapping from framework pairs to transformer
    classes, allowing the system to find the appropriate transformer for
    converting data between two supported frameworks. A direction's edge is
    registered only if its transform hook is overridden.
    """

    _auto_load_done: ClassVar[bool] = False
    _warned_zero_edge: ClassVar["weakref.WeakSet[type[BaseTransformer]]"] = weakref.WeakSet()

    def __init__(self) -> None:
        """
        Initialize the ComputeFrameworkTransformer.

        Creates an empty transformer registry and populates it with all
        available BaseTransformer subclasses.
        """
        self.transformer_map: dict[tuple[type[Any], type[Any]], type[BaseTransformer]] = {}
        self.initilize_transformer()

    def add(self, transformer: type[BaseTransformer]) -> bool:
        """
        Add a transformer to the registry.

        This method registers a transformer for converting between two frameworks.
        It checks if the required imports are available and if there are any
        conflicts with existing transformers.

        Args:
            transformer: The transformer class to register

        Returns:
            bool: True if the transformer was successfully added, False otherwise

        Raises:
            ValueError: If a different transformer is already registered for the same framework pair
        """
        if not transformer.check_imports():
            return False

        left = transformer.framework()
        right = transformer.other_framework()

        # Each edge is registered only when its transform direction is actually overridden,
        # so BFS never routes through an unimplemented direction.
        forward_overridden = self._underlying(transformer.transform_fw_to_other_fw) is not self._underlying(
            BaseTransformer.transform_fw_to_other_fw
        )
        reverse_overridden = self._underlying(transformer.transform_other_fw_to_fw) is not self._underlying(
            BaseTransformer.transform_other_fw_to_fw
        )

        registered = False
        if forward_overridden:
            self._register_edge(left, right, transformer)
            registered = True
        if reverse_overridden:
            self._register_edge(right, left, transformer)
            registered = True
        if not registered and transformer not in ComputeFrameworkTransformer._warned_zero_edge:
            ComputeFrameworkTransformer._warned_zero_edge.add(transformer)
            logger.warning(
                f"Transformer {transformer.__name__} overrides neither transform direction; no edges were registered."
            )
        return registered

    def _register_edge(self, source: type[Any], target: type[Any], transformer: type[BaseTransformer]) -> None:
        """Write one directional edge, rejecting a collision with a different incumbent.

        The incumbent is only compared by equality (never introspected), so a non-transformer
        sentinel occupying the pair is handled without calling methods on it.
        """
        incumbent = self.transformer_map.get((source, target))
        if incumbent is not None and incumbent != transformer:
            raise ValueError(
                f"Transformer {transformer} is already registered for the pair ({source}, {target}), but with a different implementation."
            )
        self.transformer_map[(source, target)] = transformer

    @staticmethod
    def _underlying(member: Any) -> Any:
        """Underlying function of a classmethod/staticmethod/plain override, for identity comparison."""
        return getattr(member, "__func__", member)

    def initilize_transformer(self) -> None:
        """
        Initialize the transformer registry with all available transformers.

        This method discovers all BaseTransformer subclasses and adds them
        to the registry. The transformer files from the compute_framework group
        are auto-loaded once per process via load_matching, so a stray
        module-scope import of a single transformer cannot leave the registry
        partial. Auto-loading is skipped while the group is disabled.
        """
        from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

        if not ComputeFrameworkTransformer._auto_load_done and "compute_framework" not in PluginLoader._disabled_groups:
            PluginLoader().load_matching("compute_framework", "*transformer*")
            ComputeFrameworkTransformer._auto_load_done = True

        transformers = get_all_subclasses(BaseTransformer)
        for transformer in transformers:
            self.add(transformer)

    def get_transformation_chain(
        self, from_framework: type[Any], to_framework: type[Any]
    ) -> Optional[list[type[BaseTransformer]]]:
        """
        Find a transformation chain between two frameworks.

        If a direct transformer exists, returns a single-element list. Otherwise
        runs a breadth-first shortest-path search over ALL registered edges in
        transformer_map, so any registered framework (pa.Table is just a regular
        node) can serve as an intermediate hop. Equal-length paths tie-break on
        first-registered edge order (map insertion order), making the result
        deterministic. The map is finite and each framework is visited at most
        once, so the search always terminates.

        Args:
            from_framework: Source framework type
            to_framework: Target framework type

        Returns:
            List of transformers to apply in sequence, or None if no path exists
        """
        # Direct edge always wins
        if (from_framework, to_framework) in self.transformer_map:
            return [self.transformer_map[(from_framework, to_framework)]]

        # BFS over registered edges, expanding neighbors in map insertion order
        queue: deque[type[Any]] = deque([from_framework])
        visited: set[type[Any]] = {from_framework}
        parent: dict[type[Any], tuple[type[Any], type[BaseTransformer]]] = {}

        while queue:
            current = queue.popleft()
            for (src, dst), transformer in self.transformer_map.items():
                if src != current or dst in visited:
                    continue
                visited.add(dst)
                parent[dst] = (current, transformer)
                if dst == to_framework:
                    chain: list[type[BaseTransformer]] = []
                    node: type[Any] = to_framework
                    while node != from_framework:
                        node, edge_transformer = parent[node]
                        chain.append(edge_transformer)
                    chain.reverse()
                    return chain
                queue.append(dst)

        return None

    def apply_chain(
        self,
        from_framework: type[Any],
        to_framework: type[Any],
        chain: list[type[BaseTransformer]],
        data: Any,
        connection: Any,
    ) -> Any:
        """Walk a transformation chain, resolving each intermediate hop's target framework
        from the transformer map, and apply each transformer in sequence."""
        current_fw = from_framework
        for i, transformer_cls in enumerate(chain):
            if i == len(chain) - 1:
                target_fw: type[Any] = to_framework
            else:
                target_fw = self._resolve_intermediate_target(transformer_cls, current_fw)

            data = transformer_cls.transform(current_fw, target_fw, data, connection)
            current_fw = target_fw

        return data

    def _resolve_intermediate_target(self, transformer_cls: type[BaseTransformer], current_fw: type[Any]) -> type[Any]:
        for (src, dst), trans in self.transformer_map.items():
            if trans == transformer_cls and src == current_fw:
                return dst
        raise KeyError(
            f"No transformer edge found for {transformer_cls} from {current_fw} in the "
            "transformer map; the transformation chain is inconsistent with the registry."
        )
