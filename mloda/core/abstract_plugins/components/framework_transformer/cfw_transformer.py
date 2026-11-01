import logging
import threading
import weakref
from collections import deque
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.optional_dependency import loaded

logger = logging.getLogger(__name__)

_LAZY_TRANSFORMER_GATES = {
    "duckdb": "duckdb",
    "iceberg": "pyiceberg.table",
    "pandas": "pandas",
    "polars": "polars",
    "spark": "pyspark.sql",
}

_BUILTIN_PREFIX = "mloda_plugins.compute_framework.base_implementations."


def _rebuild_transformer(
    cls: type["ComputeFrameworkTransformer"], carried: tuple[type[BaseTransformer], ...]
) -> "ComputeFrameworkTransformer":
    """Carried classes are unpickled first, so their modules are imported and subclass discovery registers them."""
    return cls()


class ComputeFrameworkTransformer:
    """
    Manages transformations between different compute frameworks.

    This class maintains a registry of available transformers and provides
    methods to add new transformers. Heavy backends' transformer files load only once
    their library is imported (see ``_LAZY_TRANSFORMER_GATES``).

    The transformer registry is a mapping from framework pairs to transformer
    classes, allowing the system to find the appropriate transformer for
    converting data between two supported frameworks. A direction's edge is
    registered only if its transform hook is overridden.
    """

    _auto_load_done: ClassVar[bool] = False
    _lazy_dirs_loaded: ClassVar[set[str]] = set()
    _load_lock: ClassVar[threading.RLock] = threading.RLock()
    _warned_zero_edge: ClassVar["weakref.WeakSet[type[BaseTransformer]]"] = weakref.WeakSet()

    def __init__(self) -> None:
        """
        Initialize the ComputeFrameworkTransformer.

        Creates an empty transformer registry and populates it with all
        available BaseTransformer subclasses.
        """
        self.transformer_map: dict[tuple[type[Any], type[Any]], type[BaseTransformer]] = {}
        self._seen_lazy_dirs: frozenset[str] = frozenset()
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

        Loads the light transformer files and those of already imported backends,
        then registers every BaseTransformer subclass.
        """
        self._load_available()
        self._register_all()

    @classmethod
    def _load_available(cls, force: bool = False) -> None:
        """Load the light transformer files once, and each heavy backend's once its gate module is imported."""
        from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

        with cls._load_lock:
            loader = PluginLoader()
            if not cls._auto_load_done:
                loader.load_matching("compute_framework", "*transformer*", skip_dirs=_LAZY_TRANSFORMER_GATES)
                cls._auto_load_done = True
            for directory, gate in _LAZY_TRANSFORMER_GATES.items():
                if directory in cls._lazy_dirs_loaded:
                    continue
                if force or loaded(gate) is not None:
                    loader.load_matching("compute_framework", f"{directory}/*transformer*")
                    cls._lazy_dirs_loaded.add(directory)

    @classmethod
    def load_all_transformers(cls) -> None:
        """Load every backend's transformer files, whether or not its library is imported yet."""
        cls._load_available(force=True)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and deepcopy rebuild the registry from what the receiving process has loaded, not the parent's map."""
        carried = tuple(
            dict.fromkeys(t for t in self.transformer_map.values() if not t.__module__.startswith(_BUILTIN_PREFIX))
        )
        return (_rebuild_transformer, (type(self), carried))

    def _register_all(self) -> None:
        self._seen_lazy_dirs = frozenset(ComputeFrameworkTransformer._lazy_dirs_loaded)
        # Sorted so map insertion order (the BFS tie-break) is the same in every process.
        for transformer in sorted(get_all_subclasses(BaseTransformer), key=lambda c: (c.__module__, c.__qualname__)):
            self.add(transformer)

    def _refresh(self) -> bool:
        """Pick up transformer files whose backend was imported since the last load; True if any were added."""
        self._load_available()
        if frozenset(ComputeFrameworkTransformer._lazy_dirs_loaded) == self._seen_lazy_dirs:
            return False
        self._register_all()
        return True

    def direct_transformer(self, source: type[Any], target: type[Any]) -> type[BaseTransformer] | None:
        """Direct transformer for a pair, refreshing the registry once on a miss."""
        found = self.transformer_map.get((source, target))
        if found is None and self._refresh():
            found = self.transformer_map.get((source, target))
        return found

    def get_transformation_chain(
        self, from_framework: type[Any], to_framework: type[Any]
    ) -> list[type[BaseTransformer]] | None:
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

        if self._refresh():
            return self.get_transformation_chain(from_framework, to_framework)
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
