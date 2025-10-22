from typing import Any, Dict, List, Optional, Tuple, Type

from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda_core.abstract_plugins.components.utils import get_all_subclasses

try:
    import pyarrow as pa
except ImportError:
    pa = None


class ComputeFrameworkTransformer:
    """
    Manages transformations between different compute frameworks.

    This class maintains a registry of available transformers and provides
    methods to add new transformers. It automatically discovers and registers
    all BaseTransformer subclasses during initialization.

    The transformer registry is a mapping from framework pairs to transformer
    classes, allowing the system to find the appropriate transformer for
    converting data between any two supported frameworks.
    """

    def __init__(self) -> None:
        """
        Initialize the ComputeFrameworkTransformer.

        Creates an empty transformer registry and populates it with all
        available BaseTransformer subclasses.
        """
        self.transformer_map: Dict[Tuple[Type[Any], Type[Any]], Type[BaseTransformer]] = {}
        self.initilize_transformer()

    def add(self, transformer: Type[BaseTransformer]) -> bool:
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

        # Check already added cases
        if (left, right) in self.transformer_map:
            if transformer == self.transformer_map[(left, right)]:
                return True
            raise ValueError(
                f"Transformer {transformer} is already registered for the pair ({left}, {right}), but with a different implementation."
            )

        self.transformer_map[(left, right)] = transformer
        self.transformer_map[(right, left)] = transformer
        return True

    def initilize_transformer(self) -> None:
        """
        Initialize the transformer registry with all available transformers.

        This method discovers all BaseTransformer subclasses and adds them
        to the registry.
        """
        transformers = get_all_subclasses(BaseTransformer)

        for transformer in transformers:
            self.add(transformer)

    def get_transformation_chain(
        self, from_framework: Type[Any], to_framework: Type[Any]
    ) -> Optional[List[Type[BaseTransformer]]]:
        """
        Find a transformation chain between two frameworks.

        If a direct transformer exists, returns a single-element list.
        If no direct transformer exists, tries to find a path through PyArrow
        as an intermediate format.

        Args:
            from_framework: Source framework type
            to_framework: Target framework type

        Returns:
            List of transformers to apply in sequence, or None if no path exists
        """
        # Try direct transformation first
        if (from_framework, to_framework) in self.transformer_map:
            return [self.transformer_map[(from_framework, to_framework)]]

        # If no direct path and PyArrow is available, try chaining through PyArrow
        if pa is not None:
            pa_table_type = pa.Table
            # Check if we can go: from_framework → PyArrow → to_framework
            if (from_framework, pa_table_type) in self.transformer_map and (
                pa_table_type,
                to_framework,
            ) in self.transformer_map:
                return [
                    self.transformer_map[(from_framework, pa_table_type)],
                    self.transformer_map[(pa_table_type, to_framework)],
                ]

        return None
