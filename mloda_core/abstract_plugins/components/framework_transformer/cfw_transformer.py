from typing import Any, Dict, Tuple, Type

from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda_core.abstract_plugins.components.utils import get_all_subclasses


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
