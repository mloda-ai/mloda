"""
Test data converter using the existing ComputeFrameworkTransformer system.

This module provides utilities to convert test data (List[Dict[str, Any]]) to and from
any compute framework format by leveraging the existing transformer infrastructure.
All conversions go through PyArrow as an intermediate format.
"""

from typing import Any, Dict, List, Optional, Type

from mloda_core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

try:
    import pyarrow as pa
except ImportError:
    pa = None


# Load plugins once at module import time
PluginLoader().load_group("compute_framework")


class DataConverter:
    """
    Converts test data between List[Dict] format and any framework format.

    This class leverages the existing ComputeFrameworkTransformer system to automatically
    support all frameworks that have PyArrow transformers. New frameworks are automatically
    supported when they implement a PyArrow transformer.

    All conversions follow the path:
    - To framework: List[Dict] → PyArrow → Target Framework
    - From framework: Source Framework → PyArrow → List[Dict]
    """

    def __init__(self) -> None:
        """Initialize the converter with the transformer registry."""
        # Initialize transformer registry (will auto-discover all loaded transformer subclasses)
        self.transformer = ComputeFrameworkTransformer()

    def to_framework(
        self,
        data: List[Dict[str, Any]],
        target_framework_type: Type[Any],
        connection: Optional[Any] = None,
    ) -> Any:
        """
        Convert test data (List[Dict]) to target framework format.

        Args:
            data: Test data in List[Dict[str, Any]] format
            target_framework_type: The target framework type (e.g., pd.DataFrame, pa.Table)
            connection: Optional framework connection object (e.g., for DuckDB)

        Returns:
            Any: Data in the target framework's native format

        Raises:
            KeyError: If no transformer exists for the target framework
            ImportError: If required framework is not installed
        """
        if pa is None:
            raise ImportError("PyArrow is required for test data conversion")

        # Special case: if target is already list, no conversion needed
        if target_framework_type == list:
            return data

        # Step 1: List[Dict] → PyArrow
        try:
            transformer_list_to_arrow = self.transformer.transformer_map[(list, pa.Table)]
            arrow_data = transformer_list_to_arrow.transform(list, pa.Table, data, None)
        except KeyError:
            raise KeyError(
                f"No transformer found for list → PyArrow. Ensure PythonDictPyarrowTransformer is available."
            )

        # Special case: if target is PyArrow, we're done
        if target_framework_type == pa.Table:
            return arrow_data

        # Step 2: PyArrow → Target Framework
        try:
            transformer_arrow_to_target = self.transformer.transformer_map[(pa.Table, target_framework_type)]
            return transformer_arrow_to_target.transform(pa.Table, target_framework_type, arrow_data, connection)
        except KeyError:
            raise KeyError(
                f"No transformer found for PyArrow → {target_framework_type}. "
                f"Ensure the framework has a PyArrow transformer."
            )

    def from_framework(self, data: Any, source_framework_type: Type[Any]) -> List[Dict[str, Any]]:
        """
        Convert framework data back to test data (List[Dict]) format.

        Args:
            data: Data in the source framework's native format
            source_framework_type: The source framework type (e.g., pd.DataFrame)

        Returns:
            List[Dict[str, Any]]: Data in test format

        Raises:
            KeyError: If no transformer exists for the source framework
            ImportError: If required framework is not installed
        """
        if pa is None:
            raise ImportError("PyArrow is required for test data conversion")

        # Special case: if source is already list, no conversion needed
        if source_framework_type == list:
            assert isinstance(data, list), "Expected list data"
            return data

        # Special case: if source is PyArrow, skip the first step
        if source_framework_type == pa.Table:
            arrow_data = data
        else:
            # Step 1: Source Framework → PyArrow
            try:
                transformer_source_to_arrow = self.transformer.transformer_map[(source_framework_type, pa.Table)]
                arrow_data = transformer_source_to_arrow.transform(source_framework_type, pa.Table, data, None)
            except KeyError:
                raise KeyError(
                    f"No transformer found for {source_framework_type} → PyArrow. "
                    f"Ensure the framework has a PyArrow transformer."
                )

        # Step 2: PyArrow → List[Dict]
        try:
            transformer_arrow_to_list = self.transformer.transformer_map[(pa.Table, list)]
            result: List[Dict[str, Any]] = transformer_arrow_to_list.transform(pa.Table, list, arrow_data, None)
            return result
        except KeyError:
            raise KeyError(
                f"No transformer found for PyArrow → list. Ensure PythonDictPyarrowTransformer is available."
            )
