from typing import Dict
import logging

from mloda_core.abstract_plugins.components.utils import get_all_subclasses
from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import ToolFunctionDeclaration


logger = logging.getLogger(__name__)


class ToolCollection:
    def __init__(self) -> None:
        self.data: Dict[str, ToolFunctionDeclaration] = {}
        subclasses = get_all_subclasses(BaseTool)

        self.tool_mappings = {}

        for sub in subclasses:
            self.tool_mappings[sub.__name__] = sub

    def add_tool(self, tool_name: str) -> None:
        found_mapping = self.tool_mappings.get(tool_name)

        if found_mapping is None:
            raise ValueError(f"Tool {tool_name} not found in tool mappings.")

        self.data[tool_name] = found_mapping.tool_declaration()

    def get_tool(self, tool_name: str) -> ToolFunctionDeclaration:
        return self.data[tool_name]

    def get_all_tools(self) -> Dict[str, ToolFunctionDeclaration]:
        return self.data

    def __str__(self) -> str:
        return str(self.data)
