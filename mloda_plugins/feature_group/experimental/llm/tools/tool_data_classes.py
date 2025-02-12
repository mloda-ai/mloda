from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class PytestResult:
    """Result of running pytest"""

    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class InputDataObject:
    name: str
    type: str
    description: str


@dataclass
class ToolFunctionDeclaration:
    name: str
    description: str
    parameters: list[InputDataObject]
    required: list[str]
    function: Callable[..., Any]
    tool_result: Callable[..., Any]
