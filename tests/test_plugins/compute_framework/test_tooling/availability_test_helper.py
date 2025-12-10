"""Shared helper for testing compute framework availability."""

from typing import Any, Type
from unittest.mock import patch


def assert_unavailable_when_import_blocked(
    framework_class: Type[Any],
    modules_to_block: list[str],
) -> None:
    """Assert that framework.is_available() returns False when imports are blocked.

    Args:
        framework_class: The framework class to test (must have is_available() static method)
        modules_to_block: List of module names that should raise ImportError
    """

    def side_effect(name: str, *args: Any, **kwargs: Any) -> Any:
        if any(name == module or name.startswith(f"{module}.") for module in modules_to_block):
            raise ImportError(f"No module named '{name}'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=side_effect):
        assert framework_class.is_available() is False
