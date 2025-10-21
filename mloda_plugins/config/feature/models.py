"""
Data models for feature configuration schema.

This module defines the data models used to validate and parse
feature configuration files.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FeatureConfig:
    """Model for a feature configuration with name and options."""

    name: str
    options: Dict[str, Any] = field(default_factory=dict)
    mloda_sources: Optional[List[str]] = None
    group_options: Optional[Dict[str, Any]] = None
    context_options: Optional[Dict[str, Any]] = None
    column_index: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate that options and group_options/context_options are mutually exclusive."""
        if self.options and (self.group_options or self.context_options):
            raise ValueError("Cannot use both 'options' and 'group_options'/'context_options'")


def feature_config_schema() -> Dict[str, Any]:
    """Return JSON Schema for FeatureConfig model.

    Note: This provides a basic schema representation. For full JSON Schema
    support, consider using a dedicated schema generation library.
    """
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "options": {"type": "object", "default": {}},
            "mloda_sources": {"type": "array", "items": {"type": "string"}},
            "group_options": {"type": "object"},
            "context_options": {"type": "object"},
            "column_index": {"type": "integer"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }


FeatureConfigItem = Union[str, FeatureConfig]
