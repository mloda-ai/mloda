"""
Data models for feature configuration schema.

This module defines the data models used to validate and parse
feature configuration files.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


def validate_feature_group_scope(value: Any) -> Optional[str]:
    """Config scope is a non-empty class-name string; the class-object form is Python-only."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "'feature_group' must be a feature group class-name string; "
            "the class-object form is only available in Python, not in a config"
        )
    if not value.strip():
        raise ValueError("'feature_group' must be a non-empty feature group class-name string")
    return value


def validate_feature_group_not_in_options(options: Optional[dict[str, Any]], container_name: str) -> None:
    """Scope is a config field, not an option: a top-level 'feature_group' key in a container is a misplacement."""
    if options and "feature_group" in options:
        raise ValueError(
            f"'feature_group' must not be a top-level key of '{container_name}'; "
            "move it next to 'name' as a feature config field"
        )


@dataclass
class FeatureConfig:
    """Model for a feature configuration with name and options."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)
    in_features: Optional[list[str]] = None
    group_options: Optional[dict[str, Any]] = None
    context_options: Optional[dict[str, Any]] = None
    propagate_context_keys: Optional[list[str]] = None
    column_index: Optional[int] = None
    feature_group: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate that options and group_options/context_options are mutually exclusive."""
        if self.options and (self.group_options or self.context_options):
            raise ValueError("Cannot use both 'options' and 'group_options'/'context_options'")
        if self.propagate_context_keys and not self.context_options:
            raise ValueError(
                "'propagate_context_keys' requires 'context_options'; "
                "it is meaningless without context options and would otherwise be silently dropped"
            )
        validate_feature_group_scope(self.feature_group)
        validate_feature_group_not_in_options(self.options, "options")
        validate_feature_group_not_in_options(self.group_options, "group_options")
        validate_feature_group_not_in_options(self.context_options, "context_options")


def feature_config_schema() -> dict[str, Any]:
    """Return JSON Schema for FeatureConfig model.

    Note: This provides a basic schema representation. For full JSON Schema
    support, consider using a dedicated schema generation library.
    """
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "options": {"type": "object", "default": {}},
            "in_features": {"type": "array", "items": {"type": "string"}},
            "group_options": {"type": "object"},
            "context_options": {"type": "object"},
            "propagate_context_keys": {"type": "array", "items": {"type": "string"}},
            "column_index": {"type": "integer"},
            "feature_group": {"type": "string", "minLength": 1},
        },
        "required": ["name"],
        "additionalProperties": False,
    }


FeatureConfigItem = str | FeatureConfig
