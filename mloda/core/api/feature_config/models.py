"""
Data models for feature configuration schema.

This module defines the data models used to validate and parse
feature configuration files.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

FEATURE_GROUP_SCOPE_KEY = "feature_group"


def validate_feature_group_scope(feature_group: Any) -> None:
    """Config is string-only: JSON cannot carry a class object, so the class-object scope stays Python-only.

    Shared by FeatureConfig and the nested in_features path so both raise the same config-style ValueError.
    """
    if feature_group is not None and not isinstance(feature_group, str):
        raise ValueError(
            f"'{FEATURE_GROUP_SCOPE_KEY}' must be a class-name string or None, got {type(feature_group).__name__}"
        )


def reject_misplaced_feature_group_scope(container: Optional[dict[str, Any]], container_name: str) -> None:
    """Reject the scope written as a top-level key of an option container instead of the top-level field.

    Only top-level keys are checked: a nested scope legitimately lives inside an
    ``in_features`` dict value, which is not a key of the container.
    """
    if container and FEATURE_GROUP_SCOPE_KEY in container:
        raise ValueError(
            f"'{FEATURE_GROUP_SCOPE_KEY}' must be a top-level field of the feature config, "
            f"not a key of '{container_name}'. Move it next to 'name'."
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
        """Validate option container exclusivity, propagate_context_keys, and the feature_group scope."""
        if self.options and (self.group_options or self.context_options):
            raise ValueError("Cannot use both 'options' and 'group_options'/'context_options'")
        if self.propagate_context_keys and not self.context_options:
            raise ValueError(
                "'propagate_context_keys' requires 'context_options'; "
                "it is meaningless without context options and would otherwise be silently dropped"
            )
        validate_feature_group_scope(self.feature_group)
        reject_misplaced_feature_group_scope(self.options, "options")
        reject_misplaced_feature_group_scope(self.group_options, "group_options")
        reject_misplaced_feature_group_scope(self.context_options, "context_options")


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
            "feature_group": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }


FeatureConfigItem = str | FeatureConfig
