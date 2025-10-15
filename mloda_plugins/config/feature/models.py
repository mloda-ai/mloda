"""
Pydantic models for feature configuration schema.

This module defines the data models used to validate and parse
feature configuration files.
"""

from typing import Any, Dict, Union
from pydantic import BaseModel


class FeatureConfig(BaseModel):
    """Model for a feature configuration with name and options."""

    name: str
    options: Dict[str, Any] = {}


def feature_config_schema() -> Dict[str, Any]:
    """Return JSON Schema for FeatureConfig model."""
    return FeatureConfig.model_json_schema()


FeatureConfigItem = Union[str, FeatureConfig]
