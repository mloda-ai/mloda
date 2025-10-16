"""
Pydantic models for feature configuration schema.

This module defines the data models used to validate and parse
feature configuration files.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, model_validator


class FeatureConfig(BaseModel):
    """Model for a feature configuration with name and options."""

    name: str
    options: Dict[str, Any] = {}
    mloda_source: Optional[str] = None
    mloda_sources: Optional[List[str]] = None
    group_options: Optional[Dict[str, Any]] = None
    context_options: Optional[Dict[str, Any]] = None
    column_index: Optional[int] = None

    @model_validator(mode="after")
    def validate_options_mutual_exclusion(self) -> "FeatureConfig":
        """Validate that options and group_options/context_options are mutually exclusive."""
        if self.options and (self.group_options or self.context_options):
            raise ValueError("Cannot use both 'options' and 'group_options'/'context_options'")
        return self

    @model_validator(mode="after")
    def validate_source_mutual_exclusion(self) -> "FeatureConfig":
        """Validate that mloda_source and mloda_sources are mutually exclusive."""
        if self.mloda_source and self.mloda_sources:
            raise ValueError("Cannot use both 'mloda_source' and 'mloda_sources'")
        return self


def feature_config_schema() -> Dict[str, Any]:
    """Return JSON Schema for FeatureConfig model."""
    return FeatureConfig.model_json_schema()


FeatureConfigItem = Union[str, FeatureConfig]
