"""
Data models for feature configuration schema.

This module defines the data models used to validate and parse
feature configuration files.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Optional


def validate_feature_group_scope(value: Any) -> Optional[str]:
    """Config scope is a non-empty class-name string; the class-object form is Python-only.

    The string semantics (root FeatureGroup base-name rejection, strip-to-nothing) delegate to the
    canonical ``normalize_feature_group_scope``; only the config-specific concerns stay local: configs
    accept strings only, and every rejection is a ValueError in the config validation style.
    """
    # Local import: feature pulls in the plugin machinery, which this data-model module stays free of.
    from mloda.core.abstract_plugins.components.feature import normalize_feature_group_scope

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "'feature_group' must be a feature group class-name string; "
            "the class-object form is only available in Python, not in a config"
        )
    # Boundary conversion: for a string, the canonical helper raises TypeError only for the root base name.
    try:
        normalized = normalize_feature_group_scope(value)
    except TypeError as exc:
        raise ValueError(
            "'feature_group' cannot be the root FeatureGroup base class name; it names no feature group family, "
            "so a concrete subclass name is required"
        ) from exc
    # The canonical helper strips an empty string to None; a config must reject it instead of dropping the scope.
    if normalized is None:
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

    # metadata={"nested": True} marks the fields a nested in_features dict supports; the rest are top-level only.
    name: str = field(metadata={"nested": True})
    options: dict[str, Any] = field(default_factory=dict, metadata={"nested": True})
    # A nested in_features dict reuses this field for a single source name or a further nested feature dict.
    in_features: Optional[list[str] | dict[str, Any] | str] = field(default=None, metadata={"nested": True})
    group_options: Optional[dict[str, Any]] = None
    context_options: Optional[dict[str, Any]] = None
    propagate_context_keys: Optional[list[str]] = None
    column_index: Optional[int] = None
    feature_group: Optional[str] = field(default=None, metadata={"nested": True})

    def __post_init__(self) -> None:
        """Validate the invariants shared by the top-level and the nested feature dict."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"'name' must be a non-empty string, got: {self.name!r}")
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


FEATURE_CONFIG_FIELDS = frozenset(f.name for f in fields(FeatureConfig))
NESTED_FIELDS = frozenset(f.name for f in fields(FeatureConfig) if f.metadata.get("nested"))


def describe_offender(value: Any) -> str:
    """Name a rejected dict by its keys: its repr would dump an options payload that may hold credentials."""
    if isinstance(value, dict):
        return f"a dict with keys {sorted(value)}"
    return repr(value)


def validate_top_level_in_features(value: Any) -> None:
    """Top-level in_features is an array of source feature names.

    Not a __post_init__ invariant: the nested in_features dict legitimately carries a string or a dict here.
    """
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(
            f"'in_features' must be an array of source feature names at the top level of a feature config, got: "
            f"{describe_offender(value)}"
        )
    for element in value:
        if not isinstance(element, str) or not element.strip():
            raise ValueError(
                f"'in_features' must be an array of non-empty source feature names at the top level of a feature "
                f"config, got element: {describe_offender(element)}"
            )


def validate_nested_in_features(value: Any) -> None:
    """A nested in_features value is a source name, a list of source names, or a further nested feature dict."""
    if value is None:
        return
    if isinstance(value, dict):
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"'in_features' must be a non-empty source feature name, got: {value!r}")
        return
    if isinstance(value, list):
        if not value:
            raise ValueError("'in_features' must not be an empty list; it declares the source features")
        for element in value:
            if isinstance(element, dict):
                raise ValueError(
                    "'in_features' as a list holds source feature names only; a nested feature dict is supported "
                    f"as the direct value of 'in_features', not as a list element, got: {describe_offender(element)}"
                )
            if not isinstance(element, str) or not element.strip():
                raise ValueError(
                    f"'in_features' as a list holds non-empty source feature names, got element: "
                    f"{describe_offender(element)}"
                )
        return
    raise ValueError(
        f"'in_features' must be a source feature name, a list of source feature names, or a nested feature dict, "
        f"got: {describe_offender(value)}"
    )


def parse_nested_feature_config(item: dict[str, Any]) -> FeatureConfig:
    """Validate a nested in_features dict through FeatureConfig, rejecting unknown and top-level-only keys."""
    if "name" not in item:
        raise ValueError(f"Nested in_features must have a 'name' field, keys present: {sorted(item)}")

    # An unknown key is a typo, not a misplacement: let FeatureConfig raise the native TypeError naming it,
    # before the top-level-only check can misreport it. Both checks precede __post_init__, whose invariants
    # are top-level semantics and would otherwise give misleading advice for a nested dict.
    if not FEATURE_CONFIG_FIELDS.issuperset(item):
        FeatureConfig(**item)

    top_level_only = sorted(key for key in item if key not in NESTED_FIELDS)
    if top_level_only:
        raise ValueError(
            f"{top_level_only} is only valid at the top level of a feature config; "
            "a nested 'in_features' dict supports 'name', 'options', 'in_features' and 'feature_group' only"
        )
    return FeatureConfig(**item)


def feature_config_schema() -> dict[str, Any]:
    """Return JSON Schema for FeatureConfig model.

    FeatureConfig is the enforcement point at both levels; the schema itself is never run by a validator.
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
