from enum import Enum
from typing import Any


class DefaultOptionKeys(str, Enum):
    """
    Default option keys used to configure mloda feature groups.

    These keys are used to look up configuration values in Options objects.
    The enum value serves as both the option key and the default column name.

    Time-Related Keys:
    - `reference_time`: Key for the event timestamp column. Value: "reference_time"
    - `time_travel`: Key for the validity timestamp column. Value: "time_travel"

    Data Shaping Keys:
    - `group`: Key for grouping/partitioning columns. Value: "group"
    - `order_by`: Key for sort-order columns used by sequential operations
      (rank, offset, frame_aggregate). Value: "order_by"

    These values are used as default column names when not customized via Options.
    """

    in_features = "in_features"
    reference_time = "reference_time"
    time_travel = "time_travel"
    allowed_values = "allowed_values"
    default = "default"
    context = "context"
    group = "group"
    order_by = "order_by"
    strict_validation = "strict_validation"
    element_validator = "element_validator"
    strict_type_enforcement = "strict_type_enforcement"
    required_when = "required_when"
    match_guard = "match_guard"

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)


# The SCHEMA of a PROPERTY_MAPPING spec dict: the complete set of permitted keys. A spec's
# value space is declared under ``allowed_values``, never inferred, so every other key must be
# one of these. ``FeatureChainParser.validate_property_mapping_defaults`` rejects any unknown
# key at class-definition time, which is what stops a typo'd flag from being read as a value.
PROPERTY_SPEC_KEYS: frozenset[Any] = frozenset(
    {
        "explanation",
        DefaultOptionKeys.allowed_values,
        DefaultOptionKeys.default,
        DefaultOptionKeys.context,
        DefaultOptionKeys.group,
        DefaultOptionKeys.strict_validation,
        DefaultOptionKeys.element_validator,
        DefaultOptionKeys.required_when,
        DefaultOptionKeys.match_guard,
    }
)

# Removed PROPERTY_MAPPING keys mapped to their replacement (issue #600). They are unknown keys
# like any other; this map only gives the unknown-key error a precise remedy to name.
REMOVED_PROPERTY_KEYS: dict[str, DefaultOptionKeys] = {
    "validation_function": DefaultOptionKeys.element_validator,
    "type_validator": DefaultOptionKeys.match_guard,
}
