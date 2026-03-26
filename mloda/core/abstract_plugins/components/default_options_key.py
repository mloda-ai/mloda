from enum import Enum


class DefaultOptionKeys(str, Enum):
    """
    Default option keys used to configure mloda feature groups.

    These keys are used to look up configuration values in Options objects.
    The enum value serves as both the option key and the default column name.

    Time-Related Keys:
    - `reference_time`: Key for the event timestamp column. Value: "reference_time"
    - `time_travel`: Key for the validity timestamp column. Value: "time_travel_filter"

    Data Shaping Keys:
    - `group`: Key for grouping/partitioning columns. Value: "group"
    - `order_by`: Key for sort-order columns used by sequential operations
      (rank, offset, frame_aggregate). Value: "order_by"

    These values are used as default column names when not customized via Options.
    """

    in_features = "in_features"
    feature_chainer_parser_key = "feature_chainer_parser_key"
    reference_time = "reference_time"
    time_travel = "time_travel_filter"
    default = "default"
    context = "context"
    group = "group"
    order_by = "order_by"
    strict_validation = "strict_validation"
    validation_function = "validation_function"
    strict_type_enforcement = "strict_type_enforcement"
