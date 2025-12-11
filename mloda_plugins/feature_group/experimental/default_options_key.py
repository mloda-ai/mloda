from enum import Enum


class DefaultOptionKeys(str, Enum):
    """
    Default option keys used to configure mloda feature groups.

    These keys are used to look up configuration values in Options objects.
    The enum value serves as both the option key and the default column name.

    Time-Related Keys:
    - `reference_time`: Key for the event timestamp column. Value: "reference_time"
    - `time_travel`: Key for the validity timestamp column. Value: "time_travel_filter"

    These values are used as default column names when not customized via Options.
    """

    in_features = "in_features"
    mloda_feature_chainer_parser_key = "mloda_feature_chainer_parser_key"
    reference_time = "reference_time"
    time_travel = "time_travel_filter"
    mloda_default = "default"
    mloda_context = "context"
    mloda_group = "group"
    mloda_strict_validation = "strict_validation"
    mloda_validation_function = "validation_function"
