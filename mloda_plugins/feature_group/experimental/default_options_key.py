from enum import Enum


class DefaultOptionKeys(str, Enum):
    """

    This class contains the default option keys for mloda.

    These keys are used to set options for the features as conventions.

    For faster development and prototyping, it was decided to use the Option object to move configurations around.
    When the framework matured and we learned more about the requirements, we can refactor this to a more sophisticated solution.

    However we use the DefaultOptions object to store needed keywords.

    """

    in_features = "in_features"
    mloda_feature_chainer_parser_key = "mloda_feature_chainer_parser_key"
    reference_time = "reference_time"
    mloda_default = "default"
    mloda_context = "context"
    mloda_group = "group"
    mloda_strict_validation = "strict_validation"
    mloda_validation_function = "validation_function"
    mloda_strict_type_enforcement = "strict_type_enforcement"
