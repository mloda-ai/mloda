from enum import Enum


class DefaultOptionKeys(str, Enum):
    """

    This class contains the default option keys for mloda.

    These keys are used to set options for the features as conventions.

    For faster development and prototyping, it was decided to use the Option object to move configurations around.
    When the framework matured and we learned more about the requirements, we can refactor this to a more sophisticated solution.

    However we use the DefaultOptions object to store needed keywords.

    """

    mloda_source_feature = "mloda_source_feature"

    @classmethod
    def list(cls) -> list[str]:
        return [member.value for member in cls]
