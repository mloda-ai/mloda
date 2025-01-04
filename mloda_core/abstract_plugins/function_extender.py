from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Set


class WrapperFunctionEnum(Enum):
    FEATURE_GROUP_CALCULATE_FEATURE = "feature_group_calculate_feature"
    VALIDATE_INPUT_FEATURE = "validate_input_feature"
    VALIDATE_OUTPUT_FEATURE = "validate_output_feature"


class WrapperFunctionExtender(ABC):
    """
    - Automated Metadata harvestor connector
    - Messaging Integration ( email )
    - Automation Tools
    - data lineage mapping
    - Impact Analysis
    - Audit Trail
    - Monitoring alerts
    - metadata capture
    - Event logging
    - metrics on feature calculation
    - visibility / observibility
    - Performance
    """

    @abstractmethod
    def wraps(self) -> Set[WrapperFunctionEnum]:
        pass

    @abstractmethod
    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        pass
