from enum import Enum


class FilterType(Enum):
    MIN = "min"
    MAX = "max"
    EQUAL = "equal"
    RANGE = "range"
    REGEX = "regex"
    CATEGORICAL_INCLUSION = "categorical_inclusion"
