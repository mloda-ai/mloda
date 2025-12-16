from enum import Enum


class FilterType(Enum):
    min = "min"
    max = "max"
    equal = "equal"
    range = "range"
    regex = "regex"
    categorical_inclusion = "categorical_inclusion"
