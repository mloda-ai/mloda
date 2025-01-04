from abc import ABC
from typing import Any

from mloda_core.filter.single_filter import SingleFilter


class BaseFilterEngine(ABC):
    @classmethod
    def final_filters(cls) -> bool:
        """This function should return True if the filters should be applied after the feature calculation."""
        return False

    @classmethod
    def apply_filters(cls, data: Any, features: Any) -> Any:
        return cls.apply_single_filters(data, features)

    @classmethod
    def apply_single_filters(cls, data: Any, features: Any) -> Any:
        """This function should be used to apply filters to the data if filters are applied one by one."""
        if features.filters is None:
            return data

        for single_filter in features.filters:
            if single_filter.filter_feature.name.name not in features.get_all_names():
                continue
            data = cls.do_filter(data, single_filter)

        return data

    @classmethod
    def do_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        if filter_feature.filter_type is None:
            raise ValueError(f"Filter type evaluates to None {filter_feature.filter_feature.name.name}.")

        if filter_feature.filter_type == "range":
            return cls.do_range_filter(data, filter_feature)
        elif filter_feature.filter_type == "min":
            return cls.do_min_filter(data, filter_feature)
        elif filter_feature.filter_type == "max":
            return cls.do_max_filter(data, filter_feature)
        elif filter_feature.filter_type == "equal":
            return cls.do_equal_filter(data, filter_feature)
        elif filter_feature.filter_type == "regex":
            return cls.do_regex_filter(data, filter_feature)
        elif filter_feature.filter_type == "categorical_inclusion":
            return cls.do_categorical_inclusion_filter(data, filter_feature)
        else:
            return cls.do_custom_filter(data, filter_feature)

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def do_custom_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        raise NotImplementedError

    @classmethod
    def get_min_max_operator(cls, filter_feature: SingleFilter) -> Any:
        """Convenience method to get min, max, and max operator from filter parameters"""

        min_parameter, max_parameter, is_max_exclusive = None, None, False

        for parameter in filter_feature.parameter:
            if parameter[0] == "min":
                min_parameter = parameter[1]
            elif parameter[0] == "max":
                max_parameter = parameter[1]
            elif parameter[0] == "max_exclusive":
                if parameter[1] is True:
                    is_max_exclusive = True
            else:
                raise ValueError(f"Filter parameter {parameter[0]} not supported")

        return min_parameter, max_parameter, is_max_exclusive
