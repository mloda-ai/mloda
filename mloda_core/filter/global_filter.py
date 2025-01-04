from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.domain import Domain
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_core.filter.single_filter import SingleFilter


import logging

logger = logging.getLogger(__name__)


class GlobalFilter:
    def __init__(self) -> None:
        """
        This constructor sets up two main attributes:
        1. `filters`: A set to store individual filter objects (`SingleFilter`). Each filter represents a condition
           used to restrict or sort data based on specific features.
        2. `collection`: A dictionary mapping a tuple of feature group types and feature names to a set of filter feature
           names and the uuid to the used single filter. This is used to track which features are associated with which filters for a specific feature group.
           This can be used to check after the fact if a feature is a filter feature for a specific feature group
           e.g. for debugging, logging or quality checks.

        These attributes provide the foundation for adding, managing, and applying filters across various feature groups
        and features in the context of a data processing pipeline.
        """
        self.filters: Set[SingleFilter] = set()
        self.collection: Dict[Tuple[Type[AbstractFeatureGroup], FeatureName], Set[SingleFilter]] = {}

    def add_filter(
        self, filter_feature: Union[Feature, str], filter_type: Union[str, FilterTypeEnum], parameter: Dict[str, Any]
    ) -> None:
        """
        Adds a `SingleFilter` to the `filters` set based on the provided feature, filter type, and parameters.

        Parameters:
        - filter_feature: The feature or its name used for filtering. It can be a string or a `Feature` object.
            To identify if a filter is used, we need to check if the feature is part of the feature group.
            During this process, we enrich the filter feature with the options of the feature.
        - filter_type: The type of filtering operation (e.g., equals, greater than). It can be a string or a `FilterTypeEnum`.
            This filter_type does not need to match the FilterTypeEnum, but it should be a string that is meaningful in the concrete
            Featuregroup implementation.
        - parameter: A dictionary of filter-specific options.
        """
        _single_filter = SingleFilter(filter_feature, filter_type, parameter)
        self.filters.add(_single_filter)

    def add_filter_to_collection(
        self,
        feature_group: Type[AbstractFeatureGroup],
        filtered_feature_name: FeatureName,
        single_filter: SingleFilter,
    ) -> None:
        """
        The purpose of the functionality is to store the used filter features for a specific feature group and feature.
        This way we can check after the fact if a feature is a filter feature for a specific feature group.
        """
        if (feature_group, filtered_feature_name) not in self.collection:
            self.collection[(feature_group, filtered_feature_name)] = set([single_filter])
        self.collection[(feature_group, filtered_feature_name)].add(single_filter)

    def identity_matched_filters(
        self,
        feature_group: Type[AbstractFeatureGroup],
        feat: Feature,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> Set[SingleFilter]:
        """
        We need to figure out if the filter feature is a part of the feature class and thus can be used as filter.

        This is quite similar to identifying the feature itself.
        Differences are in details:
            -   we use the options of the feature to enrich the filter feature options,
            -   we set the compute framework of the feature to determine the one of the filter feature,
            -   we do not check links, as this is done earlier already and not needed anymore.
        """

        matched_filters: Set[SingleFilter] = set()
        for filter in self.filters:
            # We are making a deepcopy so that, we do not change the original filter.
            _filter = deepcopy(filter)
            _filter.filter_feature.options = self.unify_options(feat.options, _filter.filter_feature.options)

            if self.criteria(feature_group, _filter, data_access_collection) is False:
                continue
            if self.domain(_filter, feat.domain, feature_group) is False:
                continue
            if self.compute_framework(_filter, feat) is False:
                continue
            # we don t check links, because this is not necessary as this is covered by the feature and feature group before

            matched_filters.add(_filter)
        return matched_filters

    def unify_options(self, feat_options: Options, filter_options: Options) -> Options:
        for key, value in feat_options.data.items():
            if key not in filter_options.data:
                filter_options.data[key] = value
            else:
                if filter_options.data[key] == value:
                    continue
                else:
                    logger.warning(
                        f"Options are not the same. {key} is different. {filter_options.data[key]} != {value}"
                    )
        return filter_options

    def criteria(
        self,
        feature_group: Type[AbstractFeatureGroup],
        filter: SingleFilter,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return feature_group.match_feature_group_criteria(
            filter.filter_feature.name, filter.filter_feature.options, data_access_collection
        )

    def domain(
        self, filter: SingleFilter, feature_domain: Union[None, Domain], feature_group: Type[AbstractFeatureGroup]
    ) -> bool:
        # We have matched already the feature group and the feature.
        # Thus, we take the feature group domain if the feature domain is not set.
        feature_or_group_domain = None
        if feature_domain:
            feature_or_group_domain = feature_domain
        else:
            if feature_group.get_domain() != Domain.get_default_domain():
                feature_or_group_domain = feature_group.get_domain()

        # no domains given -> ok
        if not filter.filter_feature.domain and not feature_or_group_domain:
            return True

        # In case that filter has no domain given, we assume that the feature domain is the one to take.
        # Else the feature group should not have matched the feature domain and thus, we would not be here.
        if not filter.filter_feature.domain and feature_or_group_domain:
            filter.filter_feature.domain = feature_or_group_domain
            return True

        # In case that the filter has a domain and the feature not, it means that the
        # the feature group domain must be equal to the filter feature domain
        if filter.filter_feature.domain and not feature_domain:
            if feature_group.get_domain() == filter.filter_feature.domain:
                return True

        # both domains same -> ok
        if filter.filter_feature.domain == feature_domain:
            return True

        return False

    def compute_framework(self, filter: SingleFilter, feat: Feature) -> bool:
        # case that the filter feature has no cf set -> feature defines it
        if not filter.filter_feature.compute_frameworks:
            filter.filter_feature.compute_frameworks = feat.compute_frameworks
            return True

        # case that the filter feature has an cf -> it must be equal to the feature
        if next(iter(filter.filter_feature.compute_frameworks)) == feat.get_compute_framework():
            return True

        return False

    def add_time_and_time_travel_filters(
        self,
        event_from: datetime,
        event_to: datetime,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        max_exclusive: bool = True,
        time_filter_feature: Union[str, Feature] = "time_filter",
        time_travel_filter_feature: Union[str, Feature] = "time_travel_filter",
    ) -> None:
        """
        Adds time-based filters (`event_from`, `event_to`) and optionally time-travel filters (`valid_from`, `valid_to`).
        Ensures that both `valid_from` and `valid_to` are provided together, or raises an error.

        This method is useful for filtering data based on time ranges (event) and validity periods (valid).
            Event Time Filter: For historical data (e.g., checking if a customer had a valid contract at the event time), only the event time filter is needed.

            Time Travel Filter: If prior actions (e.g., payments made before the event) are relevant,
            the time travel filter is required.

            Typically, valid_to matches the event timestamp, but in cases like payment plans, where payments occur after creation, some payments may be excluded based on the valid_to data.

        Parameters:
        - event_from (datetime): Start of the time range (with timezone).
        - event_to (datetime): End of the time range (with timezone).
        - valid_from (Optional[datetime]): Start of the validity period (optional, with timezone).
        - valid_to (Optional[datetime]): End of the validity period (optional, with timezone).
        - max_exclusive (bool): If True, the `event_to` and `valid_to` values are treated as exclusive.
        - time_filter_feature: the feature description for the time filter. Default is "time_filter".
        - time_travel_filter_feature: the feature description for the time travel filter. Default is "time_travel_filter".

        The `single_filters` created will be converted to UTC as ISO 8601 formatted strings to ensure consistency
        across time zones and avoid ambiguity when comparing or processing time-based data.
        """

        self._add_range_filter(time_filter_feature, event_from, event_to, max_exclusive)

        # validate that both valid_from and valid_to are provided together
        if (valid_from is not None and valid_to is None) or (valid_from is None and valid_to is not None):
            raise ValueError("Both `valid_from` and `valid_to` must be provided together, or neither should be.")

        if valid_from and valid_to:
            self._add_range_filter(time_travel_filter_feature, valid_from, valid_to, max_exclusive)

    def _add_range_filter(
        self, filter_feature: Union[str, Feature], time_from: datetime, time_to: datetime, max_exclusive: bool
    ) -> None:
        _time_from = self._check_and_convert_time_info(time_from)
        _time_to = self._check_and_convert_time_info(time_to)
        self.add_filter(
            filter_feature, FilterTypeEnum.range, {"min": _time_from, "max": _time_to, "max_exclusive": max_exclusive}
        )

    def _check_and_convert_time_info(self, time_with_tz: datetime) -> str:
        """
        Checks that the provided datetime has timezone information and converts it to ISO 8601 format
        in UTC for filtering.

        We are working with tzinfo, as this is since python 3.9 included in the standard library.
        Most libraries can handle it or at least use pandas for transformations.
        """

        if time_with_tz.tzinfo is None:
            raise ValueError(f"Timezone information is missing in {time_with_tz}")

        # Convert to UTC and return the ISO 8601 formatted string
        return time_with_tz.astimezone(timezone.utc).isoformat()
