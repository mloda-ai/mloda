"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
    create_configurable_parser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class AggregatedFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all aggregated feature groups.

    The AggregatedFeatureGroup performs aggregation operations on source features,
    such as sum, average, minimum, maximum, etc. It extracts the source feature from
    the feature name and applies the specified aggregation operation.

    ## Feature Naming Convention

    Aggregated features follow this naming pattern:
    `{aggregation_type}_aggr__{mloda_source_feature}`

    The source feature (mloda_source_feature) is extracted from the feature name and used
    as input for the aggregation operation. Note the double underscore before the source feature.

    Examples:
    - `sum_aggr__sales`: Sum of sales values
    - `avg_aggr__temperature`: Average of temperature values
    - `max_aggr__price`: Maximum price value

    ## Configuration-Based Creation

    AggregatedFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create an aggregated feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
            DefaultOptionKeys.mloda_source_feature: "Sales"
        })
    )

    # The Engine will automatically parse this into a feature with name "sum_aggr__Sales"
    ```

    The configuration-based approach is particularly useful when chaining multiple
    feature groups together, which need additional configuration.
    """

    # Option key for aggregation type
    AGGREGATION_TYPE = "aggregation_type"

    # Define supported aggregation types
    AGGREGATION_TYPES = {
        "sum": "Sum of values",
        "min": "Minimum value",
        "max": "Maximum value",
        "avg": "Average (mean) of values",
        "mean": "Average (mean) of values",
        "count": "Count of non-null values",
        "std": "Standard deviation of values",
        "var": "Variance of values",
        "median": "Median value",
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from the aggregated feature name."""

        mloda_source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        return {Feature(mloda_source_feature)}

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^([\w]+)_aggr__"

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        prefix_part = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if prefix_part is None:
            raise ValueError(f"Invalid aggregated feature name format: {feature_name}")
        return prefix_part

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern and aggregation type."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then check if the aggregation type is supported
            agg_type = cls.get_aggregation_type(feature_name)
            return cls._supports_aggregation_type(agg_type)
        except ValueError:
            return False

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if this feature group supports the given aggregation type."""
        return aggregation_type in cls.AGGREGATION_TYPES

    @classmethod
    def _raise_unsupported_aggregation_type(cls, aggregation_type: str) -> bool:
        """
        Raise an error for unsupported aggregation type.
        """
        raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform aggregations.

        Processes all requested features, determining the aggregation type
        and source feature from each feature name.

        Adds the aggregated results directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            aggregation_type = cls.get_aggregation_type(feature_name)
            source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            cls._check_source_feature_exists(data, source_feature)

            if aggregation_type not in cls.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            result = cls._perform_aggregation(data, aggregation_type, source_feature)

            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """
        Check if the source feature exists in the data.

        Args:
            data: The input data
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the data
        """
        raise NotImplementedError(f"_check_source_feature_exists not implemented in {cls.__name__}")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add

        Returns:
            The updated data
        """
        raise NotImplementedError(f"_add_result_to_data not implemented in {cls.__name__}")

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, mloda_source_feature: str) -> Any:
        """
        Method to perform the aggregation. Should be implemented by subclasses.

        Args:
            data: The input data
            aggregation_type: The type of aggregation to perform
            mloda_source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        raise NotImplementedError(f"_perform_aggregation not implemented in {cls.__name__}")

    @classmethod
    def configurable_feature_chain_parser(cls) -> Optional[Type[FeatureChainParserConfiguration]]:
        """
        Returns the FeatureChainParserConfiguration class for this feature group.

        This method allows the Engine to automatically create features with the correct
        naming convention based on configuration options, rather than requiring explicit
        feature names.

        Returns:
            A configured FeatureChainParserConfiguration class
        """
        return create_configurable_parser(
            parse_keys=[
                cls.AGGREGATION_TYPE,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{aggregation_type}_aggr__{mloda_source_feature}",
            validation_rules={
                cls.AGGREGATION_TYPE: lambda x: x in cls.AGGREGATION_TYPES
                or cls._raise_unsupported_aggregation_type(x),
            },
        )
