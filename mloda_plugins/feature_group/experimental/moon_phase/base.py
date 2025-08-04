"""
Base implementation for moon phase feature groups.

The moon phase feature group calculates the lunar phase for each timestamp.
It supports multiple representations of the moon phase, such as a continuous
fraction between 0 and 1, an angle in degrees from 0 to 360, or a binary
indicator signalling when the moon is near full. This base class defines
the configuration options, parsing logic for feature names, and the
high-level workflow for adding moon phase features to a dataset. Concrete
subclasses must implement the compute-specific details for checking input
features, calculating the phase, and adding the results back to the data.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Tuple, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class MoonPhaseFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all moon phase feature groups.

    Moon phase feature groups compute the lunar phase for a given timestamp.
    They allow users to obtain the phase either as a continuous fraction,
    an angle in degrees, or as a binary indicator of a full moon. Both
    string-based feature definitions and configuration-based creation
    are supported. The naming convention for string-based features is:

        ``moon_phase_{representation}__{mloda_source_feature}``

    where ``representation`` is one of ``fraction``, ``degrees`` or
    ``is_full``, and ``mloda_source_feature`` is the name of the datetime
    column used to compute the phase.
    """

    # Option key for selecting the representation of the moon phase
    PHASE_REPRESENTATION: str = "phase_representation"

    # Supported representations and their descriptions
    REPRESENTATIONS: dict[str, str] = {
        "fraction": (
            "Moon phase as a continuous value between 0 and 1 (0 = new moon, "
            "0.5 = full moon, 1.0 wraps back to new moon)"
        ),
        "degrees": (
            "Moon phase as an angle in degrees (0 = new moon, 180 = full moon, "
            "360 wraps back to new moon)"
        ),
        "is_full": (
            "Binary indicator for full moon. 1 if the phase is near full moon, "
            "0 otherwise"
        ),
    }

    # Prefix pattern used to identify this feature group in feature names.
    # Captures the representation type so it can be parsed later.
    PREFIX_PATTERN: str = r"^moon_phase_([\w]+)__"
    # Pattern used by the FeatureChainParser to split prefix and source part
    PATTERN: str = "__"

    # Property mapping for configuration-based feature creation. This
    # determines which options are allowed and whether they are treated
    # as context or group parameters. Strict validation ensures that
    # invalid representations are rejected early.
    PROPERTY_MAPPING: dict[str, dict[str, Any]] = {
        PHASE_REPRESENTATION: {
            **REPRESENTATIONS,
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
        },
        DefaultOptionKeys.mloda_source_feature: {
            "explanation": "Datetime feature representing the timestamp to compute the moon phase for",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """
        Extract the source time feature from either configuration-based options or string parsing.

        Parameters
        ----------
        options : Options
            Feature options which may include the source feature in context.
        feature_name : FeatureName
            The name of the feature to parse for the source feature.

        Returns
        -------
        Optional[Set[Feature]]
            A set containing the single source feature required by this feature group.
        """
        # Attempt to parse the source feature from the feature name using the prefix pattern
        try:
            source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
            if source_feature:
                return {Feature(source_feature)}
        except ValueError:
            # Ignore parsing errors and fall back to options
            pass

        # Fall back to configuration-based approach
        source_features = options.get_source_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature for moon phase, got {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def _extract_moon_phase_parameters(cls, feature: Feature) -> Tuple[str, str]:
        """
        Extract the phase representation and time feature name from a feature definition.

        This method first tries to parse the representation and source feature from
        the string-based feature name. If that fails, it falls back to the
        configuration stored in the feature's options.

        Parameters
        ----------
        feature : Feature
            The feature from which to extract parameters.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the representation (e.g., "fraction", "degrees", "is_full")
            and the name of the time feature.
        """
        feature_name_str: str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        # Attempt string-based parsing
        if cls.PATTERN in feature_name_str:
            # parse_feature_name returns (prefix_part, source_part)
            prefix_part, source_feature = FeatureChainParser.parse_feature_name(
                feature_name_str, cls.PATTERN, [cls.PREFIX_PATTERN]
            )
            # prefix_part will be something like "moon_phase_fraction"
            # Remove the "moon_phase_" prefix to obtain the representation
            representation = prefix_part.replace("moon_phase_", "").strip()
            return representation, source_feature

        # Fallback: use configuration options
        source_features = feature.options.get_source_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature for moon phase, got {len(source_features)}: {source_features}"
            )
        time_feature_name: str = next(iter(source_features)).get_name()
        representation = feature.options.get(cls.PHASE_REPRESENTATION)
        if representation is None:
            raise ValueError(f"Could not extract phase representation from feature: {feature.name}")
        if representation not in cls.REPRESENTATIONS:
            raise ValueError(
                f"Unsupported phase representation: {representation}. Supported representations: "
                f"{', '.join(cls.REPRESENTATIONS.keys())}"
            )
        return representation, time_feature_name

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Calculate moon phase values for each requested feature and append them to the data.

        The method iterates over all requested features, extracts the representation and
        source time feature, verifies input requirements, delegates the phase calculation
        to the compute-framework specific implementation, and finally adds the result
        back to the dataset.

        Parameters
        ----------
        data : Any
            The input data structure (e.g., a pandas DataFrame) to which the
            calculated feature will be added.
        features : FeatureSet
            A collection of features requested for computation.

        Returns
        -------
        Any
            The input data with the computed moon phase features added.
        """
        # Loop through each feature in the feature set and process individually
        for feature in features.features:
            representation, time_feature_name = cls._extract_moon_phase_parameters(feature)
            # Validate representation
            if representation not in cls.REPRESENTATIONS:
                raise ValueError(f"Unsupported phase representation: {representation}")
            # Ensure the time feature exists in the data
            cls._check_time_feature_exists(data, time_feature_name)
            # Compute the moon phase using the compute-framework specific method
            result = cls._calculate_moon_phase(data, representation, time_feature_name)
            # Add the result back to the data with the appropriate feature name
            data = cls._add_result_to_data(data, feature.get_name(), result)
        return data

    @classmethod
    def _check_time_feature_exists(cls, data: Any, time_feature: str) -> None:
        """
        Verify that the specified time feature exists in the input data.

        This method should be implemented by subclasses tailored to a specific
        compute framework (e.g., pandas, Spark). If the time feature is missing
        in the data, a ValueError should be raised.
        """
        raise NotImplementedError(f"_check_time_feature_exists not implemented in {cls.__name__}")

    @classmethod
    def _calculate_moon_phase(cls, data: Any, representation: str, time_feature: str) -> Any:
        """
        Compute the moon phase for the given representation and time feature.

        This method is responsible for the core calculation logic and must be
        implemented by subclasses specific to the compute framework. It should
        return a series or array of values corresponding to the computed
        moon phase for each row in the input data.

        Parameters
        ----------
        data : Any
            The input data structure containing the time column.
        representation : str
            The desired representation of the moon phase (e.g., "fraction",
            "degrees", or "is_full").
        time_feature : str
            The name of the column in the data containing timestamps.

        Returns
        -------
        Any
            The computed moon phase values as a vector compatible with the
            underlying data structure.
        """
        raise NotImplementedError(f"_calculate_moon_phase not implemented in {cls.__name__}")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the computed result to the data under the given feature name.

        Subclasses must implement this method to append the result to the data
        appropriately for the compute framework. For example, in pandas, this would
        assign a new column to the DataFrame.

        Parameters
        ----------
        data : Any
            The input data structure to which the result should be added.
        feature_name : str
            The name of the feature (column) to store the result.
        result : Any
            The computed moon phase values.

        Returns
        -------
        Any
            The data structure with the result added.
        """
        raise NotImplementedError(f"_add_result_to_data not implemented in {cls.__name__}")
