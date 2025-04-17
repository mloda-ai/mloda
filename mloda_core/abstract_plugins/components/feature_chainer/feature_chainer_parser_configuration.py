from typing import Optional, Set
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options


class FeatureChainParserConfiguration:
    """
    Configuration class for feature chain parsing if feature chaining is used!

    This class provides a way to parse feature names from options, allowing feature groups
    to be created from configuration rather than explicit feature names. It works in conjunction
    with the FeatureChainParser to provide a flexible way to create and manage features.

    Feature groups can implement their own parser configuration by subclassing this class
    and implementing the required methods. The parser configuration is then registered with
    the feature group by overriding the configurable_feature_chain_parser method.

    Example:
        class MyFeatureGroup(AbstractFeatureGroup):
            @classmethod
            def configurable_feature_chain_parser(cls) -> Optional[Type[FeatureChainParserConfiguration]]:
                return MyFeatureChainParserConfiguration

        class MyFeatureChainParserConfiguration(FeatureChainParserConfiguration):
            @classmethod
            def parse_keys(cls) -> Set[str]:
                return {"my_option_key", "source_feature"}

            @classmethod
            def parse_from_options(cls, options: Options) -> Optional[str]:
                # Parse options and return a feature name
                ...
    """

    @classmethod
    def parse_keys(cls) -> Set[str]:
        """
        Returns the keys that are used to parse the feature group.

        This method should return a set of keys that are relevant for parsing
        the feature group. The default implementation returns an empty set.
        """
        return set()

    @classmethod
    def parse_from_options(cls, options: Options) -> Optional[str]:
        """
        Parse a feature name from options.

        This method should be implemented by subclasses to parse a feature name from
        the provided options. It should return a string representing the feature name,
        or None if parsing fails.

        The feature name should follow the naming convention of the feature group,
        typically in the format "{prefix}__{source_feature}".

        Args:
            options: An Options object containing the configuration options

        Returns:
            A string representing the feature name, or None if parsing fails
        """
        return None

    @classmethod
    def create_feature_without_options(cls, feature: Feature) -> Optional[Feature]:
        """
        Create a feature from options, removing the parsed options from the feature.

        This method takes a feature with options, parses those options to create a new feature name,
        removes the parsed options from the feature, and returns a new feature with the parsed name.

        The parsing is done by calling parse_from_options with the feature's options. If parsing
        fails (returns None), this method also returns None.

        Args:
            feature: A Feature object containing options to parse

        Returns:
            A new Feature object with the parsed name and without the parsed options,
            or None if parsing fails
        """

        parse_keys = cls.parse_keys()
        if not parse_keys:
            return None

        feature_name = cls.parse_from_options(feature.options)
        if feature_name is None:
            return None

        feature.options.data = {k: v for k, v in feature.options.data.items() if k not in parse_keys}
        feature.name = FeatureName(feature_name)

        return feature
