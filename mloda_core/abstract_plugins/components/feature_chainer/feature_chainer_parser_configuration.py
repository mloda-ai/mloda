from typing import Any, Callable, Dict, List, Optional, Set, Type
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options


def create_configurable_parser(
    parse_keys: List[str],
    feature_name_template: str,
    validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
    required_keys: Optional[List[str]] = None,
) -> Type["FeatureChainParserConfiguration"]:
    """
    Create a configured FeatureChainParserConfiguration class.

    This factory function creates a new FeatureChainParserConfiguration subclass
    that is configured with the provided parameters. This allows feature groups
    to easily create a parser configuration without having to define a new class.

    Args:
        parse_keys: List of keys used for parsing and included in the feature name template
        feature_name_template: String template for feature names
        validation_rules: Optional dict mapping keys to validation functions
        required_keys: Optional list of keys that must be present in the options but are not used in the template

    Returns:
        A new FeatureChainParserConfiguration subclass configured with the provided parameters

    Example:
        @classmethod
        def configurable_feature_chain_parser(cls):
            return create_configurable_parser(
                parse_keys=[cls.WINDOW_FUNCTION, cls.WINDOW_SIZE, cls.TIME_UNIT, DefaultOptionKeys.mloda_source_feature],
                feature_name_template="{WINDOW_FUNCTION}_{WINDOW_SIZE}_{TIME_UNIT}_window__{mloda_source_feature}",
                validation_rules={
                    cls.WINDOW_FUNCTION: lambda x: x in cls.WINDOW_FUNCTIONS,
                    cls.TIME_UNIT: lambda x: x in cls.TIME_UNITS,
                    cls.WINDOW_SIZE: lambda x: isinstance(x, int) and x > 0
                }
            )
    """
    validation_rules = validation_rules or {}

    class ConfiguredParser(FeatureChainParserConfiguration):
        @classmethod
        def parse_keys(cls) -> Set[str]:
            return set(parse_keys)

        @classmethod
        def parse_from_options(cls, options: Options) -> Optional[str]:
            # Check required keys
            if required_keys:
                for key in required_keys:
                    value = options.get(key)
                    if value is None:
                        return None

                    # Apply validation if defined
                    if validation_rules and key in validation_rules and not validation_rules[key](value):
                        raise ValueError(f"Invalid value for {key}: {value}")

            # Extract values from options for the template
            values = {}
            for key in parse_keys:
                value = options.get(key)
                if value is None:
                    return None

                # Apply validation if defined
                if validation_rules and key in validation_rules and not validation_rules[key](value):
                    raise ValueError(f"Invalid value for {key}: {value}")

                # Handle list values by joining them with commas
                if isinstance(value, list):
                    value = ",".join(str(item) for item in value)

                values[key] = value

            # Format the feature name using the template
            try:
                return feature_name_template.format(**values)
            except KeyError as e:
                raise ValueError(f"Missing key in template: {e}")

    return ConfiguredParser


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
