"""Feature group for testing propagate_context_keys behavior in chained features."""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda.provider import FeatureGroup
from mloda.user import Feature

from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.provider import FeatureChainParser
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class PropagateContextFeatureGroupTest(FeatureGroup):
    SUFFIX_PATTERN = [r".*__propctx_([\w]+)$"]
    OPERATION_ID = "propctx_"

    PROPERTY_MAPPING = {
        "ident": {
            "identifier1": "multiplier 2",
            "identifier2": "multiplier 3",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "env": {
            "prod": "production offset 1000",
            "staging": "staging offset 500",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "explanation",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        if not FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name=feature_name,
            options=options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=cls.SUFFIX_PATTERN,
        ):
            return False

        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        features = set()

        source_features = options.get_in_features()
        for source_feature in source_features:
            source_feature.options.add(
                DefaultOptionKeys.feature_chainer_parser_key,
                frozenset(["ident", DefaultOptionKeys.in_features.value]),
            )
            features.add(source_feature)

        if features:
            return features
        raise ValueError

    @classmethod
    def perform_operation(cls, data: Any, feature: Feature) -> Any:
        source_features = feature.options.get_in_features()
        source_feature = next(iter(source_features))
        source_feature_name: str = source_feature.get_name()

        ident = feature.options.get("ident")
        if ident == "identifier1":
            multiplier = 2
        elif ident == "identifier2":
            multiplier = 3
        else:
            raise ValueError(f"Unknown ident value: {ident}")

        env = feature.options.get("env")
        if env == "prod":
            offset = 1000
        elif env == "staging":
            offset = 500
        else:
            offset = 0

        name = feature.get_name()
        data[name] = data[source_feature_name] * multiplier + offset
        return data

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data = cls.perform_operation(data, feature)
        return data
