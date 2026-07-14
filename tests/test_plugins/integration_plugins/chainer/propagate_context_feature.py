"""Feature group for testing propagate_context_keys behavior in chained features."""

from __future__ import annotations

from typing import Any, Optional

from mloda.provider import FeatureGroup
from mloda.user import Feature

from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.provider import FeatureChainParser
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec


def _env_is_never_required(options: Options) -> bool:
    """The retired dict spec marked env optional via a present default-None key;
    the PropertySpec equivalent is a required_when predicate that never fires."""
    return False


class PropagateContextFeatureGroupTest(FeatureGroup):
    SUFFIX_PATTERN = [r".*__propctx_([\w]+)$"]
    OPERATION_ID = "propctx_"

    PROPERTY_MAPPING = {
        "ident": PropertySpec(
            "Identifier selecting the multiplier",
            allowed_values={"identifier1": "multiplier 2", "identifier2": "multiplier 3"},
            context=True,
            strict_validation=True,
        ),
        "env": PropertySpec(
            "Environment selecting the offset",
            allowed_values={"prod": "production offset 1000", "staging": "staging offset 500"},
            context=True,
            strict_validation=False,
            required_when=_env_is_never_required,
        ),
        DefaultOptionKeys.in_features: PropertySpec("explanation", context=True),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
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

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        features = set()

        # Children inherit consumer group options by default; CONTEXT stays local and flows
        # only via the propagate_context_keys push.
        source_features = options.get_in_features()
        for source_feature in source_features:
            features.add(source_feature)

        if features:
            return features
        raise ValueError

    @classmethod
    def perform_operation(cls, data: Any, feature: Feature) -> Any:
        source_features = feature.options.get_in_features()
        source_feature = next(iter(source_features))
        source_feature_name: str = source_feature.name

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

        name = feature.name
        data[name] = data[source_feature_name] * multiplier + offset
        return data

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data = cls.perform_operation(data, feature)
        return data
