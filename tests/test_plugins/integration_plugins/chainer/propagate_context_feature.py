"""Feature group for testing propagate_context_keys behavior in chained features."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureChainParserMixin, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.user import Feature

from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.provider import FeatureChainParser
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec


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
            default=None,  # Optional: features without env fall back to offset 0
        ),
        DefaultOptionKeys.in_features: PropertySpec("explanation", context=True),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any | None = None,
    ) -> bool:
        if not FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name=feature_name,
            options=options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=cls.SUFFIX_PATTERN,
        ):
            return False

        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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


class NameChainedCascadeGroupTest(FeatureChainParserMixin, FeatureGroup):
    """Name-chained group: every level doubles its source and adds the offset of its env, which is required."""

    PREFIX_PATTERN = r".*__([\w]+)_ncc9x$"
    PROPERTY_MAPPING = {
        "operation": PropertySpec(
            "Operation applied to the source values",
            allowed_values={"scale": "Doubles the source values"},
            context=True,
            strict_validation=True,
        ),
        "env": PropertySpec(
            "Environment selecting the offset",
            allowed_values={"prod": "production offset 1000"},
            context=True,
            strict_validation=False,
            default=None,
        ),
        DefaultOptionKeys.in_features: PropertySpec("Source features", context=True),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            if feature.options.get("env") != "prod":
                raise ValueError(f"Feature '{feature.name}' needs env=prod")
            source = cls._extract_source_features(feature)[0]
            data[feature.name] = data[source] * 2 + 1000
        return data


CASCADE_ENV_ROOT = "cer_env_root"


class CascadeEnvRootGroupTest(DataCreatorFeatureGroup):
    """Root whose values depend on env: [10, 20, 30] for prod, otherwise [1, 2, 3]."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {CASCADE_ENV_ROOT}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        factor = 10 if next(iter(features.features)).options.get("env") == "prod" else 1
        return pd.DataFrame({CASCADE_ENV_ROOT: [factor, 2 * factor, 3 * factor]})


CASCADE_GUARD_ROOT = "cgd_root"
CASCADE_GUARD_SHARED = "cgd_shared"
CASCADE_GUARD_CASCADE_CONSUMER = "cgd_consumer_cascade"
CASCADE_GUARD_PULL_CONSUMER = "cgd_consumer_pull"
CASCADE_GUARD_HOP_CONSUMER = "cgd_consumer_hop"


class CascadeGuardRootGroupTest(DataCreatorFeatureGroup):
    """Root of the cascade guard fixtures."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {CASCADE_GUARD_ROOT}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({CASCADE_GUARD_ROOT: [1, 2, 3]})


class _CascadeGuardNamedGroup(FeatureGroup):
    FEATURE_NAMES: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any | None = None,
    ) -> bool:
        return str(feature_name) in cls.FEATURE_NAMES

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class CascadeGuardSharedGroupTest(_CascadeGuardNamedGroup):
    """Non-root input shared by the consumers: root times ten."""

    FEATURE_NAMES = frozenset({CASCADE_GUARD_SHARED})

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(CASCADE_GUARD_ROOT)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[feature.name] = data[CASCADE_GUARD_ROOT] * 10
        return data


class CascadeGuardConsumerGroupTest(_CascadeGuardNamedGroup):
    """Consumer of the context option 'target'; the pull consumer pulls k, the others rely on the request options."""

    FEATURE_NAMES = frozenset({CASCADE_GUARD_CASCADE_CONSUMER, CASCADE_GUARD_PULL_CONSUMER, CASCADE_GUARD_HOP_CONSUMER})

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        target = str(options.get("target"))
        if str(feature_name) == CASCADE_GUARD_PULL_CONSUMER:
            return {Feature(target, inherit_context_keys={"k"})}
        return {Feature(target)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[feature.name] = data[str(feature.options.get("target"))] + 1
        return data


CASCADE_FOLD_SHARED = "cgf_shared"
CASCADE_FOLD_CASCADE_CONSUMER = "cgf_consumer_cascade"
CASCADE_FOLD_PINNED_CONSUMER = "cgf_consumer_pinned"


class CascadeFoldSharedGroupTest(CascadeGuardSharedGroupTest):
    """Shared input allowed on two compute frameworks."""

    FEATURE_NAMES = frozenset({CASCADE_FOLD_SHARED})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame, PyArrowTable}


class CascadeFoldConsumerGroupTest(_CascadeGuardNamedGroup):
    """The pinned consumer pins PandasDataFrame on its input; the cascade consumer leaves it open."""

    FEATURE_NAMES = frozenset({CASCADE_FOLD_CASCADE_CONSUMER, CASCADE_FOLD_PINNED_CONSUMER})

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if str(feature_name) == CASCADE_FOLD_PINNED_CONSUMER:
            return {Feature(CASCADE_FOLD_SHARED, inherit_context_keys={"k"}, compute_framework="PandasDataFrame")}
        return {Feature(CASCADE_FOLD_SHARED)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[feature.name] = data[CASCADE_FOLD_SHARED] + 1
        return data
