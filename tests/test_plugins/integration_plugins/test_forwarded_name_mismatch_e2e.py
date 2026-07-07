"""Failing end-to-end test pinning the forwarded-name-mismatch check through the engine.

A consumer feature group requests a string-parsed chained child
("namemis579_source__sum_namemis579"). The consumer is requested with the group
option {"operation_namemis579": "max"}, which forward-by-default inherits onto
the child. The child's name parses "sum", so the inherited "max" would be
silently ignored today. The new check in
FeatureChainParserMixin.match_feature_group_criteria must instead raise
ValueError during resolution, so mloda.run_all fails.

All fixture names carry a "namemis579" marker so they cannot collide with
other tests in the global plugin registry.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


SOURCE_NAME = "namemis579_source"
OPERATION_KEY = "operation_namemis579"
CHAINED_CHILD_NAME = f"{SOURCE_NAME}__sum_namemis579"
CONSUMER_NAME = "namemis579_consumer"


class NameMis579SourceGroup(FeatureGroup):
    """Root group providing the source column via DataCreator."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({SOURCE_NAME})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({SOURCE_NAME: [1, 2, 3]})


class NameMis579ChainedGroup(FeatureChainParserMixin, FeatureGroup):
    """String-parsed chained group declaring the group-categorized operation key."""

    PREFIX_PATTERN = r".*__(sum|max)_namemis579$"
    PROPERTY_MAPPING = {
        OPERATION_KEY: {
            "sum": "Sum of the in feature (namemis579 fixture)",
            "max": "Maximum of the in feature (namemis579 fixture)",
            DefaultOptionKeys.strict_validation: True,
        }
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        # Fixture guard: only claim chained names. The consumer feature and the root
        # source feature legitimately carry the operation key in their group options,
        # which would otherwise config-match this group and make resolution ambiguous.
        # The string-parse path under test still runs through the mixin via super().
        if "__" not in str(feature_name):
            return False
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            operation = cls._resolve_operation(str(feature.name), feature.options, OPERATION_KEY)
            source_column = str(feature.name).rsplit("__", 1)[0]
            if operation == "max":
                data[feature.name] = data[source_column].max()
            else:
                data[feature.name] = data[source_column].sum()
        return data


class NameMis579ConsumerGroup(FeatureGroup):
    """Consumer requesting the chained child; forwards its group options by default."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        return str(feature_name) == CONSUMER_NAME

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(CHAINED_CHILD_NAME)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[feature.name] = data[CHAINED_CHILD_NAME]
        return data


class TestForwardedNameMismatchEndToEnd:
    def test_run_all_raises_on_forwarded_name_mismatch(self) -> None:
        """A consumer group option contradicting the child's name-parsed value fails the run."""
        consumer = Feature(CONSUMER_NAME, options=Options(group={OPERATION_KEY: "max"}))

        with pytest.raises(ValueError, match=OPERATION_KEY):
            mloda.run_all(
                [consumer],
                compute_frameworks={PandasDataFrame},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {NameMis579SourceGroup, NameMis579ChainedGroup, NameMis579ConsumerGroup}
                ),
            )
