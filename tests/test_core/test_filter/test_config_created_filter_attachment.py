"""A name-only global filter must still attach to a config-created chained feature.

The filter feature's name is a raw column matching no PREFIX_PATTERN, so its match runs the option
path, where an absent required in_features key is an immediate non-match until unify_options supplies it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, DefaultOptionKeys, FeatureGroup, FeatureSet
from mloda.user import Feature, FilterType, GlobalFilter, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

# The cfa_ prefix keeps these names unique repo-wide.
CFA_SOURCE = "cfa_sales"
CFA_FILTER_COLUMN = "cfa_region"
CFA_HOST = "cfa_sum_sales"


class CfaRoot(FeatureGroup):
    """Root serving the aggregation source column and the column only the filter ever names."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({CFA_SOURCE, CFA_FILTER_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({CFA_SOURCE: [1, 2, 3], CFA_FILTER_COLUMN: [1, 2, 3]})


_CFA_ENABLED = PluginCollector.enabled_feature_groups({CfaRoot, PandasAggregatedFeatureGroup})


def _planned() -> GlobalFilter:
    """Plan one options-only aggregation, no source encoded in its name, under a name-only filter."""
    global_filter = GlobalFilter()
    global_filter.add_filter(CFA_FILTER_COLUMN, FilterType.MIN, {"value": 2})

    mloda.prepare(
        [
            Feature(
                CFA_HOST,
                Options(
                    context={
                        AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                        DefaultOptionKeys.in_features.value: CFA_SOURCE,
                    }
                ),
            )
        ],
        compute_frameworks={PandasDataFrame},
        plugin_collector=_CFA_ENABLED,
        global_filter=global_filter,
    )
    return global_filter


def test_the_filter_attaches_to_the_config_created_feature() -> None:
    """Without the host's in_features key the filter feature cannot match the aggregation group at all."""
    global_filter = _planned()

    attached = {
        str(name): sorted(single.name for single in stored)
        for (group, name), stored in global_filter.collection.items()
        if group is PandasAggregatedFeatureGroup
    }
    assert attached == {CFA_HOST: [CFA_FILTER_COLUMN]}, f"the filter must reach the aggregation step: {attached!r}"


def test_the_filter_still_attaches_to_the_root_it_reads_from() -> None:
    """Guard: the filter matching some other group is what keeps a detached aggregation step silent."""
    global_filter = _planned()

    attached = {group.get_class_name() for group, _name in global_filter.collection}
    assert CfaRoot.get_class_name() in attached, f"the root must keep its own attachment: {sorted(attached)!r}"
