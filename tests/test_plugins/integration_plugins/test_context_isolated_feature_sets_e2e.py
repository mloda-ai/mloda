"""End-to-end coverage for context-isolated FeatureSets.

Two requested consumer features carry different context values (tenant "acme" vs
"beta") and each declares the SAME child feature name with
inherit_context_keys={"tenant"}. The child feature group's output depends on the
pulled tenant. Because the execution plan currently groups features into
FeatureSets by context-blind similarity hashes, both child features collapse into
ONE FeatureSet: calculate_feature runs once and one consumer silently receives
the other tenant's data.

New spec: features whose options.context differ must NOT share a FeatureSet, so
each consumer's final output reflects its own tenant and the child computes twice.

The child source group records the contexts it is resolved with into class-level
state, which the test resets before running. All fixture feature names carry a
"ctxiso_fs_e2e" marker so they cannot collide with other tests in the global
plugin registry.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import pandas as pd

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


SOURCE_NAME = "ctxiso_fs_e2e_source"

TENANT_VALUES: dict[str, list[int]] = {
    "acme": [1, 2, 3],
    "beta": [100, 200, 300],
}


class ContextIsoSourceGroup(FeatureGroup):
    """Root child group whose output depends on the pulled "tenant" context value.

    Records every resolved feature's context and counts calculate_feature invocations
    so the test can pin that context-distinct children compute separately.
    """

    seen_contexts: ClassVar[list[dict[str, Any]]] = []
    invocation_count: ClassVar[int] = 0

    @classmethod
    def reset_recording(cls) -> None:
        cls.seen_contexts = []
        cls.invocation_count = 0

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({SOURCE_NAME})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.invocation_count += 1
        tenant = None
        for feature in features.features:
            cls.seen_contexts.append(dict(feature.options.context))
            tenant = feature.options.get("tenant")
        values = TENANT_VALUES.get(tenant, [-1, -1, -1]) if tenant is not None else [-1, -1, -1]
        return pd.DataFrame({SOURCE_NAME: values})


class _ContextIsoConsumerBase(FeatureGroup):
    """Shared consumer behavior: copy the child column into the consumer column.

    The consumer output therefore equals exactly the child data it received, making
    any cross-tenant data leak directly visible in the final frames.
    """

    FEATURE_NAME: ClassVar[str] = ""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == cls.FEATURE_NAME

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, inherit_context_keys={"tenant"})}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[feature.name] = data[SOURCE_NAME]
        return data


class ContextIsoAcmeConsumer(_ContextIsoConsumerBase):
    """Consumer requested with context {"tenant": "acme"}."""

    FEATURE_NAME = "ctxiso_fs_e2e_acme_consumer"


class ContextIsoBetaConsumer(_ContextIsoConsumerBase):
    """Consumer requested with context {"tenant": "beta"}."""

    FEATURE_NAME = "ctxiso_fs_e2e_beta_consumer"


def _frame_with_column(results: list[Any], column: str) -> Any:
    for frame in results:
        if column in frame.columns:
            return frame
    raise AssertionError(f"Column '{column}' not found in any result frame: {[list(r.columns) for r in results]}")


def _run(features: list[Feature | str], groups: set[type[FeatureGroup]]) -> list[Any]:
    return mloda.run_all(
        features,
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


class TestContextIsolatedFeatureSetsE2E:
    def test_context_differing_children_compute_separately(self) -> None:
        """Each consumer's output must reflect ITS OWN tenant value.

        Both tenants are asserted simultaneously: as long as the two same-named,
        context-distinct child features collapse into one FeatureSet, both consumers
        receive the same tenant's data and at least one of the two lists is wrong,
        regardless of which tenant happens to win the collapsed computation.
        """
        ContextIsoSourceGroup.reset_recording()

        consumers: list[Feature | str] = [
            Feature(
                ContextIsoAcmeConsumer.FEATURE_NAME,
                options=Options(context={"tenant": "acme"}),
            ),
            Feature(
                ContextIsoBetaConsumer.FEATURE_NAME,
                options=Options(context={"tenant": "beta"}),
            ),
        ]
        results = _run(
            consumers,
            {ContextIsoSourceGroup, ContextIsoAcmeConsumer, ContextIsoBetaConsumer},
        )

        acme_frame = _frame_with_column(results, ContextIsoAcmeConsumer.FEATURE_NAME)
        beta_frame = _frame_with_column(results, ContextIsoBetaConsumer.FEATURE_NAME)
        acme_values = acme_frame[ContextIsoAcmeConsumer.FEATURE_NAME].tolist()
        beta_values = beta_frame[ContextIsoBetaConsumer.FEATURE_NAME].tolist()

        assert (acme_values, beta_values) == (TENANT_VALUES["acme"], TENANT_VALUES["beta"]), (
            "Each consumer must receive the child data computed with its own tenant context. "
            f"Got acme={acme_values} (expected {TENANT_VALUES['acme']}) and "
            f"beta={beta_values} (expected {TENANT_VALUES['beta']}). Equal lists mean the "
            "context-distinct child features collapsed into one FeatureSet."
        )

        # Pin: the child computed TWICE, once per tenant context.
        assert ContextIsoSourceGroup.invocation_count == 2, (
            "The child feature group must run once per context-distinct FeatureSet. "
            f"Got {ContextIsoSourceGroup.invocation_count} invocation(s) with recorded "
            f"contexts {ContextIsoSourceGroup.seen_contexts}."
        )
        seen_tenants = sorted(str(context.get("tenant")) for context in ContextIsoSourceGroup.seen_contexts)
        assert seen_tenants == ["acme", "beta"]
