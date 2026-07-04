"""End-to-end tests for the property_spec validation_function passthrough (issue #536).

Demonstrates the user-facing story of PR feat/property-spec-passthroughs: a feature
group author declares a "window_size" option via
``property_spec(..., strict=True, validation_function=...)`` and gets two distinct
validation moments for free:

1. Planning time: an end-user request through the public mloda API with an invalid
   window_size raises ValueError before any computation runs.
2. Authoring time: ``property_spec`` itself rejects a declared default that fails
   the validation_function, at the call site (class-definition time).

The derived feature group uses FeatureChainParserMixin for input_features but
overrides ``match_feature_group_criteria`` to call
``FeatureChainParser.match_configuration_feature_chain_parser`` directly, following
``test_list_valued_options_e2e.py``. The mixin's default match catches the
validation ValueError and returns False (see
``test_strict_validation_returns_false.py``), which would surface only as a generic
"No feature groups found" error; the direct call preserves the actionable message
naming the rejected option.
"""

from typing import Any, Optional

import pandas as pd
import pytest

from mloda.provider import (
    BaseInputData,
    ComputeFramework,
    DataCreator,
    DefaultOptionKeys,
    FeatureChainParser,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    property_spec,
)
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

# Records calculate_feature invocations so tests can assert that planning-time
# rejection happens before any computation runs.
CALCULATE_INVOCATIONS: list[str] = []


def is_valid_window_size(value: Any) -> bool:
    """Module-level validator (named function, not a lambda, for pickling safety)."""
    return isinstance(value, int) and 0 < value <= 13


class PropertySpecE2EDataCreator(FeatureGroup):
    """Root data provider supplying a numeric base column."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"property_spec_e2e_base"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        CALCULATE_INVOCATIONS.append(cls.__name__)
        return pd.DataFrame({"property_spec_e2e_base": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class PropertySpecE2EWindowedFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Derived feature group whose "window_size" option is guarded by property_spec.

    The computation (base column + window_size) is intentionally trivial; the point
    is the validation_function passthrough, not rolling-window logic.
    """

    PROPERTY_MAPPING = {
        "window_size": property_spec(
            "Size of the time window, at most 13",
            strict=True,
            validation_function=is_valid_window_size,
        ),
        DefaultOptionKeys.in_features: property_spec("Source feature to window over"),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        _name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return FeatureChainParser.match_configuration_feature_chain_parser(
            _name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        CALCULATE_INVOCATIONS.append(cls.__name__)
        for feature in features.features:
            window_size = feature.options.get("window_size")
            data[feature.name] = data["property_spec_e2e_base"] + window_size
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class TestPropertySpecValidationFunctionE2E:
    """End-to-end coverage of the two validation moments enabled by property_spec."""

    plugin_collector = PluginCollector.enabled_feature_groups(
        {PropertySpecE2EDataCreator, PropertySpecE2EWindowedFeatureGroup}
    )

    def test_valid_window_size_runs_end_to_end(self) -> None:
        """window_size=7 passes the validator; the run returns the computed column."""
        feature = Feature(
            name="property_spec_e2e_windowed",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "property_spec_e2e_base",
                    "window_size": 7,
                },
            ),
        )

        result = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        assert len(result) >= 1

        for df in result:
            if "property_spec_e2e_windowed" in df.columns:
                # base [1, 2, 3] + window_size 7
                assert df["property_spec_e2e_windowed"].tolist() == [8, 9, 10]
                assert PropertySpecE2EWindowedFeatureGroup.__name__ in CALCULATE_INVOCATIONS
                return

        raise AssertionError("property_spec_e2e_windowed not found in results")

    def test_invalid_window_size_rejected_at_planning_time(self) -> None:
        """window_size=14 fails the validator: ValueError before any computation runs."""
        invocations_before = len(CALCULATE_INVOCATIONS)

        feature = Feature(
            name="property_spec_e2e_windowed_invalid",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "property_spec_e2e_base",
                    "window_size": 14,
                },
            ),
        )

        with pytest.raises(ValueError, match="window_size"):
            mloda.run_all(
                [feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=self.plugin_collector,
            )

        assert len(CALCULATE_INVOCATIONS) == invocations_before, "rejection must happen before any computation"

    def test_author_side_default_rejected_at_call_site(self) -> None:
        """A default rejected by the validation_function fails at the property_spec call."""
        with pytest.raises(ValueError, match="default 14 is rejected by the validation_function"):
            property_spec(
                "Size of the time window, at most 13",
                strict=True,
                validation_function=is_valid_window_size,
                default=14,
            )
