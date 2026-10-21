"""Engine handling of recorded match rejections.

A nested unresolvable input names its chain and drops the did-you-mean hint beside an input_data decline; an
unknown stage string falls back to ``value_rejection``. All names carry a ``rej727`` marker to stay inert for other tests.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from mloda.core.abstract_plugins.components.match_rejection import (
    record_match_rejection,
)
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.provider import (
    ClaimRoute,
    ComputeFramework,
    FeatureGroup,
    NamePolicy,
    SourceMatch,
)
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyFormatBase,
    toy_dac,
)


FEATURE_NAME_REJ727 = "rej727_column"
BOGUS_STAGE_FEATURE_REJ727 = "rej727_bogus_stage_column"
BOGUS_STAGE_REASON_REJ727 = "custom decline"


class Rej727ToyFormatFG(ToyFormatBase):
    """Format group serving only rej727_column from toy sources; inert for every other name."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("rej727", NamePolicy.CHECKED, True),)

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        return [column for column in match.access if column == FEATURE_NAME_REJ727]


class Rej727BogusStageFG(FeatureGroup):
    """Declines its single gated feature with an unknown custom stage; inert for every other name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != BOGUS_STAGE_FEATURE_REJ727:
            return False
        record_match_rejection(cls.get_class_name(), BOGUS_STAGE_REASON_REJ727, stage="rej727_bogus_stage")
        return False


class Rej727NetSpend(FeatureGroup):
    """Requested consumer of rej727_line_value; inert for every other name."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("rej727_line_value")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rej727_net_spend"}


class Rej727LineValue(FeatureGroup):
    """Middle consumer whose input is the unresolvable rej727_column; inert for every other name."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(FEATURE_NAME_REJ727)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rej727_line_value"}


class TestEngineHarvestsRejection:
    """Engine integration, deliberately WITHOUT a window fixture: the engine owns the per-candidate window."""

    def test_nested_input_names_its_chain_and_drops_the_did_you_mean_hint(self) -> None:
        """An unresolvable input two levels down names the chain and skips the hint beside an input_data decline."""
        with pytest.raises(FeatureResolutionError) as excinfo:
            mloda.prepare(
                ["rej727_net_spend"],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {Rej727ToyFormatFG, Rej727NetSpend, Rej727LineValue}
                ),
                data_access_collection=toy_dac(rej727_handle={"other_a": [1]}),
            )

        lines = str(excinfo.value).split("\n")
        assert lines[0] == (
            f"No feature groups found for feature name: '{FEATURE_NAME_REJ727}' "
            "(needed by rej727_net_spend -> rej727_line_value)."
        )
        assert any(line.startswith(f"  - {Rej727ToyFormatFG.__name__} (input data):") for line in lines)
        assert not any(line.startswith("Did you mean") for line in lines)


class TestEngineFallsBackOnUnknownStage:
    """The neutral channel does not validate stages; the engine maps unknown ones to value_rejection."""

    def test_unknown_recorded_stage_falls_back_to_value_rejection(self) -> None:
        """A custom stage string neither crashes the run nor leaks into the elimination stage."""
        feature = Feature(name=BOGUS_STAGE_FEATURE_REJ727)
        accessible_plugins: FeatureGroupEnvironmentMapping = {Rej727BogusStageFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(Rej727BogusStageFG)
        assert elimination is not None
        assert elimination.stage == "value_rejection"
        assert elimination.reason == BOGUS_STAGE_REASON_REJ727
