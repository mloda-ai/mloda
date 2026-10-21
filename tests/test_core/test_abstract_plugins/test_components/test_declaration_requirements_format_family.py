"""Consumer declaration requirements against format groups that declare, or lack, the required key.

Format groups serve only their own columns, so they stay inert for other suites. Load order is logged so
plan-time refusal can be proven (nothing loaded).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.provider import ClaimRoute, ComputeFramework, FeatureGroup, FeatureSet, NamePolicy, SourceMatch
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    JoinSpec,
    Link,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
    resolve_feature,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyFormatBase,
    toy_dac,
)

DEPTH = "declfmt_depth"
FRAME = "declfmt_frame"
LONELY_DEPTH = "declfmt_lonely_depth"
OUT_GLOBAL = "declfmt_out_global"
OUT_SCOPED = "declfmt_out_scoped"
OUT_LONELY_REQUIRING = "declfmt_out_lonely_requiring"
OUT_LONELY_PLAIN = "declfmt_out_lonely_plain"

JOIN_KEY = "declfmt_join_key"

LOAD_LOG: list[str] = []
SHARED_LONELY_INPUT: list[Feature] = []


class _DeclServingFormat(ToyFormatBase):
    """Serves only the names in `served`; logs every load."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("declfmt", NamePolicy.CHECKED, True),)
    served: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        return [column for column in match.access if column in cls.served]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        LOAD_LOG.append(cls.__name__)
        return pa.table({column: values for column, values in match.access.items() if column in cls.served})


class DeclScaledDepthFormat(_DeclServingFormat):
    """Declares a scale."""

    served = frozenset({DEPTH, JOIN_KEY})

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {"scale": 0.001}


class DeclLonelyFormat(_DeclServingFormat):
    """The only provider of the lonely depth; declares nothing."""

    served = frozenset({LONELY_DEPTH})


class DeclFrameFG(FeatureGroup):
    """Provides the frame input, which no consumer constrains."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {FRAME}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({FRAME: [1], JOIN_KEY: [1]})


class _DeclConsumer(FeatureGroup):
    """Consumer producing OUT from its inputs; requires scale of DEPTH-like inputs only when REQUIRES."""

    OUT: ClassVar[str] = ""
    REQUIRES: ClassVar[bool] = True

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.OUT} if cls.OUT else set()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def depth_requirement(cls) -> dict[str, str | int | float | bool | None] | None:
        return {"scale": None} if cls.REQUIRES else None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(DEPTH, required_declarations=self.depth_requirement()), Feature(FRAME)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({cls.OUT: [1]})


class DeclDepthToMetresGlobal(_DeclConsumer):
    """Requires scale of its depth input, addressed through the global data access collection."""

    OUT = OUT_GLOBAL


class DeclDepthToMetresScoped(_DeclConsumer):
    """Requires scale of its depth input, addressed per feature through the format group's option key."""

    OUT = OUT_SCOPED

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        addressed = {DeclScaledDepthFormat.__name__: {DEPTH: [1], JOIN_KEY: [1]}}
        depth = Feature(DEPTH, options=addressed, required_declarations=self.depth_requirement())
        return {depth, Feature(FRAME)}


class DeclLonelyRequiringConsumer(_DeclConsumer):
    """Requires scale of an input only the declaration-less lonely format group can provide."""

    OUT = OUT_LONELY_REQUIRING

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(LONELY_DEPTH, required_declarations=self.depth_requirement())}


class DeclLonelyPlainConsumer(DeclLonelyRequiringConsumer):
    """Same input, no requirement."""

    OUT = OUT_LONELY_PLAIN
    REQUIRES = False


class DeclSharedInstanceRequiringConsumer(DeclLonelyRequiringConsumer):
    """Returns the one shared requiring input Feature instance."""

    OUT = "declfmt_out_shared_requiring"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {SHARED_LONELY_INPUT[0]}


class DeclSharedInstanceSecondConsumer(DeclLonelyRequiringConsumer):
    """Also returns the one shared requiring input Feature instance."""

    OUT = "declfmt_out_shared_second"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {SHARED_LONELY_INPUT[0]}


ALL_GROUPS: set[type[FeatureGroup]] = {
    DeclScaledDepthFormat,
    DeclLonelyFormat,
    DeclFrameFG,
    DeclDepthToMetresGlobal,
    DeclDepthToMetresScoped,
    DeclLonelyRequiringConsumer,
    DeclLonelyPlainConsumer,
    DeclSharedInstanceRequiringConsumer,
    DeclSharedInstanceSecondConsumer,
}
ENABLED = PluginCollector.enabled_feature_groups(ALL_GROUPS)
DEPTH_DAC: DataAccessCollection = toy_dac(declfmt_depth_handle={DEPTH: [1], JOIN_KEY: [1]})
LONELY_DAC: DataAccessCollection = toy_dac(declfmt_lonely_handle={LONELY_DEPTH: [1]})
DEPTH_FRAME_LINK = Link.inner(JoinSpec(DeclScaledDepthFormat, JOIN_KEY), JoinSpec(DeclFrameFG, JOIN_KEY))


@pytest.fixture(autouse=True)
def _clear_load_log() -> None:
    LOAD_LOG.clear()
    SHARED_LONELY_INPUT.clear()


def _run(names: list[Feature | str], dac: DataAccessCollection | None, links: set[Link] | None = None) -> Any:
    return mloda.run_all(
        names,
        compute_frameworks=[PyArrowTable],
        parallelization_modes={ParallelizationMode.SYNC},
        data_access_collection=dac,
        plugin_collector=ENABLED,
        links=links,
    )


class TestDeclarationRequirementsEndToEnd:
    """A consumer's Feature carries the requirement per input and resolution honours it."""

    def test_global_format_group_supplies_the_depth(self) -> None:
        result = _run([OUT_GLOBAL], DEPTH_DAC, {DEPTH_FRAME_LINK})

        assert result[0][OUT_GLOBAL].to_pylist() == [1]
        assert LOAD_LOG == [DeclScaledDepthFormat.__name__]

    def test_scoped_format_group_supplies_the_depth(self) -> None:
        result = _run([OUT_SCOPED], None, {DEPTH_FRAME_LINK})

        assert result[0][OUT_SCOPED].to_pylist() == [1]
        assert LOAD_LOG == [DeclScaledDepthFormat.__name__]

    def test_unmet_requirement_fails_at_plan_time_before_any_load(self) -> None:
        with pytest.raises(FeatureResolutionError) as excinfo:
            _run([OUT_LONELY_REQUIRING], LONELY_DAC)

        message = str(excinfo.value)
        assert DeclLonelyRequiringConsumer.get_class_name() in message
        assert "'scale'" in message
        assert LOAD_LOG == []

    def test_only_the_requiring_consumer_of_a_shared_input_is_refused(self) -> None:
        with pytest.raises(FeatureResolutionError) as excinfo:
            _run([OUT_LONELY_REQUIRING, OUT_LONELY_PLAIN], LONELY_DAC)

        message = str(excinfo.value)
        assert DeclLonelyRequiringConsumer.get_class_name() in message
        assert DeclLonelyPlainConsumer.get_class_name() not in message
        assert LOAD_LOG == []

    def test_non_requiring_consumer_resolves_the_same_input_normally(self) -> None:
        result = _run([OUT_LONELY_PLAIN], LONELY_DAC)

        assert result[0][OUT_LONELY_PLAIN].to_pylist() == [1]
        assert LOAD_LOG == [DeclLonelyFormat.__name__]

    @pytest.mark.parametrize(
        "consumers",
        [
            (DeclSharedInstanceRequiringConsumer, DeclSharedInstanceSecondConsumer),
            (DeclSharedInstanceSecondConsumer, DeclSharedInstanceRequiringConsumer),
        ],
        ids=["a_then_b", "b_then_a"],
    )
    def test_shared_requiring_instance_is_refused_for_every_consumer_that_returns_it(
        self, consumers: tuple[type[_DeclConsumer], type[_DeclConsumer]]
    ) -> None:
        SHARED_LONELY_INPUT.append(Feature(LONELY_DEPTH, required_declarations={"scale": None}))

        for consumer, other in (consumers, consumers[::-1]):
            with pytest.raises(FeatureResolutionError) as excinfo:
                _run([consumer.OUT], LONELY_DAC)
            message = str(excinfo.value)
            assert f"{consumer.get_class_name()} requires declared 'scale'" in message
            assert other.get_class_name() not in message
        assert LOAD_LOG == []

    def test_shared_instance_requested_top_level_after_a_consumer_run_names_no_consumer(self) -> None:
        shared = Feature(LONELY_DEPTH, required_declarations={"scale": None})
        SHARED_LONELY_INPUT.append(shared)
        with pytest.raises(FeatureResolutionError):
            _run([DeclSharedInstanceRequiringConsumer.OUT], LONELY_DAC)

        with pytest.raises(FeatureResolutionError) as excinfo:
            _run([shared], LONELY_DAC)

        message = str(excinfo.value)
        assert f"request for '{LONELY_DEPTH}'" in message
        assert DeclSharedInstanceRequiringConsumer.get_class_name() not in message
        assert DeclSharedInstanceSecondConsumer.get_class_name() not in message
        assert "needed by" not in message
        assert LOAD_LOG == []

        resolved = resolve_feature(
            shared,
            plugin_collector=ENABLED,
            data_access_collection=LONELY_DAC,
            compute_frameworks={PyArrowTable},
        )

        assert resolved.error is not None
        assert f"request for '{LONELY_DEPTH}'" in resolved.error
        assert "needed by" not in resolved.error
        assert DeclSharedInstanceRequiringConsumer.get_class_name() not in resolved.error


class TestTopLevelRequestRequirement:
    """A requiring Feature passed straight to run_all is checked like a consumer's input."""

    def test_unmet_requirement_is_refused_before_any_load(self) -> None:
        request = Feature(LONELY_DEPTH, required_declarations={"scale": None})

        with pytest.raises(FeatureResolutionError) as excinfo:
            _run([request], LONELY_DAC)

        message = str(excinfo.value)
        assert f"request for '{LONELY_DEPTH}'" in message
        assert "'scale'" in message
        assert LOAD_LOG == []

    def test_met_requirement_resolves_through_the_declaring_format_group(self) -> None:
        request = Feature(DEPTH, required_declarations={"scale": None})

        result = _run([request], DEPTH_DAC)

        assert result[0][DEPTH].to_pylist() == [1]
        assert LOAD_LOG == [DeclScaledDepthFormat.__name__]
