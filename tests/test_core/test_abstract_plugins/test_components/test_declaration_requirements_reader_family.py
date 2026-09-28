"""Consumer declaration requirements with reader families: a sibling reader that declares the key wins.

Readers match only their module-unique handle or access string and only the DEPTH name, so they stay inert
for other suites. Load order is logged so plan-time refusal can be proven (nothing loaded).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.declared_attributes import (
    DeclarationRequirement,
    declaration_requirement_scope,
)
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise

DEPTH = "decl_depth_1648"
FRAME = "decl_frame_1648"
LONELY_DEPTH = "decl_lonely_depth_1648"
OUT_GLOBAL = "decl_out_global_1648"
OUT_SCOPED = "decl_out_scoped_1648"
OUT_LONELY_REQUIRING = "decl_out_lonely_requiring_1648"
OUT_LONELY_PLAIN = "decl_out_lonely_plain_1648"

PLAIN_ACCESS = "decl_plain_access_1648"
SCALED_ACCESS = "decl_scaled_access_1648"
LONELY_ACCESS = "decl_lonely_access_1648"
DEPTH_HANDLE = "decl_depth_handle_1648"
LONELY_HANDLE = "decl_lonely_handle_1648"
NAMERULE = "decl_namerule_1648"
NAMERULE_ACCESS = "decl_namerule_access_1648"
NAMERULE_HANDLE = "decl_namerule_handle_1648"
OTHER_HANDLE = "decl_other_handle_1648"

LOAD_LOG: list[str] = []
SHARED_LONELY_INPUT: list[Feature] = []


class _DeclMarkedReader(BaseInputData):
    """Family base: a child matches only its own access string or handle, and only its feature name."""

    ACCESS: ClassVar[str] = ""
    HANDLE: ClassVar[str] = ""
    FEATURE: ClassVar[str] = ""

    @classmethod
    def match_subclass_data_access(
        cls, data_access: Any, feature_names: list[str], options: Options | None = None
    ) -> Any:
        if not cls.ACCESS or cls.FEATURE not in feature_names:
            return None
        if isinstance(data_access, str) and data_access == cls.ACCESS:
            return data_access
        if isinstance(data_access, DataAccessCollection) and cls.HANDLE in data_access.folders:
            return cls.ACCESS
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        LOAD_LOG.append(cls.__name__)
        return {cls.FEATURE: [1]}


class DeclDepthFamily1648(_DeclMarkedReader):
    """Family of the two depth readers."""


class DeclPlainDepthReader1648(DeclDepthFamily1648):
    """Declares nothing (subclass iteration is set-ordered, so it may be probed before or after its sibling)."""

    ACCESS = PLAIN_ACCESS
    HANDLE = DEPTH_HANDLE
    FEATURE = DEPTH


class DeclScaledDepthReader1648(DeclDepthFamily1648):
    """Declares a scale."""

    ACCESS = SCALED_ACCESS
    HANDLE = DEPTH_HANDLE
    FEATURE = DEPTH

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {"scale": 0.001}


class DeclLonelyFamily1648(_DeclMarkedReader):
    """Family with a single reader that declares nothing."""


class DeclLonelyReader1648(DeclLonelyFamily1648):
    """The only provider of the lonely depth."""

    ACCESS = LONELY_ACCESS
    HANDLE = LONELY_HANDLE
    FEATURE = LONELY_DEPTH


class DeclNameRuleFamily1648(_DeclMarkedReader):
    """Family whose only reader declares a scale that differs from its feature group's."""


class DeclNameRuleReader1648(DeclNameRuleFamily1648):
    """Matches the name-rule data but declares scale 1."""

    ACCESS = NAMERULE_ACCESS
    HANDLE = NAMERULE_HANDLE
    FEATURE = NAMERULE

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {"scale": 1}


class DeclNameRuleFG1648(FeatureGroup):
    """Declares scale 5000 and lists its feature name, so the name rule can recover a skipped reader."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {NAMERULE}

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DeclNameRuleFamily1648()

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {"scale": 5000}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {NAMERULE: [1]}


class DeclDerivedDepthFG1648(FeatureGroup):
    """Serves the depth name without a reader and declares nothing."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {DEPTH}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {DEPTH: [1]}


class DeclDepthFG1648(FeatureGroup):
    """Root group resolved through the depth reader family."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DeclDepthFamily1648()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        assert reader is not None
        return reader.load(features)


class DeclLonelyFG1648(DeclDepthFG1648):
    """Root group resolved through the lonely reader family."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DeclLonelyFamily1648()


class DeclFrameFG1648(FeatureGroup):
    """Provides the frame input, which no consumer constrains."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({FRAME})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {FRAME: [1]}


class _DeclConsumer(FeatureGroup):
    """Consumer producing OUT from its inputs; requires scale of DEPTH-like inputs only when REQUIRES."""

    OUT: ClassVar[str] = ""
    REQUIRES: ClassVar[bool] = True

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.OUT} if cls.OUT else set()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def required_input_declarations(cls, input_feature_name: str) -> Mapping[str, str | int | float | bool | None]:
        if cls.REQUIRES and input_feature_name != FRAME:
            return {"scale": None}
        return {}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(DEPTH), Feature(FRAME)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.OUT: [1]}


class DeclDepthToMetresGlobal1648(_DeclConsumer):
    """Requires scale of its depth input, addressed through the global data access collection."""

    OUT = OUT_GLOBAL


class DeclDepthToMetresScoped1648(_DeclConsumer):
    """Requires scale of its depth input, addressed per feature through reader options."""

    OUT = OUT_SCOPED

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        addressed = {DeclPlainDepthReader1648.__name__: PLAIN_ACCESS, DeclScaledDepthReader1648.__name__: SCALED_ACCESS}
        return {Feature(DEPTH, options=addressed), Feature(FRAME)}


class DeclLonelyRequiringConsumer1648(_DeclConsumer):
    """Requires scale of an input only the declaration-less lonely reader can provide."""

    OUT = OUT_LONELY_REQUIRING

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(LONELY_DEPTH)}


class DeclLonelyPlainConsumer1648(DeclLonelyRequiringConsumer1648):
    """Same input, no requirement."""

    OUT = OUT_LONELY_PLAIN
    REQUIRES = False


class DeclSharedInstanceRequiringConsumer1648(DeclLonelyRequiringConsumer1648):
    """Returns the one shared input Feature instance, requiring scale of it."""

    OUT = "decl_out_shared_requiring_1648"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {SHARED_LONELY_INPUT[0]}


class DeclSharedInstancePlainConsumer1648(DeclLonelyRequiringConsumer1648):
    """Returns the one shared input Feature instance, requiring nothing."""

    OUT = "decl_out_shared_plain_1648"
    REQUIRES = False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {SHARED_LONELY_INPUT[0]}


ALL_GROUPS: set[type[FeatureGroup]] = {
    DeclDepthFG1648,
    DeclLonelyFG1648,
    DeclFrameFG1648,
    DeclDepthToMetresGlobal1648,
    DeclDepthToMetresScoped1648,
    DeclLonelyRequiringConsumer1648,
    DeclLonelyPlainConsumer1648,
    DeclSharedInstanceRequiringConsumer1648,
    DeclSharedInstancePlainConsumer1648,
}
ENABLED = PluginCollector.enabled_feature_groups(ALL_GROUPS)
DEPTH_DAC = DataAccessCollection(folders={DEPTH_HANDLE: "/decl/nowhere"})
LONELY_DAC = DataAccessCollection(folders={LONELY_HANDLE: "/decl/nowhere"})
OTHER_DAC = DataAccessCollection(folders={OTHER_HANDLE: "/decl/nowhere"})
NAMERULE_DAC = DataAccessCollection(folders={NAMERULE_HANDLE: "/decl/nowhere"})
DEPTH_MAPPING: FeatureGroupEnvironmentMapping = {DeclDepthFG1648: {PythonDictFramework}}
LONELY_MAPPING: FeatureGroupEnvironmentMapping = {DeclLonelyFG1648: {PythonDictFramework}}
NAMERULE_MAPPING: FeatureGroupEnvironmentMapping = {DeclNameRuleFG1648: {PythonDictFramework}}
READER_FIRST_MAPPING: FeatureGroupEnvironmentMapping = {
    DeclDepthFG1648: {PythonDictFramework},
    DeclDerivedDepthFG1648: {PythonDictFramework},
}
DERIVED_FIRST_MAPPING: FeatureGroupEnvironmentMapping = {
    DeclDerivedDepthFG1648: {PythonDictFramework},
    DeclDepthFG1648: {PythonDictFramework},
}


@pytest.fixture(autouse=True)
def _clear_load_log() -> None:
    LOAD_LOG.clear()
    SHARED_LONELY_INPUT.clear()


def _requiring(feature: Feature) -> Feature:
    feature.declaration_requirement = ("DepthToMetres", {"scale": None})
    return feature


def _scoped_depth() -> Feature:
    addressed = {DeclPlainDepthReader1648.__name__: PLAIN_ACCESS, DeclScaledDepthReader1648.__name__: SCALED_ACCESS}
    return Feature(DEPTH, options=addressed)


def _run(names: list[Feature | str], dac: DataAccessCollection | None) -> Any:
    return mloda.run_all(
        names,
        compute_frameworks={PythonDictFramework},
        parallelization_modes={ParallelizationMode.SYNC},
        data_access_collection=dac,
        plugin_collector=ENABLED,
    )


class TestReaderFamilyResolution:
    """A reader that fails the requirement is skipped so a sibling reader in the family wins."""

    def test_global_path_skips_the_reader_without_the_key(self) -> None:
        feature = _requiring(Feature(DEPTH))

        evaluate_or_raise(feature, DEPTH_MAPPING, data_access_collection=DEPTH_DAC)

        assert feature.options.get("BaseInputData") == (DeclScaledDepthReader1648, SCALED_ACCESS)

    def test_feature_scope_path_skips_the_reader_without_the_key(self) -> None:
        feature = _requiring(_scoped_depth())

        evaluate_or_raise(feature, DEPTH_MAPPING)

        assert feature.options.get("BaseInputData") == (DeclScaledDepthReader1648, SCALED_ACCESS)

    def test_two_requests_of_one_input_only_the_requiring_one_is_steered(self) -> None:
        requiring = _requiring(Feature(DEPTH))
        plain = Feature(DEPTH)

        evaluate_or_raise(requiring, DEPTH_MAPPING, data_access_collection=DEPTH_DAC)
        evaluate_or_raise(plain, DEPTH_MAPPING, data_access_collection=DEPTH_DAC)

        assert requiring.options.get("BaseInputData") == (DeclScaledDepthReader1648, SCALED_ACCESS)
        assert plain.options.get("BaseInputData") in {
            (DeclPlainDepthReader1648, PLAIN_ACCESS),
            (DeclScaledDepthReader1648, SCALED_ACCESS),
        }

    def test_family_without_a_satisfying_reader_is_refused_before_any_load(self) -> None:
        feature = _requiring(Feature(LONELY_DEPTH))

        result = IdentifyFeatureGroupClass.evaluate(feature, LONELY_MAPPING, None, LONELY_DAC)

        assert result.identified == {}
        elimination = result.eliminations[DeclLonelyFG1648]
        assert elimination.stage == "input_data"
        assert "DepthToMetres" in elimination.reason
        assert "'scale'" in elimination.reason
        assert LOAD_LOG == []

    @pytest.mark.parametrize(
        "mapping", [READER_FIRST_MAPPING, DERIVED_FIRST_MAPPING], ids=["reader_first", "derived_first"]
    )
    def test_reader_written_by_one_candidate_does_not_satisfy_another(
        self, mapping: FeatureGroupEnvironmentMapping
    ) -> None:
        feature = _requiring(Feature(DEPTH))

        result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, DEPTH_DAC)

        assert set(result.identified) == {DeclDepthFG1648}
        assert result.eliminations[DeclDerivedDepthFG1648].stage == "declarations"

    def test_name_rule_does_not_recover_a_reader_skipped_for_the_requirement_global(self) -> None:
        feature = Feature(NAMERULE)
        feature.declaration_requirement = ("DepthToMetres", {"scale": 5000})

        with pytest.raises(FeatureResolutionError):
            evaluate_or_raise(feature, NAMERULE_MAPPING, data_access_collection=NAMERULE_DAC)

        assert LOAD_LOG == []

    def test_name_rule_does_not_recover_a_reader_skipped_for_the_requirement_scoped(self) -> None:
        feature = Feature(NAMERULE, options={DeclNameRuleReader1648.__name__: NAMERULE_ACCESS})
        feature.declaration_requirement = ("DepthToMetres", {"scale": 5000})

        with pytest.raises(FeatureResolutionError):
            evaluate_or_raise(feature, NAMERULE_MAPPING)

    def test_global_reader_that_does_not_own_the_data_records_no_declaration_rejection(self) -> None:
        feature = _requiring(Feature(DEPTH))

        result = IdentifyFeatureGroupClass.evaluate(feature, DEPTH_MAPPING, None, OTHER_DAC)

        assert all("requires declared" not in e.reason for e in result.eliminations.values())

    def test_scoped_reader_that_does_not_own_the_data_records_no_declaration_rejection(self) -> None:
        feature = _requiring(Feature(DEPTH, options={DeclPlainDepthReader1648.__name__: "decl_not_my_access_1648"}))

        result = IdentifyFeatureGroupClass.evaluate(feature, DEPTH_MAPPING, None, None)

        assert all("requires declared" not in e.reason for e in result.eliminations.values())

    def test_outer_requirement_scope_does_not_leak_into_a_feature_without_one(self) -> None:
        outer = DeclarationRequirement("OuterConsumer", {"scale": None}, DeclLonelyFG1648, {})
        feature = Feature(LONELY_DEPTH)

        with declaration_requirement_scope(outer):
            result = IdentifyFeatureGroupClass.evaluate(feature, LONELY_MAPPING, None, LONELY_DAC)

        assert set(result.identified) == {DeclLonelyFG1648}


class TestDeclarationRequirementsEndToEnd:
    """The engine assigns the consumer requirement per input and resolution honours it."""

    def test_global_family_sibling_reader_supplies_the_depth(self) -> None:
        result = _run([OUT_GLOBAL], DEPTH_DAC)

        assert result[0][OUT_GLOBAL] == [1]
        assert LOAD_LOG == [DeclScaledDepthReader1648.__name__]

    def test_scoped_family_sibling_reader_supplies_the_depth(self) -> None:
        result = _run([OUT_SCOPED], None)

        assert result[0][OUT_SCOPED] == [1]
        assert LOAD_LOG == [DeclScaledDepthReader1648.__name__]

    def test_unmet_requirement_fails_at_plan_time_before_any_load(self) -> None:
        with pytest.raises(FeatureResolutionError) as excinfo:
            _run([OUT_LONELY_REQUIRING], LONELY_DAC)

        message = str(excinfo.value)
        assert DeclLonelyRequiringConsumer1648.get_class_name() in message
        assert "'scale'" in message
        assert LOAD_LOG == []

    def test_only_the_requiring_consumer_of_a_shared_input_is_refused(self) -> None:
        with pytest.raises(FeatureResolutionError) as excinfo:
            _run([OUT_LONELY_REQUIRING, OUT_LONELY_PLAIN], LONELY_DAC)

        message = str(excinfo.value)
        assert DeclLonelyRequiringConsumer1648.get_class_name() in message
        assert DeclLonelyPlainConsumer1648.get_class_name() not in message
        assert LOAD_LOG == []

    def test_non_requiring_consumer_resolves_the_same_input_normally(self) -> None:
        result = _run([OUT_LONELY_PLAIN], LONELY_DAC)

        assert result[0][OUT_LONELY_PLAIN] == [1]
        assert LOAD_LOG == [DeclLonelyReader1648.__name__]

    def test_reused_feature_instance_does_not_keep_a_stale_requirement(self) -> None:
        SHARED_LONELY_INPUT.append(Feature(LONELY_DEPTH))

        with pytest.raises(FeatureResolutionError):
            _run([DeclSharedInstanceRequiringConsumer1648.OUT], LONELY_DAC)
        result = _run([DeclSharedInstancePlainConsumer1648.OUT], LONELY_DAC)

        assert result[0][DeclSharedInstancePlainConsumer1648.OUT] == [1]
        assert LOAD_LOG == [DeclLonelyReader1648.__name__]
