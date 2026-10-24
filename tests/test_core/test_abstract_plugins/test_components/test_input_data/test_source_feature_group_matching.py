"""SourceFeatureGroup class-definition rule, claim-route matching, pointing, ambiguity and plan identity."""

import gc
import logging
from abc import abstractmethod
from collections.abc import Collection
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.core.abstract_plugins.components.input_data.claim_route import feature_group_scope
from mloda.core.abstract_plugins.components.input_data.source_feature_group import (
    SourcePointerError,
)
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection, collect_match_rejections
from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import ApiDataFG, ClaimRoute, SourceFeatureGroup, NamePolicy, ReadFileFG, SourceMatch
from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregation
from tests.mixins.reader_feature_groups.format_file_writers import write_csv, write_parquet
from tests.mixins.reader_feature_groups.lazy_source_group import load_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    ToyDeclaredFG,
    ToySourceBase,
    ToySourceFG,
    ToySourceSubFG,
    ToyOpenFG,
    ToyOtherSourceFG,
    ToyRequiredOptionFG,
    column_values,
    foreign_dac,
    toy_dac,
)
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise

COL = "toyfmt_col"
_AGGREGATION_IMPLEMENTATIONS = (PyArrowAggregation,)  # loaded by import, serves AggregatedFeatureGroup on PyArrow


def _plugins(*groups: type) -> FeatureGroupEnvironmentMapping:
    return {g: {PyArrowTable} for g in groups}


def _identify(feature: Feature, dac: DataAccessCollection | None, *groups: type) -> EvaluationResult:
    return IdentifyFeatureGroupClass.evaluate(feature, _plugins(*groups), None, dac)


def _claims(feature: Feature, dac: DataAccessCollection | None, group: type) -> bool:
    return group in _identify(feature, dac, group).identified


class _BadSpecMixin:
    """Plain mixin whose PROPERTY_MAPPING holds a non-PropertySpec value."""

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {"toyfmt_bad": 42}


class TestClassDefinitionRule:
    def test_open_searched_route_raises(self) -> None:
        with pytest.raises(TypeError):

            class _Bad(ToySourceBase):
                CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.OPEN, True),)

    def test_open_searched_route_among_valid_routes_raises(self) -> None:
        with pytest.raises(TypeError):

            class _BadMixed(ToySourceBase):
                CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
                    ClaimRoute("toy", NamePolicy.CHECKED, True),
                    ClaimRoute("toy2", NamePolicy.OPEN, True),
                )

    def test_group_without_routes_never_claims(self) -> None:
        assert not _claims(Feature(COL), toy_dac(h1={COL: [1]}), ToySourceBase)

    @pytest.mark.parametrize("own_mapping", [False, True], ids=["inherited_only", "own_mapping_merged"])
    def test_a_non_spec_value_from_a_plain_mixin_raises_at_definition(self, own_mapping: bool) -> None:
        attributes: dict[str, Any] = {"PROPERTY_MAPPING": {"toyfmt_ok": PropertySpec("ok", default=None)}}
        with pytest.raises(ValueError) as exc_info:
            type("ToyfmtBadMixinReadFile", (_BadSpecMixin, ReadFileFG), attributes if own_mapping else {})
        message = str(exc_info.value)
        del exc_info
        gc.collect()

        assert "ToyfmtBadMixinReadFile.PROPERTY_MAPPING['toyfmt_bad'] is a int, not a PropertySpec" in message


class TestCheckedNames:
    def test_present_column_matches_and_records_pair(self) -> None:
        feature = Feature(COL)
        result = _identify(feature, toy_dac(h1={COL: [1]}), ToySourceFG)

        assert ToySourceFG in result.identified
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is ToySourceFG
        assert pair[1] == SourceMatch(source="h1:toy", access=None)
        assert pair[1].access == {COL: [1]}

    def test_missing_column_declines_with_input_data_rejection(self) -> None:
        result = _identify(Feature(COL), toy_dac(h1={"other": [1]}), ToySourceFG)

        assert result.identified == {}
        elimination = result.eliminations[ToySourceFG]
        assert elimination.stage == "input_data"
        assert "h1:toy" in elimination.reason

    def test_no_collection_declines(self) -> None:
        assert not _claims(Feature(COL), None, ToySourceFG)


class TestDeclaredNames:
    @pytest.mark.parametrize("name", ["ToyDeclaredFG", "toyfmt_declared_col", "toyfmt_pre_anything"])
    def test_declared_name_with_a_source_matches(self, name: str) -> None:
        assert _claims(Feature(name), toy_dac(h1={"x": [1]}), ToyDeclaredFG)

    @pytest.mark.parametrize("name", ["ToyDeclaredFG", "toyfmt_declared_col", "toyfmt_pre_anything"])
    def test_declared_name_without_a_source_declines(self, name: str) -> None:
        assert not _claims(Feature(name), None, ToyDeclaredFG)
        assert not _claims(Feature(name), foreign_dac(), ToyDeclaredFG)

    def test_undeclared_name_declines_even_with_a_source(self) -> None:
        assert not _claims(Feature("toyfmt_undeclared"), toy_dac(h1={"toyfmt_undeclared": [1]}), ToyDeclaredFG)


class TestOpenNames:
    def test_open_group_never_claims_unpointed(self) -> None:
        assert not _claims(Feature("toyfmt_any"), toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_pointed_by_class_name_option_key(self) -> None:
        feature = Feature("toyfmt_any", Options({"ToyOpenFG": {COL: [1]}}))
        assert _claims(feature, None, ToyOpenFG)

    def test_data_access_handle_alone_does_not_unlock_open_names(self) -> None:
        feature = Feature("toyfmt_any", Options({"data_access_handle": "h1"}))
        assert not _claims(feature, toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_pointed_by_feature_group_scope(self) -> None:
        feature = Feature("toyfmt_any", feature_group=ToyOpenFG)
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_scope_by_name_string_points_too(self) -> None:
        feature = Feature("toyfmt_any", feature_group="ToyOpenFG")
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_scope_does_not_point_other_groups_candidates_when_out_of_scope(self) -> None:
        feature = Feature("toyfmt_any", feature_group=ToyOpenFG)
        result = _identify(feature, toy_dac(h1={COL: [1]}), ToySourceFG, ToyOpenFG)
        assert ToySourceFG not in result.identified


class TestRequiredOptionRoute:
    def test_declines_without_the_required_option(self) -> None:
        assert not _claims(Feature(COL), toy_dac(h1={COL: [1]}), ToyRequiredOptionFG)

    def test_required_option_presence_points_the_route(self) -> None:
        feature = Feature(COL, Options({"toyfmt_required": "yes"}))
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyRequiredOptionFG)


class _ToyAliasFG(ToySourceFG):
    """Source group whose data_access_name() is an alias distinct from its class name."""

    @classmethod
    def data_access_name(cls) -> str:
        return "toyfmt_alias"


class TestAliasedGroupPointing:
    @pytest.mark.parametrize(
        ("key", "pointed"),
        [("_ToyAliasFG", True), (_ToyAliasFG, True), ("toyfmt_alias", False)],
        ids=["class_name_string", "class_object", "alias_string"],
    )
    def test_only_the_class_name_forms_point_an_aliasing_group(self, key: Any, pointed: bool) -> None:
        feature = Feature(COL, Options({key: {COL: [1]}}))

        assert _claims(feature, None, _ToyAliasFG) is pointed
        assert _ToyAliasFG.is_pointed(COL, feature.options, None) is pointed


class TestPointedButMissingAborts:
    def test_class_name_key_source_lacking_the_column_raises(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"other": [1]}}))
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToySourceFG), None, None)
        message = str(exc_info.value)
        assert "ToySourceFG" in message
        assert "scoped:toy" in message
        assert COL in message
        assert "other" in message
        error = exc_info.value
        assert isinstance(error, ValueError)
        assert error.feature_name == COL
        assert error.result.identified == {}
        elimination = error.result.eliminations[ToySourceFG]
        assert elimination.stage == "input_data"
        assert elimination.reason in message

    def test_pointed_missing_column_through_run_all_raises_feature_resolution_error(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"other": [1]}}))
        with pytest.raises(FeatureResolutionError, match="ToySourceFG"):
            mloda.run_all(
                [feature],
                compute_frameworks=[PyArrowTable],
                plugin_collector=PluginCollector.enabled_feature_groups({ToySourceFG}),
            )

    def test_pointed_missing_column_is_projected_by_diagnose(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"other": [1]}}))
        diagnosis = mloda.diagnose(
            [feature],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({ToySourceFG}),
        )
        assert diagnosis.complete is False
        assert diagnosis.feature_name == COL
        assert diagnosis.message is not None
        assert "ToySourceFG" in diagnosis.message
        assert "other" in diagnosis.message

    def test_handle_alone_narrows_but_does_not_escalate_a_missing_column(self) -> None:
        feature = Feature(COL, Options({"data_access_handle": "h1"}))
        dac = toy_dac(h1={"other": [1]}, h2={COL: [1]})
        result = _identify(feature, dac, ToySourceFG)
        assert result.identified == {}
        assert "h1:toy" in result.eliminations[ToySourceFG].reason

    def test_abort_does_not_fall_through_to_the_other_candidate(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"other": [1]}}))
        dac = toy_dac(h1={COL: [1]})
        assert _claims(Feature(COL), dac, ToyOtherSourceFG)
        with pytest.raises(FeatureResolutionError, match="ToySourceFG"):
            evaluate_or_raise(feature, _plugins(ToySourceFG, ToyOtherSourceFG), None, dac)


_VALUE_KEY = "toyfmt_mode"
_BAD_VALUE = "toyfmt_bad_mode"


class _ValueCheckedFG(ToySourceFG):
    """A pointed-at source group whose later PROPERTY_MAPPING value check can fail."""

    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        _VALUE_KEY: PropertySpec("Mode", allowed_values={"good": "Good mode"}, context=True, strict_validation=True),
    }


class _PlainRootColFG(FeatureGroup):
    """Plain root (non-source) group claiming COL by its own rules."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {COL}

    @classmethod
    def input_features(cls, options: Options, feature_name: FeatureName) -> None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


class TestPointedGroupValueFailureIsNotADecline:
    def test_later_value_failure_is_reported_not_a_source_decline(self) -> None:
        feature = Feature(COL, Options({"_ValueCheckedFG": {COL: [1]}, _VALUE_KEY: _BAD_VALUE}))
        result = _identify(feature, None, _ValueCheckedFG)

        assert result.identified == {}
        elimination = result.eliminations[_ValueCheckedFG]
        assert elimination.stage == "value_rejection"
        assert f"'{_BAD_VALUE}' not found in mapping for '{_VALUE_KEY}'" in elimination.reason
        assert "matched nothing" not in elimination.reason

    def test_a_valid_value_still_claims(self) -> None:
        feature = Feature(COL, Options({"_ValueCheckedFG": {COL: [1]}, _VALUE_KEY: "good"}))

        assert _claims(feature, None, _ValueCheckedFG)

    def test_plain_root_group_wins_when_the_pointed_group_vetoes_a_value(self) -> None:
        """A pointed decline is not final: a plain root group claiming the name wins."""
        options = Options({"_ValueCheckedFG": {COL: [1]}, _VALUE_KEY: _BAD_VALUE})
        result = _identify(Feature(COL, options), None, _ValueCheckedFG, _PlainRootColFG)

        assert set(result.identified) == {_PlainRootColFG}
        assert _ValueCheckedFG in result.eliminations


_REQ_KEY = "toyfmt_req"
_COND_KEY = "toyfmt_cond"
_EQ_KEY = "toyfmt_eq"
_EQ_SECRET = "toyfmt-s3cr3t"  # nosec B105


class _ReqMixin:
    """Plain mixin holding a required key and a required_when key."""

    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        _REQ_KEY: PropertySpec("Required"),
        _COND_KEY: PropertySpec("Conditional", default=None, required_when=lambda options: True),
    }


class _ReqViaMixinFG(_ReqMixin, ToySourceFG):
    """Declares its own mapping, so the mixin keys reach it through the merge."""

    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {"toyfmt_opt": PropertySpec("Optional", default=None)}


class _ElementValidatedFG(ToySourceFG):
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        _VALUE_KEY: PropertySpec(
            "Mode",
            allowed_values={"a": "A"},
            context=True,
            strict_validation=True,
            element_validator=lambda value: value == "b",
        ),
    }


class _RaisingEqValue:
    """Hashable value whose equality check raises with a secret in the message."""

    def __init__(self, error: type[Exception]) -> None:
        self.error = error

    def __eq__(self, other: object) -> bool:
        raise self.error(f"leak {_EQ_SECRET}")

    def __hash__(self) -> int:
        return 1


def _eq_group(name: str, allowed: tuple[Any, ...]) -> type:
    spec = PropertySpec("Eq", allowed_values=allowed, context=True, strict_validation=True)
    return type(name, (ToySourceFG,), {"PROPERTY_MAPPING": {_EQ_KEY: spec}})


_EQ_ELEMENT_GROUPS: dict[type[Exception], type] = {
    ValueError: _eq_group("_EqElementValueErrorFG", (_RaisingEqValue(ValueError),)),
    RuntimeError: _eq_group("_EqElementRuntimeErrorFG", (_RaisingEqValue(RuntimeError),)),
}
_EqValueFG = _eq_group("_EqValueFG", ("a",))


class TestPropertyMappingMatching:
    @pytest.mark.parametrize("absent", [_REQ_KEY, _COND_KEY])
    def test_mixin_keys_decline_when_absent_and_claim_when_present(self, absent: str) -> None:
        keys = {_REQ_KEY: "x", _COND_KEY: "y"}
        pointer = {"_ReqViaMixinFG": {COL: [1]}}

        assert _claims(Feature(COL, Options({**pointer, **keys})), None, _ReqViaMixinFG)

        del keys[absent]
        result = _identify(Feature(COL, Options({**pointer, **keys})), None, _ReqViaMixinFG)
        assert result.identified == {}
        elimination = result.eliminations[_ReqViaMixinFG]
        assert elimination.stage == "value_rejection"
        assert absent in elimination.reason

    def test_a_value_the_element_validator_accepts_claims_despite_the_allowed_values(self) -> None:
        feature = Feature(COL, Options({"_ElementValidatedFG": {COL: [1]}, _VALUE_KEY: "b"}))

        assert _claims(feature, None, _ElementValidatedFG)

    def test_a_value_the_element_validator_rejects_is_a_value_rejection(self) -> None:
        feature = Feature(COL, Options({"_ElementValidatedFG": {COL: [1]}, _VALUE_KEY: "a"}))
        result = _identify(feature, None, _ElementValidatedFG)

        assert result.identified == {}
        assert result.eliminations[_ElementValidatedFG].stage == "value_rejection"

    @pytest.mark.parametrize("error", [ValueError, RuntimeError])
    @pytest.mark.parametrize("direction", ["element_raises", "value_raises"])
    def test_a_raising_eq_in_the_membership_check_is_a_clean_rejection(
        self, error: type[Exception], direction: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        group, value = (
            (_EQ_ELEMENT_GROUPS[error], "a") if direction == "element_raises" else (_EqValueFG, _RaisingEqValue(error))
        )
        feature = Feature(COL, Options({group.__name__: {COL: [1]}, _EQ_KEY: value}))

        caplog.set_level(logging.DEBUG)
        result = _identify(feature, None, group)

        assert result.identified == {}
        elimination = result.eliminations[group]
        assert elimination.stage == "value_rejection"
        assert _EQ_SECRET not in elimination.reason
        assert not any(_EQ_SECRET in record.getMessage() for record in caplog.records)
        if error is RuntimeError:
            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any("RuntimeError" in m and _EQ_KEY in m for m in warnings)


class TestTwoPointedGroups:
    OPTIONS = {"_ValueCheckedFG": {COL: [1]}, "ToyOtherSourceFG": {COL: [2]}}

    @pytest.mark.parametrize(
        "order",
        [(_ValueCheckedFG, ToyOtherSourceFG), (ToyOtherSourceFG, _ValueCheckedFG)],
        ids=["vetoed_first", "vetoed_last"],
    )
    def test_the_vetoed_group_leaves_the_other_groups_pair_recorded(self, order: tuple[type, type]) -> None:
        feature = Feature(COL, Options({**self.OPTIONS, _VALUE_KEY: _BAD_VALUE}))

        result = _identify(feature, None, *order)

        assert set(result.identified) == {ToyOtherSourceFG}
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is ToyOtherSourceFG
        assert pair[1].access == {COL: [2]}

    def test_two_valid_pointed_groups_are_ambiguous(self) -> None:
        feature = Feature(COL, Options({**self.OPTIONS, _VALUE_KEY: "good"}))

        with pytest.raises(ValueError, match="Multiple feature groups") as exc_info:
            evaluate_or_raise(feature, _plugins(_ValueCheckedFG, ToyOtherSourceFG), None, None)
        assert "_ValueCheckedFG" in str(exc_info.value)
        assert "ToyOtherSourceFG" in str(exc_info.value)


class TestPointedMissingColumnText:
    def test_fix_text_is_one_sentence_without_the_ambiguity_fix(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"other": [1]}}))
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToySourceFG), None, None)
        message = str(exc_info.value)
        assert "column_to_file" not in message
        assert message.count(" or ") == 1

    def test_long_column_list_is_truncated_to_twenty_names(self) -> None:
        columns = {f"toyfmt_c{i:02d}": [i] for i in range(25)}
        feature = Feature(COL, Options({"ToySourceFG": columns}))
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToySourceFG), None, None)
        message = str(exc_info.value)
        assert "toyfmt_c00" in message
        assert "toyfmt_c19" in message
        assert "toyfmt_c20" not in message
        assert "... and 5 more" in message


class TestBoundedScrubbedMissingColumnText:
    def test_unpointed_sources_beyond_five_are_counted_not_named(self) -> None:
        dac = toy_dac(**{f"h{i}": {"other": [i]} for i in range(1, 8)})
        reason = _identify(Feature(COL), dac, ToySourceFG).eliminations[ToySourceFG].reason
        assert "h5:toy" in reason
        assert "h6:toy" not in reason
        assert "h7:toy" not in reason
        assert "and 2 more" in reason

    def test_pointed_column_names_are_scrubbed_before_they_are_listed(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"token=toyfmt-secret-1234": [1]}}))
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToySourceFG), None, None)
        assert "toyfmt-secret-1234" not in str(exc_info.value)

    def test_pointed_column_names_are_clipped(self) -> None:
        long_name = "toyfmt_long_" + "x" * 100
        feature = Feature(COL, Options({"ToySourceFG": {long_name: [1]}}))
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToySourceFG), None, None)
        assert long_name not in str(exc_info.value)
        assert long_name[:40] in str(exc_info.value)

    def test_the_several_sources_abort_is_scrubbed(self) -> None:
        dac = toy_dac(h1={COL: [1]}, h2={COL: [2]})
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(Feature(COL), _plugins(_SecretFixTextFG), None, dac)
        message = str(exc_info.value)
        assert "h1:toy" in message
        assert "toyfmt-secret-1234" not in message


class _SecretFixTextFG(ToySourceFG):
    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        return "retry with token=toyfmt-secret-1234."


class TestAmbiguousSources:
    def test_two_fitting_sources_raise_naming_both_and_the_fix(self) -> None:
        dac = toy_dac(h1={COL: [1]}, h2={COL: [2]})
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(Feature(COL), _plugins(ToySourceFG), None, dac)
        message = str(exc_info.value)
        assert "ToySourceFG" in message
        assert "h1:toy" in message
        assert "h2:toy" in message
        assert "feature_group=" not in message
        assert "data_access_handle" in message
        assert "column_to_file" in message

    def test_data_access_handle_resolves_the_ambiguity(self) -> None:
        feature = Feature(COL, Options({"data_access_handle": "h2"}))
        evaluate_or_raise(feature, _plugins(ToySourceFG), None, toy_dac(h1={COL: [1]}, h2={COL: [2]}))
        pair = feature.input_data_match
        assert pair is not None
        assert pair[1] == SourceMatch(source="h2:toy", access=None)


class TestSubclassTakeover:
    def test_subclass_wins_and_records_its_own_pair(self) -> None:
        feature = Feature(COL, feature_group=ToySourceFG)
        result = evaluate_or_raise(feature, _plugins(ToySourceFG, ToySourceSubFG), None, toy_dac(h1={COL: [1]}))

        assert next(iter(result.identified)) is ToySourceSubFG
        assert result.specialized_from == (ToySourceFG,)
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is ToySourceSubFG
        assert pair[1] == SourceMatch(source="h1:toy", access=None)


class TestPlanIdentity:
    def test_two_sources_of_one_group_make_two_steps_naming_group_and_source(self) -> None:
        dac = toy_dac(h1={"toyfmt_a": [1]}, h2={"toyfmt_b": [2]})
        session = mloda.prepare(
            [Feature("toyfmt_a"), Feature("toyfmt_b")],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({ToySourceFG}),
            data_access_collection=dac,
        )

        steps = [s for s in session.resolved_plan() if s.step_kind == "compute"]

        assert len(steps) == 2
        assert {s.feature_group for s in steps} == {ToySourceFG}
        assert {s.data_access_identity for s in steps} == {"h1:toy", "h2:toy"}
        assert {s.feature_names for s in steps} == {("toyfmt_a",), ("toyfmt_b",)}

    def test_two_features_of_one_source_share_one_step(self) -> None:
        dac = toy_dac(h1={"toyfmt_a": [1], "toyfmt_b": [2]})
        session = mloda.prepare(
            [Feature("toyfmt_a"), Feature("toyfmt_b")],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({ToySourceFG}),
            data_access_collection=dac,
        )

        steps = [s for s in session.resolved_plan() if s.step_kind == "compute"]

        assert len(steps) == 1
        assert steps[0].data_access_identity == "h1:toy"


class _ToyUnservedFramework(ComputeFramework):
    """Uniquely named framework no candidate serves, so a pin on it eliminates every candidate."""


class _TakeoverChildFG(ToySourceFG):
    """Reads its source through the parent's pointer key and finds the column that source lacks."""

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        for key in cls.pointer_keys():
            scoped = options.get(key)
            if scoped is not None:
                return [SourceMatch(source="scoped:toy", access=scoped)]
        return []

    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool | None:
        return column == COL


class _RaisingInputsFG(FeatureGroup):
    """Criteria-matches one name but its input_features raises."""

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
    ) -> bool:
        return str(feature_name) == "toyfmt_typo_probe"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        raise ValueError("input_features failed")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


class _ComputingColFG(FeatureGroup):
    """Computing (non-root) candidate for COL."""

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
    ) -> bool:
        return str(feature_name) == COL

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("toyfmt_input")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


class _AbstractToyMid(ToySourceBase):
    """Abstract shared base: naming it as a scope must not point at its subclasses."""

    @classmethod
    @abstractmethod
    def marker(cls) -> str: ...


class _MidHasColumn(_AbstractToyMid):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)

    @classmethod
    def marker(cls) -> str:
        return "a"


class _MidLacksColumn(_AbstractToyMid):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        return ["toyfmt_other"]

    @classmethod
    def marker(cls) -> str:
        return "b"


class TestScopePointingRefinement:
    @pytest.mark.parametrize("scope", [_AbstractToyMid, "_AbstractToyMid"])
    def test_abstract_base_scope_does_not_point_so_the_lacking_sibling_declines(self, scope: object) -> None:
        feature = Feature(COL, feature_group=scope)  # type: ignore[arg-type]
        result = evaluate_or_raise(feature, _plugins(_MidHasColumn, _MidLacksColumn), None, toy_dac(h1={COL: [1]}))

        assert set(result.identified) == {_MidHasColumn}

    def test_concrete_ancestor_scope_still_points_and_aborts_on_missing_column(self) -> None:
        feature = Feature(COL, feature_group=ToySourceFG)
        with pytest.raises(FeatureResolutionError, match="h1:toy"):
            evaluate_or_raise(feature, _plugins(ToySourceSubFG), None, toy_dac(h1={"other": [1]}))

    def test_scope_naming_the_candidate_itself_points(self) -> None:
        feature = Feature(COL, feature_group=_MidLacksColumn)
        with pytest.raises(FeatureResolutionError, match="h1:toy"):
            evaluate_or_raise(feature, _plugins(_MidLacksColumn), None, toy_dac(h1={COL: [1]}))


class TestPointingIsExplicitOnly:
    """Group options reach derived features' inputs: only the class-name key and a feature_group= scope point."""

    DERIVED = "toyfmt_col__sum_aggr"

    @pytest.mark.parametrize("fg", [ToySourceFG, ToyOpenFG])
    def test_handle_on_a_derived_feature_resolves_to_the_aggregation(self, fg: type) -> None:
        feature = Feature(self.DERIVED, Options(group={"data_access_handle": "h1"}))
        result = evaluate_or_raise(feature, _plugins(fg, AggregatedFeatureGroup), None, toy_dac(h1={COL: [1, 2]}))

        assert set(result.identified) == {AggregatedFeatureGroup}

    def test_required_option_route_does_not_escalate_a_missing_column(self) -> None:
        feature = Feature(self.DERIVED, Options({"toyfmt_required": "yes"}))
        result = evaluate_or_raise(
            feature,
            _plugins(ToyRequiredOptionFG, AggregatedFeatureGroup),
            None,
            toy_dac(h1={COL: [1]}),
        )

        assert set(result.identified) == {AggregatedFeatureGroup}


_DERIVED = "toyfmt_col__sum_aggr"


class _PlainClaimsDerived(FeatureGroup):
    """Non-source survivor claiming the derived name."""

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
    ) -> bool:
        return str(feature_name) == _DERIVED

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


class TestPointedGroupDefersToAComputingCandidate:
    DERIVED = _DERIVED

    @pytest.mark.parametrize("fg", [ToySourceFG, ToyRequiredOptionFG])
    def test_class_name_key_on_a_derived_feature_resolves_to_the_aggregation(self, fg: type) -> None:
        feature = Feature(self.DERIVED, Options({fg.__name__: {COL: [1, 2]}, "toyfmt_required": "yes"}))
        result = evaluate_or_raise(feature, _plugins(fg, AggregatedFeatureGroup), None, None)

        assert set(result.identified) == {AggregatedFeatureGroup}
        elimination = result.eliminations[fg]
        assert elimination.stage == "input_data"
        assert "is in none of the sources" in elimination.reason

    @pytest.mark.parametrize(
        ("name", "groups", "dac"),
        [
            ("toyfmt_unclaimed", (ToySourceFG, AggregatedFeatureGroup), None),
            (_DERIVED, (ToySourceFG, _PlainClaimsDerived, AggregatedFeatureGroup), None),
        ],
        ids=["no_other_claimant", "another_reader_survives"],
    )
    def test_pointed_reader_still_aborts(
        self, name: str, groups: tuple[type, ...], dac: DataAccessCollection | None
    ) -> None:
        feature = Feature(name, Options({"ToySourceFG": {COL: [1]}, "toyfmt_required": "yes"}))
        with pytest.raises(FeatureResolutionError, match=rf"column '{name}' is in none of the sources of ToySourceFG"):
            evaluate_or_raise(feature, _plugins(*groups), None, dac)

    def test_computing_candidate_eliminated_later_still_counts_as_computed(self) -> None:
        feature = Feature(
            self.DERIVED,
            Options({"ToySourceFG": {COL: [1, 2]}}),
            compute_framework=_ToyUnservedFramework.get_class_name(),
        )
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToySourceFG, AggregatedFeatureGroup), None, None)

        result = exc_info.value.result
        assert result.identified == {}
        assert result.eliminations[AggregatedFeatureGroup].stage == "framework_pin"

    def test_pointed_reader_aborts_when_the_other_claimant_input_features_raises(self) -> None:
        feature = Feature("toyfmt_typo_probe", Options({"ToySourceFG": {COL: [1]}}))
        with pytest.raises(
            FeatureResolutionError, match="column 'toyfmt_typo_probe' is in none of the sources of ToySourceFG"
        ):
            evaluate_or_raise(feature, _plugins(ToySourceFG, _RaisingInputsFG), None, None)

    def test_strictness_holds_per_deferring_group_despite_another_groups_takeover(self) -> None:
        options = Options({"ToySourceFG": {"other": [1]}, "ToyOtherSourceFG": {"other": [1]}})
        feature = Feature(COL, options, compute_framework=PyArrowTable.get_class_name())
        plugins: FeatureGroupEnvironmentMapping = {
            ToySourceFG: {PyArrowTable},
            ToyOtherSourceFG: {PyArrowTable},
            _TakeoverChildFG: {PyArrowTable},
            _ComputingColFG: {_ToyUnservedFramework},
        }
        with pytest.raises(FeatureResolutionError, match="is in none of the sources of ToyOtherSourceFG"):
            evaluate_or_raise(feature, plugins, None, None)

    def test_pointed_parent_defers_to_its_surviving_subclass(self) -> None:
        feature = Feature(COL, Options({"ToySourceFG": {"other": [1]}}))
        result = evaluate_or_raise(feature, _plugins(ToySourceFG, _TakeoverChildFG), None, None)

        assert set(result.identified) == {_TakeoverChildFG}
        assert result.eliminations[ToySourceFG].stage == "input_data"


class _OpenDeclaringDerivedFG(ToyOpenFG):
    """Open route that also declares the derived name."""

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        return frozenset({_DERIVED})


class TestOpenClaimDefersToAComputingCandidate:
    DERIVED = _DERIVED

    def test_open_group_declaring_the_derived_name_does_not_step_aside(self) -> None:
        feature = Feature(_DERIVED, Options({"_OpenDeclaringDerivedFG": {COL: [1, 2]}}))
        with pytest.raises(ValueError, match="Multiple feature groups found"):
            evaluate_or_raise(feature, _plugins(_OpenDeclaringDerivedFG, AggregatedFeatureGroup), None, None)

    def _open_feature(self, name: str, **kwargs: Any) -> Feature:
        return Feature(name, Options({"ToyOpenFG": {COL: [1, 2]}}), **kwargs)

    @pytest.mark.parametrize("name", [_DERIVED, "toyfmt_col__sum_aggr__max_aggr"], ids=["derived", "chained"])
    def test_pointed_open_group_steps_aside_for_the_aggregation(self, name: str) -> None:
        result = evaluate_or_raise(self._open_feature(name), _plugins(ToyOpenFG, AggregatedFeatureGroup), None, None)

        assert set(result.identified) == {AggregatedFeatureGroup}
        elimination = result.eliminations[ToyOpenFG]
        assert elimination.stage == "input_data"
        assert "AggregatedFeatureGroup" in elimination.reason
        assert "feature_group=" in elimination.reason
        assert "ToyOpenFG" in elimination.reason

    def test_computing_candidate_eliminated_later_still_masks_the_open_claim(self) -> None:
        feature = self._open_feature(self.DERIVED, compute_framework=_ToyUnservedFramework.get_class_name())
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToyOpenFG, AggregatedFeatureGroup), None, None)

        result = exc_info.value.result
        assert result.identified == {}
        assert result.eliminations[AggregatedFeatureGroup].stage == "framework_pin"

    def test_feature_group_scope_lets_the_open_group_claim_the_derived_name(self) -> None:
        feature = self._open_feature(self.DERIVED, feature_group=ToyOpenFG)
        result = evaluate_or_raise(feature, _plugins(ToyOpenFG, AggregatedFeatureGroup), None, None)

        assert set(result.identified) == {ToyOpenFG}

    def test_another_plain_claimant_still_makes_the_name_ambiguous(self) -> None:
        groups = (ToyOpenFG, _PlainClaimsDerived, AggregatedFeatureGroup)
        with pytest.raises(ValueError, match="Multiple feature groups found"):
            evaluate_or_raise(self._open_feature(self.DERIVED), _plugins(*groups), None, None)

    def test_open_route_on_a_leaf_name_still_claims(self) -> None:
        result = evaluate_or_raise(self._open_feature(COL), _plugins(ToyOpenFG, AggregatedFeatureGroup), None, None)

        assert set(result.identified) == {ToyOpenFG}


_PART_BASE = "toyfmt_pbase"
_PART_NAME = _PART_BASE + "~value"


class _PartOnlyFG(FeatureGroup):
    """Matches only the base name; the whole name is read as an output part."""

    OUTPUT_PARTS = ("value",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_PART_BASE}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


class TestSearchedRouteDefersToAComputingCandidate:
    @staticmethod
    def _dac() -> DataAccessCollection:
        return toy_dac(h1={_DERIVED: [1], COL: [1, 2]})

    def test_unpointed_searched_route_steps_aside_for_the_computing_group(self) -> None:
        result = evaluate_or_raise(Feature(_DERIVED), _plugins(ToySourceFG, AggregatedFeatureGroup), None, self._dac())

        assert set(result.identified) == {AggregatedFeatureGroup}

    def test_source_still_serves_the_name_when_the_computing_group_is_eliminated(self) -> None:
        plugins: FeatureGroupEnvironmentMapping = {
            ToySourceFG: {PyArrowTable},
            AggregatedFeatureGroup: {_ToyUnservedFramework},
        }
        result = evaluate_or_raise(Feature(_DERIVED), plugins, None, self._dac())

        assert set(result.identified) == {ToySourceFG}
        assert result.eliminations[AggregatedFeatureGroup].stage == "capability"

    def test_a_data_access_handle_does_not_keep_the_source_in_play(self) -> None:
        feature = Feature(_DERIVED, Options({"data_access_handle": "h1"}))
        result = evaluate_or_raise(feature, _plugins(ToySourceFG, AggregatedFeatureGroup), None, self._dac())

        assert set(result.identified) == {AggregatedFeatureGroup}

    def test_a_group_matching_only_an_output_part_does_not_make_the_source_step_aside(self) -> None:
        dac = toy_dac(h1={_PART_NAME: [1]})
        result = evaluate_or_raise(Feature(_PART_NAME), _plugins(ToySourceFG, _PartOnlyFG), None, dac)

        assert set(result.identified) == {ToySourceFG}


class TestPointedRoutesDeclineWithReason:
    def test_pointed_route_without_a_source_records_an_input_data_rejection(self) -> None:
        result = _identify(Feature(COL, feature_group=ToySourceFG), None, ToySourceFG)

        assert result.identified == {}
        elimination = result.eliminations[ToySourceFG]
        assert elimination.stage == "input_data"
        assert "ToySourceFG was pointed at but found no source for 'toyfmt_col'" in elimination.reason

    def test_pointed_declared_route_with_an_undeclared_name_records_a_rejection(self) -> None:
        feature = Feature("toyfmt_undeclared", feature_group=ToyDeclaredFG)
        result = _identify(feature, toy_dac(h1={"toyfmt_undeclared": [1]}), ToyDeclaredFG)

        assert result.identified == {}
        elimination = result.eliminations[ToyDeclaredFG]
        assert elimination.stage == "input_data"
        assert "toyfmt_undeclared" in elimination.reason


class _AbstractRouted(SourceFeatureGroup):
    """Abstract group that inherits routes but implements no hooks."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)


class _AbstractRoutedChild(_AbstractRouted):
    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return None


class TestAbstractGroupsNeverClaim:
    def test_identification_skips_an_abstract_group_with_inherited_routes(self) -> None:
        result = _identify(Feature(COL), toy_dac(h1={COL: [1]}), _AbstractRouted, _AbstractRoutedChild)

        assert result.identified == {}

    def test_matcher_does_not_call_the_abstract_hooks(self) -> None:
        assert _AbstractRouted._matches_by_default_rules(COL, Options(), toy_dac(h1={COL: [1]})) is False

    def test_scoped_abstract_group_records_its_missing_abstract_methods(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        with feature_group_scope(_AbstractRoutedChild):
            assert _AbstractRoutedChild._matches_by_default_rules(COL, Options(), toy_dac(h1={COL: [1]})) is False

        reason = rejection_window["_AbstractRoutedChild"].reason
        assert str(sorted(_AbstractRoutedChild.__abstractmethods__)) in reason

    def test_pointed_abstract_group_records_its_missing_abstract_methods(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options({"_AbstractRouted": {COL: [1]}})
        assert _AbstractRouted._matches_by_default_rules(COL, options, None) is False

        reason = rejection_window["_AbstractRouted"].reason
        assert str(sorted(_AbstractRouted.__abstractmethods__)) in reason


class _VirtualColumnFG(ToySourceFG):
    """has_column answers for a column that columns() does not list."""

    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool | None:
        return column == "toyfmt_virtual"


class _NoColumnsFG(ToySourceBase):
    @classmethod
    def columns(cls, match: SourceMatch) -> None:
        return None


COLUMNS_CALLS: list[str] = []


class _SpyColumnsFG(ToySourceFG):
    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool | None:
        return column == COL

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        COLUMNS_CALLS.append(match.source)
        return [COL]


class TestHasColumnHook:
    def test_default_derives_from_columns(self) -> None:
        match = SourceMatch(source="h1:toy", access={COL: [1]})

        assert ToySourceFG.has_column(match, COL) is True
        assert ToySourceFG.has_column(match, "toyfmt_absent") is False

    def test_default_is_none_when_columns_is_unknown(self) -> None:
        assert _NoColumnsFG.has_column(SourceMatch(source="h1:toy", access={}), COL) is None

    def test_has_column_decides_fitting_over_columns(self) -> None:
        feature = Feature("toyfmt_virtual")
        result = _identify(feature, toy_dac(h1={COL: [1]}), _VirtualColumnFG)

        assert _VirtualColumnFG in result.identified
        pair = feature.input_data_match
        assert pair is not None
        assert pair[1].source == "h1:toy"

    def test_has_column_false_declines_even_when_columns_lists_the_name(self) -> None:
        assert not _claims(Feature(COL), toy_dac(h1={COL: [1]}), _VirtualColumnFG)

    def test_columns_is_not_called_on_a_successful_fit(self) -> None:
        COLUMNS_CALLS.clear()

        assert _claims(Feature(COL), toy_dac(h1={COL: [1]}), _SpyColumnsFG)
        assert COLUMNS_CALLS == []

    def test_columns_is_called_for_rejection_text(self) -> None:
        COLUMNS_CALLS.clear()

        result = _identify(Feature("toyfmt_absent"), toy_dac(h1={COL: [1]}), _SpyColumnsFG)

        assert "h1:toy" in result.eliminations[_SpyColumnsFG].reason
        assert COLUMNS_CALLS


FIX_TEXT = "toyfmt-specific: remove one of the duplicate sources."


class _FixTextFG(ToySourceFG):
    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        return FIX_TEXT


class TestAmbiguityFixHook:
    def test_default_keeps_todays_text(self) -> None:
        text = ToySourceFG.ambiguity_fix(COL, [], None)

        assert "data_access_handle" in text
        assert "column_to_file" in text

    def test_override_text_ends_the_abort(self) -> None:
        dac = toy_dac(h1={COL: [1]}, h2={COL: [2]})
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(Feature(COL), _plugins(_FixTextFG), None, dac)

        message = str(exc_info.value)
        assert message.endswith(FIX_TEXT)
        assert "h1:toy" in message
        assert "h2:toy" in message
        assert "column_to_file" not in message


class _PointedParent(ToySourceBase):
    """Open, pointed-only; sources come from its own class-name key."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.OPEN, False),)

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        for key in ("_PointedParent", "_AbstractToyMid"):
            scoped = options.get(key)
            if scoped is not None:
                return [SourceMatch(source="scoped:toy", access=scoped)]
        return []


class _PointedChild(_PointedParent):
    """Concrete subclass of a concrete source group."""


class _AbstractOpenMid(ToySourceBase):
    @classmethod
    @abstractmethod
    def marker(cls) -> str: ...


class _OpenUnderAbstract(_AbstractOpenMid):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.OPEN, False),)

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        scoped = options.get("_AbstractOpenMid")
        return [] if scoped is None else [SourceMatch(source="scoped:toy", access=scoped)]

    @classmethod
    def marker(cls) -> str:
        return "x"


class TestPointerKeysFollowTheMro:
    def test_own_name_first_then_concrete_source_ancestors(self) -> None:
        keys = ToySourceSubFG.pointer_keys()

        assert keys[:2] == ("ToySourceSubFG", "ToySourceFG")
        assert "SourceFeatureGroup" not in keys

    def test_abstract_ancestor_is_not_a_pointer_key(self) -> None:
        keys = _OpenUnderAbstract.pointer_keys()

        assert keys[0] == "_OpenUnderAbstract"
        assert "_AbstractOpenMid" not in keys

    def test_parent_key_points_a_concrete_subclass(self) -> None:
        feature = Feature("toyfmt_any", Options({"_PointedParent": {COL: [1]}}))

        assert _claims(feature, None, _PointedChild)

    def test_abstract_ancestor_key_does_not_point(self) -> None:
        feature = Feature("toyfmt_any", Options({"_AbstractOpenMid": {COL: [1]}}))

        assert not _claims(feature, None, _OpenUnderAbstract)

    def test_subclass_takes_over_a_parent_pointed_feature(self) -> None:
        feature = Feature("toyfmt_any", Options({"_PointedParent": {COL: [1]}}))
        result = evaluate_or_raise(feature, _plugins(_PointedParent, _PointedChild), None, None)

        assert set(result.identified) == {_PointedChild}
        assert result.specialized_from == (_PointedParent,)
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is _PointedChild

    def test_takeover_runs_end_to_end_with_enabled_feature_groups_gating(self) -> None:
        feature = Feature("toyfmt_any", Options({"_PointedParent": {"toyfmt_any": [4, 5]}}))
        session = mloda.prepare(
            [feature],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({_PointedParent, _PointedChild}),
        )

        steps = [st for st in session.resolved_plan() if st.step_kind == "compute"]
        assert {st.feature_group for st in steps} == {_PointedChild}
        assert column_values(session.run(), "toyfmt_any") == [4, 5]


CACHED_CALLS: list[str] = []


class _CachedColumnsFG(ToySourceFG):
    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached

        def compute() -> list[str]:
            CACHED_CALLS.append(match.source)
            return list(match.access)

        return run_cached(("toyfmt_cache", match.source), compute)


class TestMatchesAreSharedWithinOneRun:
    DERIVED = "toyfmt_cache_a__sum_aggr"

    def _run(self) -> None:
        mloda.run_all(
            ["toyfmt_cache_a", "toyfmt_cache_b", self.DERIVED],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({_CachedColumnsFG, AggregatedFeatureGroup}),
            data_access_collection=toy_dac(hcache={"toyfmt_cache_a": [1, 2], "toyfmt_cache_b": [3, 4]}),
        )

    def test_columns_is_computed_once_per_run_and_again_for_a_second_run(self) -> None:
        CACHED_CALLS.clear()

        self._run()
        assert CACHED_CALLS == ["hcache:toy"]

        self._run()
        assert CACHED_CALLS == ["hcache:toy", "hcache:toy"]


def _stock(module: str, name: str) -> Any:
    return load_group(module, name)


def _write_csv_and_parquet(folder: Path) -> tuple[str, str]:
    csv_path = folder / "fpg_a.csv"
    parquet_path = folder / "fpg_b.parquet"
    write_csv(csv_path, {"fpg_x": [1, 2]})
    write_parquet(parquet_path, {"fpg_x": [9, 9]})
    return str(csv_path), str(parquet_path)


class TestForeignPointerMakesSearchedRoutesDecline:
    def test_an_option_key_naming_no_source_group_is_ignored(self) -> None:
        feature = Feature(COL, Options({"toyfmt_not_a_group": 1}))

        assert _claims(feature, toy_dac(h1={COL: [1]}), ToySourceFG)

    @pytest.mark.parametrize(
        ("key", "feature_group"),
        [("CsvReader", None), (None, "ReadFileFeature")],
        ids=["option_key", "scope"],
    )
    def test_a_former_reader_name_is_an_unknown_name_and_raises_no_pointer_error(
        self, key: str | None, feature_group: str | None
    ) -> None:
        feature = Feature("fpg_x", Options({key: "x"} if key else {}), feature_group=feature_group)
        result = IdentifyFeatureGroupClass.evaluate(feature, _plugins(_stock("csv_fg", "CsvFG")), None, None)
        assert not result.identified

    def test_a_csv_pointer_returns_the_pointed_value_though_the_collection_holds_the_column_elsewhere(
        self, tmp_path: Path
    ) -> None:
        csv_path, _ = _write_csv_and_parquet(tmp_path)
        result = mloda.run_all(
            [Feature("fpg_x", Options({"CsvFG": csv_path}))],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {_stock("csv_fg", "CsvFG"), _stock("parquet_fg", "ParquetFG")}
            ),
            data_access_collection=DataAccessCollection(folders={"fpg_dir": str(tmp_path)}),
        )

        assert column_values(result, "fpg_x") == [1, 2]

    def test_a_foreign_pointer_also_declines_a_required_option_route(self) -> None:
        from mloda.provider import ReadDBFG
        from tests.test_plugins.feature_group.input_data.test_read_dbs.test_read_db_fg import KEY, ToyBaseDB

        routes = (*ReadDBFG.CLAIM_ROUTES, ReadDBFG.QUERY_ROUTE)
        group_a = type("_QueryDBA", (ToyBaseDB,), {"CLAIM_ROUTES": routes})
        group_b = type("_QueryDBB", (ToyBaseDB,), {"CLAIM_ROUTES": routes})
        feature = Feature("anything", Options({"query_text": "SELECT 1", "_QueryDBA": {KEY: "a"}}))
        result = _identify(feature, DataAccessCollection(credentials={"toy_handle": {KEY: "a"}}), group_a, group_b)

        assert set(result.identified) == {group_a}
        assert "options point at _QueryDBA" in result.eliminations[group_b].reason
        del group_a, group_b, result, feature
        gc.collect()

    def test_a_group_without_a_query_route_keeps_the_foreign_pointer_reason_on_query_text(self) -> None:
        route = ClaimRoute("toy", NamePolicy.CHECKED, False, required_options=("query_text",))
        pointed = type("_QueryToyFG", (ToySourceBase,), {"CLAIM_ROUTES": (route,)})
        feature = Feature(COL, Options({"query_text": "SELECT 1", "_QueryToyFG": {COL: [1]}}))
        result = _identify(feature, toy_dac(h1={COL: [1]}), pointed, ToySourceFG)

        assert set(result.identified) == {pointed}
        reason = result.eliminations[ToySourceFG].reason
        assert "options point at _QueryToyFG" in reason
        assert "query_text" not in reason
        del pointed, result, feature
        gc.collect()


class TestUnreachablePointerRaises:
    def _run(self, options: Options | None, feature_group: Any, collector: PluginCollector, folder: Path) -> None:
        write_csv(folder / "fpg_a.csv", {"fpg_x": [1]})
        mloda.run_all(
            [Feature("fpg_x", options if options is not None else Options(), feature_group=feature_group)],
            compute_frameworks=[PyArrowTable],
            plugin_collector=collector,
            data_access_collection=DataAccessCollection(folders={"fpg_dir": str(folder)}),
        )

    @pytest.mark.parametrize(
        ("scope", "expected"),
        [
            (False, "JsonFG is not accessible in this run"),
            (True, "feature_group=JsonFG"),
        ],
        ids=["pointer", "scope"],
    )
    def test_a_disabled_group_raises_naming_the_group_and_the_collector(
        self, scope: bool, expected: str, tmp_path: Path
    ) -> None:
        collector = PluginCollector.disabled_feature_groups({_stock("json_fg", "JsonFG")})
        options = None if scope else Options({"JsonFG": str(tmp_path / "b.json")})
        with pytest.raises(SourcePointerError) as exc_info:
            self._run(options, "JsonFG" if scope else None, collector, tmp_path)

        message = str(exc_info.value)
        assert expected in message
        assert "disabled by the PluginCollector" in message

    def test_a_pointer_at_a_strict_dropped_group_raises_naming_strict_mode(self, tmp_path: Path) -> None:
        from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin

        csv_fg = _stock("csv_fg", "CsvFG")
        json_fg = _stock("json_fg", "JsonFG")
        PluginRegistry.default().clear()
        register_plugin(csv_fg, replace=True)
        collector = PluginCollector.enabled_feature_groups({csv_fg, json_fg}).set_strict_mode("strict")
        with pytest.raises(SourcePointerError) as exc_info:
            self._run(Options({"JsonFG": str(tmp_path / "b.json")}), None, collector, tmp_path)

        assert "dropped by strict mode" in str(exc_info.value)

    def test_a_scope_with_a_pointer_at_a_different_group_raises_naming_both(self, tmp_path: Path) -> None:
        csv_fg = _stock("csv_fg", "CsvFG")
        json_fg = _stock("json_fg", "JsonFG")
        feature = Feature("fpg_x", Options({"JsonFG": str(tmp_path / "b.json")}), feature_group="CsvFG")
        with pytest.raises(SourcePointerError) as exc_info:
            IdentifyFeatureGroupClass.evaluate(feature, _plugins(csv_fg, json_fg), None, None)

        assert "CsvFG" in str(exc_info.value)
        assert "JsonFG" in str(exc_info.value)

    @pytest.mark.parametrize(("key", "group"), [("JsonFg", "JsonFG"), ("csvfg", "CsvFG")])
    def test_a_case_typo_pointer_key_raises_with_a_suggestion(self, key: str, group: str) -> None:
        csv_fg = _stock("csv_fg", "CsvFG")
        json_fg = _stock("json_fg", "JsonFG")
        feature = Feature("fpg_x", Options({key: "x.json"}))
        with pytest.raises(SourcePointerError) as exc_info:
            IdentifyFeatureGroupClass.evaluate(feature, _plugins(csv_fg, json_fg), None, None)

        assert key in str(exc_info.value)
        assert f"did you mean {group}" in str(exc_info.value)


_COLS = {COL: [1]}
_REQ = {"toyfmt_required": 1}


class _AbstractToyRouted(_AbstractToyMid):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.CHECKED, True),)


class TestHasSource:
    @pytest.mark.parametrize(
        ("group", "options", "dac", "scope", "expected"),
        [
            (ToySourceFG, {}, toy_dac(h1=_COLS), None, True),
            (ToySourceFG, {}, None, None, False),
            (ToySourceFG, {"query_text": "x"}, toy_dac(h1=_COLS), None, False),
            (_PointedChild, {"_PointedParent": _COLS}, None, None, True),
            (ToySourceFG, {"ToyOtherSourceFG": _COLS}, toy_dac(h1=_COLS), None, False),
            (ToyOpenFG, {}, toy_dac(h1=_COLS), None, False),
            (ToyOpenFG, {"ToyOpenFG": _COLS}, None, None, True),
            (ToyOpenFG, {}, toy_dac(h1=_COLS), ToyOpenFG, True),
            (ToyRequiredOptionFG, {}, toy_dac(h1=_COLS), None, False),
            (ToyRequiredOptionFG, _REQ, toy_dac(h1=_COLS), None, True),
            (_AbstractToyRouted, {}, toy_dac(h1=_COLS), None, False),
            (
                ApiDataFG,
                {"ApiDataFG": "missing"},
                DataAccessCollection().with_api_data_columns({"k": (COL,)}),
                None,
                False,
            ),
        ],
        ids=[
            "searched_with_source",
            "searched_without_collection",
            "query_text_without_query_route",
            "parent_class_name_key",
            "foreign_pointer",
            "open_unpointed",
            "open_class_name_key",
            "open_scope",
            "required_option_absent",
            "required_option_present",
            "abstract",
            "api_data_unknown_set_key_is_contained",
        ],
    )
    def test_has_source(
        self,
        group: type[SourceFeatureGroup],
        options: dict[str, Any],
        dac: DataAccessCollection | None,
        scope: type | None,
        expected: bool,
    ) -> None:
        with collect_match_rejections() as outer:
            with feature_group_scope(scope) if scope else nullcontext():
                assert group.has_source(Options(options), dac, COL) is expected

        assert outer == {}

    def test_a_true_probe_checks_no_name_and_leaves_the_options_untouched(self) -> None:
        options = Options({"toyfmt_required": 1})

        assert ToyRequiredOptionFG.has_source(options, toy_dac(h1=_COLS), "toyfmt_not_a_column") is True

        assert RESERVED_READER_OPTION_KEY not in options.group
        assert RESERVED_READER_OPTION_KEY not in options.context
