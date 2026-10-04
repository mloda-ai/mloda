"""FormatFeatureGroup class-definition rule, claim-route matching, pointing, ambiguity and plan identity."""

import copy
from abc import abstractmethod
from typing import ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.provider import ClaimRoute, NamePolicy, SourceMatch
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyDeclaredFG,
    ToyFormatBase,
    ToyFormatFG,
    ToyFormatSubFG,
    ToyOpenFG,
    ToyOtherFormatFG,
    ToyRequiredOptionFG,
    foreign_dac,
    toy_dac,
)
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise

COL = "toyfmt_col"


def _plugins(*groups: type) -> FeatureGroupEnvironmentMapping:
    return {g: {PyArrowTable} for g in groups}


def _identify(feature: Feature, dac: DataAccessCollection | None, *groups: type) -> EvaluationResult:
    return IdentifyFeatureGroupClass.evaluate(feature, _plugins(*groups), None, dac)


def _claims(feature: Feature, dac: DataAccessCollection | None, group: type) -> bool:
    return group in _identify(feature, dac, group).identified


class TestClassDefinitionRule:
    def test_open_searched_route_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):

            class _Bad(ToyFormatBase):
                CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.OPEN, True),)

    def test_open_searched_route_among_valid_routes_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):

            class _BadMixed(ToyFormatBase):
                CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
                    ClaimRoute("toy", NamePolicy.CHECKED, True),
                    ClaimRoute("toy2", NamePolicy.OPEN, True),
                )

    def test_open_pointed_only_route_is_allowed(self) -> None:
        assert ToyOpenFG.CLAIM_ROUTES == (ClaimRoute("toy", NamePolicy.OPEN, False),)

    def test_group_without_routes_never_claims(self) -> None:
        assert not _claims(Feature(COL), toy_dac(h1={COL: [1]}), ToyFormatBase)


class TestCheckedNames:
    def test_present_column_matches_and_records_pair(self) -> None:
        feature = Feature(COL)
        result = _identify(feature, toy_dac(h1={COL: [1]}), ToyFormatFG)

        assert ToyFormatFG in result.identified
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is ToyFormatFG
        assert pair[1] == SourceMatch(source="h1:toy", access=None)
        assert pair[1].access == {COL: [1]}

    def test_missing_column_declines_with_input_data_rejection(self) -> None:
        result = _identify(Feature(COL), toy_dac(h1={"other": [1]}), ToyFormatFG)

        assert result.identified == {}
        elimination = result.eliminations[ToyFormatFG]
        assert elimination.stage == "input_data"
        assert "h1:toy" in elimination.reason or COL in elimination.reason

    def test_foreign_source_declines(self) -> None:
        assert not _claims(Feature(COL), foreign_dac(), ToyFormatFG)

    def test_no_collection_declines(self) -> None:
        assert not _claims(Feature(COL), None, ToyFormatFG)

    def test_class_name_shortcut_does_not_replace_the_checked_column(self) -> None:
        assert not _claims(Feature("ToyFormatFG"), toy_dac(h1={COL: [1]}), ToyFormatFG)


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

    def test_pointed_by_data_access_handle_option(self) -> None:
        feature = Feature("toyfmt_any", Options({"data_access_handle": "h1"}))
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_pointed_by_feature_group_scope(self) -> None:
        feature = Feature("toyfmt_any", feature_group=ToyOpenFG)
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_scope_by_name_string_points_too(self) -> None:
        feature = Feature("toyfmt_any", feature_group="ToyOpenFG")
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyOpenFG)

    def test_scope_does_not_point_other_groups_candidates_when_out_of_scope(self) -> None:
        feature = Feature("toyfmt_any", feature_group=ToyOpenFG)
        assert not _claims(feature, toy_dac(h1={COL: [1]}), ToyFormatFG)


class TestRequiredOptionRoute:
    def test_declines_without_the_required_option(self) -> None:
        assert not _claims(Feature(COL), toy_dac(h1={COL: [1]}), ToyRequiredOptionFG)

    def test_required_option_presence_points_the_route(self) -> None:
        feature = Feature(COL, Options({"toyfmt_required": "yes"}))
        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyRequiredOptionFG)


class TestPointedButMissingAborts:
    def test_class_name_key_source_lacking_the_column_raises(self) -> None:
        feature = Feature(COL, Options({"ToyFormatFG": {"other": [1]}}))
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToyFormatFG), None, None)
        message = str(exc_info.value)
        assert "ToyFormatFG" in message
        assert "scoped:toy" in message
        assert COL in message
        assert "other" in message

    def test_handle_pointing_at_a_source_lacking_the_column_raises(self) -> None:
        feature = Feature(COL, Options({"data_access_handle": "h1"}))
        dac = toy_dac(h1={"other": [1]}, h2={COL: [1]})
        with pytest.raises(ValueError, match="h1:toy"):
            evaluate_or_raise(feature, _plugins(ToyFormatFG), None, dac)

    def test_abort_does_not_fall_through_to_the_other_candidate(self) -> None:
        feature = Feature(COL, Options({"ToyFormatFG": {"other": [1]}}))
        dac = toy_dac(h1={COL: [1]})
        assert _claims(Feature(COL), dac, ToyOtherFormatFG)
        with pytest.raises(ValueError, match="ToyFormatFG"):
            evaluate_or_raise(feature, _plugins(ToyFormatFG, ToyOtherFormatFG), None, dac)


class TestAmbiguousSources:
    def test_two_fitting_sources_raise_naming_both_and_the_fix(self) -> None:
        dac = toy_dac(h1={COL: [1]}, h2={COL: [2]})
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(Feature(COL), _plugins(ToyFormatFG), None, dac)
        message = str(exc_info.value)
        assert "ToyFormatFG" in message
        assert "h1:toy" in message
        assert "h2:toy" in message
        assert "feature_group=" in message
        assert "data_access_handle" in message
        assert "column_to_file" in message

    def test_data_access_handle_resolves_the_ambiguity(self) -> None:
        feature = Feature(COL, Options({"data_access_handle": "h2"}))
        evaluate_or_raise(feature, _plugins(ToyFormatFG), None, toy_dac(h1={COL: [1]}, h2={COL: [2]}))
        pair = feature.input_data_match
        assert pair is not None
        assert pair[1] == SourceMatch(source="h2:toy", access=None)


class TestNoMutation:
    def test_collection_and_options_unchanged_after_matching(self) -> None:
        dac = toy_dac(h1={COL: [1]})
        credentials_before = copy.deepcopy(dac.credentials)
        options = Options({"keep": 1})
        feature = Feature(COL, options)
        group_before = copy.deepcopy(feature.options.group)
        context_before = copy.deepcopy(feature.options.context)

        _identify(feature, dac, ToyFormatFG)

        assert dac.credentials == credentials_before
        assert feature.options.group == group_before
        assert feature.options.context == context_before
        assert RESERVED_READER_OPTION_KEY not in feature.options.group
        assert RESERVED_READER_OPTION_KEY not in feature.options.context


class TestSubclassTakeover:
    def test_subclass_wins_and_records_its_own_pair(self) -> None:
        feature = Feature(COL, feature_group=ToyFormatFG)
        result = evaluate_or_raise(feature, _plugins(ToyFormatFG, ToyFormatSubFG), None, toy_dac(h1={COL: [1]}))

        assert next(iter(result.identified)) is ToyFormatSubFG
        assert result.specialized_from == (ToyFormatFG,)
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is ToyFormatSubFG
        assert pair[1] == SourceMatch(source="h1:toy", access=None)


class TestPlanIdentity:
    def test_two_sources_of_one_group_make_two_steps_naming_group_and_source(self) -> None:
        dac = toy_dac(h1={"toyfmt_a": [1]}, h2={"toyfmt_b": [2]})
        session = mloda.prepare(
            [Feature("toyfmt_a"), Feature("toyfmt_b")],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyFormatFG}),
            data_access_collection=dac,
        )

        steps = [s for s in session.resolved_plan() if s.step_kind == "compute"]

        assert len(steps) == 2
        assert {s.feature_group for s in steps} == {ToyFormatFG}
        assert {s.data_access_identity for s in steps} == {"h1:toy", "h2:toy"}
        assert {s.feature_names for s in steps} == {("toyfmt_a",), ("toyfmt_b",)}

    def test_two_features_of_one_source_share_one_step(self) -> None:
        dac = toy_dac(h1={"toyfmt_a": [1], "toyfmt_b": [2]})
        session = mloda.prepare(
            [Feature("toyfmt_a"), Feature("toyfmt_b")],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyFormatFG}),
            data_access_collection=dac,
        )

        steps = [s for s in session.resolved_plan() if s.step_kind == "compute"]

        assert len(steps) == 1
        assert steps[0].data_access_identity == "h1:toy"


class _AbstractToyMid(ToyFormatBase):
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
        feature = Feature(COL, feature_group=ToyFormatFG)
        with pytest.raises(ValueError, match="h1:toy"):
            evaluate_or_raise(feature, _plugins(ToyFormatSubFG), None, toy_dac(h1={"other": [1]}))

    def test_scope_naming_the_candidate_itself_points(self) -> None:
        feature = Feature(COL, feature_group=_MidLacksColumn)
        with pytest.raises(ValueError, match="h1:toy"):
            evaluate_or_raise(feature, _plugins(_MidLacksColumn), None, toy_dac(h1={COL: [1]}))
