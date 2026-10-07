"""FormatFeatureGroup class-definition rule, claim-route matching, pointing, ambiguity and plan identity."""

import gc
from abc import abstractmethod
from collections.abc import Collection
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.core.abstract_plugins.components.input_data.claim_route import feature_group_scope
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    FormatPointerError,
)
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import ClaimRoute, FormatFeatureGroup, NamePolicy, SourceMatch
from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup
from tests.mixins.reader_feature_groups.format_file_writers import write_csv, write_parquet
from tests.mixins.reader_feature_groups.lazy_format_group import load_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyDeclaredFG,
    ToyFormatBase,
    ToyFormatFG,
    ToyFormatSubFG,
    ToyOpenFG,
    ToyOtherFormatFG,
    ToyRequiredOptionFG,
    column_values,
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
        with pytest.raises(TypeError):

            class _Bad(ToyFormatBase):
                CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("toy", NamePolicy.OPEN, True),)

    def test_open_searched_route_among_valid_routes_raises(self) -> None:
        with pytest.raises(TypeError):

            class _BadMixed(ToyFormatBase):
                CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
                    ClaimRoute("toy", NamePolicy.CHECKED, True),
                    ClaimRoute("toy2", NamePolicy.OPEN, True),
                )

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
        assert "h1:toy" in elimination.reason

    def test_no_collection_declines(self) -> None:
        assert not _claims(Feature(COL), None, ToyFormatFG)


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
        result = _identify(feature, toy_dac(h1={COL: [1]}), ToyFormatFG, ToyOpenFG)
        assert ToyFormatFG not in result.identified


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

    def test_handle_alone_narrows_but_does_not_escalate_a_missing_column(self) -> None:
        feature = Feature(COL, Options({"data_access_handle": "h1"}))
        dac = toy_dac(h1={"other": [1]}, h2={COL: [1]})
        result = _identify(feature, dac, ToyFormatFG)
        assert result.identified == {}
        assert "h1:toy" in result.eliminations[ToyFormatFG].reason

    def test_abort_does_not_fall_through_to_the_other_candidate(self) -> None:
        feature = Feature(COL, Options({"ToyFormatFG": {"other": [1]}}))
        dac = toy_dac(h1={COL: [1]})
        assert _claims(Feature(COL), dac, ToyOtherFormatFG)
        with pytest.raises(ValueError, match="ToyFormatFG"):
            evaluate_or_raise(feature, _plugins(ToyFormatFG, ToyOtherFormatFG), None, dac)


class TestPointedMissingColumnText:
    def test_fix_text_is_one_sentence_without_the_ambiguity_fix(self) -> None:
        feature = Feature(COL, Options({"ToyFormatFG": {"other": [1]}}))
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToyFormatFG), None, None)
        message = str(exc_info.value)
        assert "column_to_file" not in message
        assert message.count(" or ") == 1

    def test_long_column_list_is_truncated_to_twenty_names(self) -> None:
        columns = {f"toyfmt_c{i:02d}": [i] for i in range(25)}
        feature = Feature(COL, Options({"ToyFormatFG": columns}))
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToyFormatFG), None, None)
        message = str(exc_info.value)
        assert "toyfmt_c00" in message
        assert "toyfmt_c19" in message
        assert "toyfmt_c20" not in message
        assert "... and 5 more" in message


class TestAmbiguousSources:
    def test_two_fitting_sources_raise_naming_both_and_the_fix(self) -> None:
        dac = toy_dac(h1={COL: [1]}, h2={COL: [2]})
        with pytest.raises(ValueError) as exc_info:
            evaluate_or_raise(Feature(COL), _plugins(ToyFormatFG), None, dac)
        message = str(exc_info.value)
        assert "ToyFormatFG" in message
        assert "h1:toy" in message
        assert "h2:toy" in message
        assert "feature_group=" not in message
        assert "data_access_handle" in message
        assert "column_to_file" in message

    def test_data_access_handle_resolves_the_ambiguity(self) -> None:
        feature = Feature(COL, Options({"data_access_handle": "h2"}))
        evaluate_or_raise(feature, _plugins(ToyFormatFG), None, toy_dac(h1={COL: [1]}, h2={COL: [2]}))
        pair = feature.input_data_match
        assert pair is not None
        assert pair[1] == SourceMatch(source="h2:toy", access=None)


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


class _ToyUnservedFramework(ComputeFramework):
    """Uniquely named framework no candidate serves, so a pin on it eliminates every candidate."""


class _TakeoverChildFG(ToyFormatFG):
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


class TestPointingIsExplicitOnly:
    """Group options reach derived features' inputs: only the class-name key and a feature_group= scope point."""

    DERIVED = "toyfmt_col__sum_aggr"

    @pytest.mark.parametrize("fg", [ToyFormatFG, ToyOpenFG])
    def test_handle_on_a_derived_feature_resolves_to_the_aggregation(self, fg: type) -> None:
        feature = Feature(self.DERIVED, Options(group={"data_access_handle": "h1"}))
        result = evaluate_or_raise(
            feature, _plugins(fg, PyArrowAggregatedFeatureGroup), None, toy_dac(h1={COL: [1, 2]})
        )

        assert set(result.identified) == {PyArrowAggregatedFeatureGroup}

    def test_required_option_route_does_not_escalate_a_missing_column(self) -> None:
        feature = Feature(self.DERIVED, Options({"toyfmt_required": "yes"}))
        result = evaluate_or_raise(
            feature,
            _plugins(ToyRequiredOptionFG, PyArrowAggregatedFeatureGroup),
            None,
            toy_dac(h1={COL: [1]}),
        )

        assert set(result.identified) == {PyArrowAggregatedFeatureGroup}


_DERIVED = "toyfmt_col__sum_aggr"


class _PlainClaimsDerived(FeatureGroup):
    """Non-format survivor claiming the derived name."""

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

    @pytest.mark.parametrize("fg", [ToyFormatFG, ToyRequiredOptionFG])
    def test_class_name_key_on_a_derived_feature_resolves_to_the_aggregation(self, fg: type) -> None:
        feature = Feature(self.DERIVED, Options({fg.__name__: {COL: [1, 2]}, "toyfmt_required": "yes"}))
        result = evaluate_or_raise(feature, _plugins(fg, PyArrowAggregatedFeatureGroup), None, None)

        assert set(result.identified) == {PyArrowAggregatedFeatureGroup}
        elimination = result.eliminations[fg]
        assert elimination.stage == "input_data"
        assert "is in none of the sources" in elimination.reason

    @pytest.mark.parametrize(
        ("name", "groups", "dac"),
        [
            ("toyfmt_unclaimed", (ToyFormatFG, PyArrowAggregatedFeatureGroup), None),
            (_DERIVED, (ToyFormatFG, _PlainClaimsDerived, PyArrowAggregatedFeatureGroup), None),
        ],
        ids=["no_other_claimant", "another_reader_survives"],
    )
    def test_pointed_reader_still_aborts(
        self, name: str, groups: tuple[type, ...], dac: DataAccessCollection | None
    ) -> None:
        feature = Feature(name, Options({"ToyFormatFG": {COL: [1]}, "toyfmt_required": "yes"}))
        with pytest.raises(ValueError, match=rf"column '{name}' is in none of the sources of ToyFormatFG"):
            evaluate_or_raise(feature, _plugins(*groups), None, dac)

    def test_computing_candidate_eliminated_later_still_counts_as_computed(self) -> None:
        feature = Feature(
            self.DERIVED,
            Options({"ToyFormatFG": {COL: [1, 2]}}),
            compute_framework=_ToyUnservedFramework.get_class_name(),
        )
        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, _plugins(ToyFormatFG, PyArrowAggregatedFeatureGroup), None, None)

        result = exc_info.value.result
        assert result.identified == {}
        assert result.eliminations[PyArrowAggregatedFeatureGroup].stage == "framework_pin"

    def test_pointed_reader_aborts_when_the_other_claimant_input_features_raises(self) -> None:
        feature = Feature("toyfmt_typo_probe", Options({"ToyFormatFG": {COL: [1]}}))
        with pytest.raises(ValueError, match="column 'toyfmt_typo_probe' is in none of the sources of ToyFormatFG"):
            evaluate_or_raise(feature, _plugins(ToyFormatFG, _RaisingInputsFG), None, None)

    def test_strictness_holds_per_deferring_group_despite_another_groups_takeover(self) -> None:
        options = Options({"ToyFormatFG": {"other": [1]}, "ToyOtherFormatFG": {"other": [1]}})
        feature = Feature(COL, options, compute_framework=PyArrowTable.get_class_name())
        plugins: FeatureGroupEnvironmentMapping = {
            ToyFormatFG: {PyArrowTable},
            ToyOtherFormatFG: {PyArrowTable},
            _TakeoverChildFG: {PyArrowTable},
            _ComputingColFG: {_ToyUnservedFramework},
        }
        with pytest.raises(ValueError, match="is in none of the sources of ToyOtherFormatFG"):
            evaluate_or_raise(feature, plugins, None, None)

    def test_pointed_parent_defers_to_its_surviving_subclass(self) -> None:
        feature = Feature(COL, Options({"ToyFormatFG": {"other": [1]}}))
        result = evaluate_or_raise(feature, _plugins(ToyFormatFG, _TakeoverChildFG), None, None)

        assert set(result.identified) == {_TakeoverChildFG}
        assert result.eliminations[ToyFormatFG].stage == "input_data"


class TestPointedRoutesDeclineWithReason:
    def test_pointed_route_without_a_source_records_an_input_data_rejection(self) -> None:
        result = _identify(Feature(COL, feature_group=ToyFormatFG), None, ToyFormatFG)

        assert result.identified == {}
        assert result.eliminations[ToyFormatFG].stage == "input_data"

    def test_pointed_declared_route_with_an_undeclared_name_records_a_rejection(self) -> None:
        feature = Feature("toyfmt_undeclared", feature_group=ToyDeclaredFG)
        result = _identify(feature, toy_dac(h1={"toyfmt_undeclared": [1]}), ToyDeclaredFG)

        assert result.identified == {}
        elimination = result.eliminations[ToyDeclaredFG]
        assert elimination.stage == "input_data"
        assert "toyfmt_undeclared" in elimination.reason


class _AbstractRouted(FormatFeatureGroup):
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


class _VirtualColumnFG(ToyFormatFG):
    """has_column answers for a column that columns() does not list."""

    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool | None:
        return column == "toyfmt_virtual"


class _NoColumnsFG(ToyFormatBase):
    @classmethod
    def columns(cls, match: SourceMatch) -> None:
        return None


COLUMNS_CALLS: list[str] = []


class _SpyColumnsFG(ToyFormatFG):
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

        assert ToyFormatFG.has_column(match, COL) is True
        assert ToyFormatFG.has_column(match, "toyfmt_absent") is False

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


class _FixTextFG(ToyFormatFG):
    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        return FIX_TEXT


class TestAmbiguityFixHook:
    def test_default_keeps_todays_text(self) -> None:
        text = ToyFormatFG.ambiguity_fix(COL, [], None)

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


class _PointedParent(ToyFormatBase):
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
    """Concrete subclass of a concrete format group."""


class _AbstractOpenMid(ToyFormatBase):
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
    def test_own_name_first_then_concrete_format_ancestors(self) -> None:
        keys = ToyFormatSubFG.pointer_keys()

        assert keys[:2] == ("ToyFormatSubFG", "ToyFormatFG")
        assert "FormatFeatureGroup" not in keys

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


class _CachedColumnsFG(ToyFormatFG):
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
            plugin_collector=PluginCollector.enabled_feature_groups({_CachedColumnsFG, PyArrowAggregatedFeatureGroup}),
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
    def test_an_option_key_naming_no_format_group_is_ignored(self) -> None:
        feature = Feature(COL, Options({"toyfmt_not_a_group": 1}))

        assert _claims(feature, toy_dac(h1={COL: [1]}), ToyFormatFG)

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
        with pytest.raises(FormatPointerError) as exc_info:
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
        with pytest.raises(FormatPointerError) as exc_info:
            self._run(Options({"JsonFG": str(tmp_path / "b.json")}), None, collector, tmp_path)

        assert "dropped by strict mode" in str(exc_info.value)

    @pytest.mark.parametrize("as_scope", [False, True], ids=["pointer", "scope"])
    def test_an_unloaded_stock_group_raises_naming_the_loader(
        self, as_scope: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mloda.core.prepare import identify_feature_group as module

        csv_fg = _stock("csv_fg", "CsvFG")
        _stock("json_fg", "JsonFG")
        real = module.concrete_format_groups()
        monkeypatch.setattr(module, "concrete_format_groups", lambda: {k: v for k, v in real.items() if k != "JsonFG"})
        feature = (
            Feature("fpg_x", feature_group="JsonFG") if as_scope else Feature("fpg_x", Options({"JsonFG": "x.json"}))
        )
        with pytest.raises(FormatPointerError) as exc_info:
            IdentifyFeatureGroupClass.evaluate(feature, _plugins(csv_fg), None, None)

        assert "JsonFG is not loaded" in str(exc_info.value)
        assert "PluginLoader.all()" in str(exc_info.value)

    def test_a_scope_with_a_pointer_at_a_different_group_raises_naming_both(self, tmp_path: Path) -> None:
        csv_fg = _stock("csv_fg", "CsvFG")
        json_fg = _stock("json_fg", "JsonFG")
        feature = Feature("fpg_x", Options({"JsonFG": str(tmp_path / "b.json")}), feature_group="CsvFG")
        with pytest.raises(FormatPointerError) as exc_info:
            IdentifyFeatureGroupClass.evaluate(feature, _plugins(csv_fg, json_fg), None, None)

        assert "CsvFG" in str(exc_info.value)
        assert "JsonFG" in str(exc_info.value)
