"""ApiDataFG against the shared SourceFeatureGroup contract; tests it diverges from are redefined on the host."""

from typing import Any, cast

import pytest

from mloda._testing.source.source_suite import MatchRefuseAll, SourceFeatureGroupTestMixin, _spy
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.source_feature_group import HANDLE_OPTION
from mloda.core.abstract_plugins.components.match_rejection import CoreDeclineCode
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.provider import ApiDataFG, Cause, SourceFeatureGroup
from mloda.steward import ReaderProbeScope, reader_probe_scope
from mloda.user import Feature

_TWO_SETS = {"k": ("api_col",), "k2": ("api_col", "other_col")}


class TestApiDataFGContract(SourceFeatureGroupTestMixin):
    feature_group_class: type[SourceFeatureGroup] = ApiDataFG
    present_column = "api_col"
    missing_column = "api_missing"
    expected_source = "api_data:k"
    switched_off = {
        "test_a_refusing_input_data_match_gate_keeps_the_source_untouched_and_reports_the_reason": Cause.OTHER,
        "test_an_unknown_handle_records_a_rejection_listing_the_known_handles": Cause.OTHER,
        "test_a_non_str_handle_is_no_narrowing_and_is_rejected_by_the_option_check": Cause.OTHER,
        "test_pointed_with_a_missing_column_aborts_naming_source_and_columns": Cause.OTHER,
        "test_missing_column_text_hides_the_columns_unless_a_handle_narrows_the_search": Cause.OTHER,
        "test_a_typo_of_a_known_column_suggests_it_only_where_the_columns_are_shown": Cause.OTHER,
        "test_a_subclass_takes_over_a_pointer_keyed_on_its_parent": Cause.OTHER,
    }

    def own_dac(self) -> DataAccessCollection:
        return DataAccessCollection().with_api_data_columns({"k": ("api_col",)})

    def foreign_dac(self) -> DataAccessCollection:
        dac = DataAccessCollection()
        dac.add_file("foreign_file", "foreign.csv")
        return dac

    def own_pointer(self) -> Any:
        return "k"

    def other_pointer(self) -> tuple[Any, str]:
        return "k", "api_data:k"

    def foreign_handle_dac(self) -> tuple[DataAccessCollection, str]:
        return self.foreign_dac(), "foreign_file"

    def known_handles_dac(self) -> tuple[DataAccessCollection, list[str]]:
        return self.own_dac(), ["k"]

    def run_all_kwargs(self) -> dict[str, Any]:
        return {"api_data": {"k": {"api_col": [1, 2]}}}

    def _pointed(self, name: str, key: Any = "k") -> Feature:
        return self._feature(name, {self._group_name(): key})

    # Diverges: a column named in a set counts as pointed, so a refusal aborts resolution instead of declining.
    def test_a_refusing_input_data_match_gate_keeps_the_source_untouched_and_reports_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        touched: list[str] = []
        for name in ("find_sources", "columns", "has_column"):
            monkeypatch.setattr(ApiDataFG, name, _spy(name, getattr(ApiDataFG, name), touched))

        with reader_probe_scope(ReaderProbeScope(MatchRefuseAll(), None, None, None, None)):
            with pytest.raises(FeatureResolutionError) as exc_info:
                self._evaluate(self._own_feature(), self.own_dac())

        assert touched == []
        elimination = exc_info.value.result.eliminations[ApiDataFG]
        assert "refused at INPUT_DATA_MATCH" in elimination.reason
        assert "no clearance" in elimination.reason
        assert elimination.code is CoreDeclineCode.EXTENDER_REFUSED

    # Diverges: data_access_handle is not an ApiDataFG option, so any handle value is ignored.
    def test_an_unknown_handle_records_a_rejection_listing_the_known_handles(self) -> None:
        feature = Feature(self.present_column, self._options(context={HANDLE_OPTION: "unknown_handle_for_suite"}))
        assert self._claims(feature, self.own_dac())

    # Diverges: a non-str handle is ignored too, not rejected.
    def test_a_non_str_handle_is_no_narrowing_and_is_rejected_by_the_option_check(self) -> None:
        feature = Feature(self.present_column, self._options(context={HANDLE_OPTION: ["own_handle"]}))
        assert self._claims(feature, self.own_dac())

    # Diverges: the pointer names a set that must be in the collection, so it resolves against own_dac.
    def test_pointed_with_a_missing_column_aborts_naming_source_and_columns(self) -> None:
        with pytest.raises(FeatureResolutionError) as exc_info:
            self._resolve(self._pointed(self.missing_column), self.own_dac())
        message = str(exc_info.value)
        assert self.expected_source in message
        assert self.present_column in message
        assert exc_info.value.result.eliminations[ApiDataFG].code is CoreDeclineCode.COLUMN_NOT_FOUND

    # Diverges: a pointer, not a handle, narrows the search and reveals the columns.
    def test_missing_column_text_hides_the_columns_unless_a_handle_narrows_the_search(self) -> None:
        unpointed = self._evaluate(Feature(self.missing_column), self.own_dac())
        reason = unpointed.eliminations[ApiDataFG].reason
        assert self.expected_source in reason
        assert self.present_column not in reason

        with pytest.raises(FeatureResolutionError) as exc_info:
            self._resolve(self._pointed(self.missing_column), self.own_dac())
        assert self.present_column in str(exc_info.value)

    # Diverges: a pointed typo is checked against the collection's set, and no handle narrows.
    def test_a_typo_of_a_known_column_suggests_it_only_where_the_columns_are_shown(self) -> None:
        typo = f"{self.present_column}x"
        with pytest.raises(ValueError) as exc_info:
            self._resolve(self._pointed(typo), self.own_dac())
        assert f"did you mean '{self.present_column}'?" in str(exc_info.value)

        unpointed = self._evaluate(Feature(typo), self.own_dac())
        assert "did you mean" not in unpointed.eliminations[ApiDataFG].reason

    # Diverges: the pointed set must be in the collection, so the takeover is evaluated with own_dac.
    def test_a_subclass_takes_over_a_pointer_keyed_on_its_parent(self) -> None:
        sub = self._gated_subclass()
        mapping: FeatureGroupEnvironmentMapping = {cast(type[SourceFeatureGroup], sub): {self.load_framework}}
        feature = self._feature(self.present_column, {ApiDataFG.__name__: "k"})
        result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, self.own_dac())
        assert sub in result.identified
        del sub, mapping, result, feature

    def test_a_pointer_resolves_against_the_collections_sets(self) -> None:
        dac = DataAccessCollection().with_api_data_columns(_TWO_SETS)
        feature = self._pointed("other_col", "k2")
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == "api_data:k2"

    def test_an_unknown_pointer_key_aborts_naming_the_known_keys(self) -> None:
        dac = DataAccessCollection().with_api_data_columns(_TWO_SETS)
        with pytest.raises(FeatureResolutionError) as exc_info:
            self._resolve(self._pointed(self.present_column, "nope"), dac)
        message = str(exc_info.value)
        assert "nope" in message
        assert "k" in message
        assert "k2" in message
        assert exc_info.value.result.eliminations[ApiDataFG].code is CoreDeclineCode.HANDLE_UNKNOWN

    def test_a_pointer_with_no_collection_aborts_with_no_known_keys(self) -> None:
        with pytest.raises(FeatureResolutionError, match="known keys: \\[\\]"):
            self._resolve(self._pointed(self.present_column), None)
