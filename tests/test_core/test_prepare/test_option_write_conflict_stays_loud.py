"""Option-write conflicts: a source group's conflicting reader pair is contained as a non-match.

A source group's conflicting reader pair names classes, never the value.
A user-supplied reserved reader key is rejected at Feature construction and at resolution.
"""

from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import FilterType, GlobalFilter, SingleFilter
from mloda.provider import SourceMatch
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    ToySourceFG,
    ToyOtherSourceFG,
    toy_dac,
)


OPAQUE_OPTION_VALUE = "tok_opaque_value"


class TestUserSuppliedReservedReaderKeyIsRejected:
    """A user-supplied reserved reader key is rejected at construction and at resolution without echoing its value."""

    @pytest.mark.parametrize(
        "options",
        [
            {"BaseInputData": OPAQUE_OPTION_VALUE},
            Options(group={"BaseInputData": OPAQUE_OPTION_VALUE}),
            Options(context={"BaseInputData": OPAQUE_OPTION_VALUE}),
        ],
        ids=["dict", "group", "context"],
    )
    def test_feature_construction_rejects_reserved_key(self, options: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            Feature("col", options=options)

        message = str(excinfo.value)
        assert "BaseInputData" in message
        assert OPAQUE_OPTION_VALUE not in message

    @staticmethod
    def _add_to_group(feature: Feature) -> None:
        feature.options.add_to_group("BaseInputData", OPAQUE_OPTION_VALUE)

    @staticmethod
    def _add_to_context(feature: Feature) -> None:
        feature.options.add_to_context("BaseInputData", OPAQUE_OPTION_VALUE)

    @staticmethod
    def _reassign_options(feature: Feature) -> None:
        feature.options = Options(group={"BaseInputData": OPAQUE_OPTION_VALUE})

    @pytest.mark.parametrize(
        "smuggle",
        [_add_to_group, _add_to_context, _reassign_options],
        ids=["add_to_group", "add_to_context", "reassign_options"],
    )
    def test_resolution_rejects_key_added_after_construction(self, smuggle: Callable[[Feature], None]) -> None:
        feature = Feature("col")
        smuggle(feature)
        plugins: FeatureGroupEnvironmentMapping = {}

        with pytest.raises(ValueError) as excinfo:
            IdentifyFeatureGroupClass.evaluate(feature, plugins, None)

        message = str(excinfo.value)
        assert "BaseInputData" in message
        assert OPAQUE_OPTION_VALUE not in message

    def test_global_filter_criteria_rejects_key_added_after_construction(self) -> None:
        single_filter = SingleFilter(Feature("col"), FilterType.EQUAL, {"value": 1})
        single_filter.filter_feature.options.add_to_group("BaseInputData", (OPAQUE_OPTION_VALUE, OPAQUE_OPTION_VALUE))

        with pytest.raises(ValueError) as excinfo:
            GlobalFilter().criteria(FeatureGroup, single_filter, None)

        message = str(excinfo.value)
        assert "BaseInputData" in message
        assert OPAQUE_OPTION_VALUE not in message


_PAIR_MARKER = "toyfmt_pair_conflict_marker_path"


class TestSourceGroupReaderPairConflictIsContained:
    KEY = "BaseInputData"

    def _probe(self, options: Options) -> Any:
        return probe_match_criteria(ToySourceFG, "toyfmt_pair_col", options, toy_dac(h={"toyfmt_pair_col": [1]}))

    @pytest.mark.parametrize(
        "existing",
        [(ToyOtherSourceFG, SourceMatch(source="x", access=_PAIR_MARKER)), _PAIR_MARKER],
        ids=["different_pair", "non_pair_value"],
    )
    @pytest.mark.parametrize("where", ["group", "context"])
    def test_a_different_existing_value_is_contained_and_names_no_value(self, existing: Any, where: str) -> None:
        data = {self.KEY: existing}
        options = Options(group=data) if where == "group" else Options(context=data)

        outcome = self._probe(options)

        assert not outcome.matched
        assert outcome.matcher_error is not None
        message = str(outcome.matcher_error)
        assert _PAIR_MARKER not in message
        assert "SourceMatch(" not in message
        assert "ToySourceFG" in message
        if isinstance(existing, tuple):
            assert "ToyOtherSourceFG" in message
        else:
            assert "str" in message

    def test_an_equal_existing_pair_still_matches(self) -> None:
        match = ToySourceFG.find_sources(
            ToySourceFG.CLAIM_ROUTES[0], "toyfmt_pair_col", Options(), toy_dac(h={"toyfmt_pair_col": [1]})
        )[0]

        outcome = self._probe(Options(group={self.KEY: (ToySourceFG, match)}))

        assert outcome.matched
        assert outcome.matcher_error is None
