"""The source suite catches a group that claims foreign sources, leaks into options and declares a secret option."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

INNER = """
from mloda._testing.source.source_suite import SourceFeatureGroupTestMixin
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.provider import ClaimRoute, NamePolicy, SourceMatch
from tests.test_core.test_abstract_plugins.test_components.test_input_data import toy_source_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    TOY_KEY,
    ToySourceBase,
)


class ToyfmtBrokenFG(ToySourceBase):
    CLAIM_ROUTES = (ClaimRoute("toy", NamePolicy.CHECKED, True),)
    PROPERTY_MAPPING = {"password": PropertySpec(explanation="Broken secret option.", default=None)}

    @classmethod
    def find_sources(cls, route, feature_name, options, data_access_collection):
        if feature_name != "toyfmt_broken_col" or data_access_collection is None:
            return []
        matches = [
            SourceMatch(source=f"{handle}:toy", access={"toyfmt_broken_col": [1]})
            for handle in sorted(data_access_collection.handles())
        ]
        for match in matches:
            options.add_to_group("toyfmt_leaked", match.source)
        return matches


class TestBroken(SourceFeatureGroupTestMixin):
    feature_group_class = ToyfmtBrokenFG
    present_column = "toyfmt_broken_col"
    missing_column = "toyfmt_broken_missing"
    expected_source = "h1:toy"
    pointed_source = "scoped:toy"

    def own_dac(self):
        return toy_source_group.toy_dac(h1={"toyfmt_broken_col": [1]})

    def foreign_dac(self):
        return toy_source_group.foreign_dac()

    def own_pointer(self):
        return {"toyfmt_broken_col": [1]}

    def other_pointer(self):
        return {"toyfmt_broken_col": [9]}, "scoped:toy"

    def foreign_handle_dac(self):
        return toy_source_group.foreign_dac(), "toyfmt_foreign"

    def known_handles_dac(self):
        return toy_source_group.toy_dac(h1={"toyfmt_broken_col": [1]}), ["h1"]
"""

SELECTED = (
    "test_declines_foreign_sources or test_leaves_collection_and_options_unchanged"
    " or test_declares_no_secret_shaped_option or test_claims_its_own_source"
)


def test_a_broken_source_group_fails_the_suite_for_each_defect(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_inner=INNER)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly", "--tb=short", "-k", SELECTED)

    result.assert_outcomes(passed=1, failed=3)
    result.stdout.fnmatch_lines(
        [
            "*_ TestBroken.test_declines_foreign_sources _*",
            "*assert not self._claims(Feature(self.present_column), self.foreign_dac())*",
            "*_ TestBroken.test_leaves_collection_and_options_unchanged _*",
            "*assert feature.options.group == group*",
            "*_ TestBroken.test_declares_no_secret_shaped_option _*",
            "*secrets: option 'password'*",
        ]
    )
