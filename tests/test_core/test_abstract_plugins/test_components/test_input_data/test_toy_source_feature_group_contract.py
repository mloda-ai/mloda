"""ToySourceFG against the shared SourceFeatureGroup contract."""

from typing import Any

from mloda._testing.source.source_suite import SourceFeatureGroupTestMixin
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.provider import SourceFeatureGroup
from tests.test_core.test_abstract_plugins.test_components.test_input_data import toy_source_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    ToyOpenFG,
    ToyPointedCheckedFG,
    ToySourceFG,
)


class _ToyHooks:
    def own_dac(self) -> DataAccessCollection:
        return toy_source_group.toy_dac(h1={"toyfmt_col": [1, 2]})

    def foreign_dac(self) -> DataAccessCollection:
        return toy_source_group.foreign_dac()

    def own_pointer(self) -> Any:
        return {"toyfmt_col": [1, 2]}

    def other_pointer(self) -> tuple[Any, str]:
        return {"toyfmt_col": [9]}, "scoped:toy"

    def foreign_handle_dac(self) -> tuple[DataAccessCollection, str]:
        return toy_source_group.foreign_dac(), "toyfmt_foreign"

    def known_handles_dac(self) -> tuple[DataAccessCollection, list[str]]:
        return toy_source_group.toy_dac(h1={"toyfmt_col": [1]}, h2={"toyfmt_col": [2]}), ["h1", "h2"]


class TestToySourceFeatureGroupContract(_ToyHooks, SourceFeatureGroupTestMixin):
    feature_group_class: type[SourceFeatureGroup] = ToySourceFG
    present_column = "toyfmt_col"
    missing_column = "toyfmt_missing"
    expected_source = "h1:toy"
    pointed_source = "scoped:toy"


class TestToyOpenSourceFeatureGroupContract(_ToyHooks, SourceFeatureGroupTestMixin):
    feature_group_class: type[SourceFeatureGroup] = ToyOpenFG
    present_column = "toyfmt_col"
    missing_column = "toyfmt_missing"
    expected_source = "scoped:toy"
    pointed_source = "scoped:toy"


class TestToyPointedCheckedSourceFeatureGroupContract(_ToyHooks, SourceFeatureGroupTestMixin):
    feature_group_class: type[SourceFeatureGroup] = ToyPointedCheckedFG
    present_column = "toyfmt_col"
    missing_column = "toyfmt_missing"
    expected_source = "scoped:toy"
    pointed_source = "scoped:toy"
