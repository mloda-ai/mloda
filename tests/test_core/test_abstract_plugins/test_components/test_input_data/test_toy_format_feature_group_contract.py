"""ToyFormatFG against the shared FormatFeatureGroup contract."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.provider import FormatFeatureGroup
from tests.mixins.reader_feature_groups.format_feature_group_test_mixin import FormatFeatureGroupTestMixin
from tests.test_core.test_abstract_plugins.test_components.test_input_data import toy_format_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import ToyFormatFG


class TestToyFormatFeatureGroupContract(FormatFeatureGroupTestMixin):
    feature_group_class: type[FormatFeatureGroup] = ToyFormatFG
    present_column = "toyfmt_col"
    missing_column = "toyfmt_missing"
    expected_source = "h1:toy"

    def own_dac(self) -> DataAccessCollection:
        return toy_format_group.toy_dac(h1={"toyfmt_col": [1, 2]})

    def foreign_dac(self) -> DataAccessCollection:
        return toy_format_group.foreign_dac()

    def own_pointer(self) -> Any:
        return {"toyfmt_col": [1, 2]}

    def other_pointer(self) -> tuple[Any, str]:
        return {"toyfmt_col": [9]}, "scoped:toy"

    def foreign_handle_dac(self) -> tuple[DataAccessCollection, str]:
        return toy_format_group.foreign_dac(), "toyfmt_foreign"

    def test_an_unknown_handle_records_a_rejection_listing_the_known_handles(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        pytest.skip("the toy group records no handle rejections")

    def test_pointed_with_a_missing_column_aborts_naming_source_and_columns(self) -> None:
        pytest.skip("the toy group does not abort on a pointed source")

    def test_a_subclass_takes_over_a_pointer_keyed_on_its_parent(self) -> None:
        pytest.skip("the toy group reads its pointer under its own class name, not its parent's")
