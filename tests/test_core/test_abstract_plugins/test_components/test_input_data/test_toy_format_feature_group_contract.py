"""ToyFormatFG against the shared FormatFeatureGroup contract."""

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
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
