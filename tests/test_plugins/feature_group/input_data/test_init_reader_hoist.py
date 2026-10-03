"""Pins init_reader as the single concrete implementation on BaseInputData.

Contract: init_reader(self, reader_data_access) takes the (ReaderClass, data_access) pair and
returns (reader instance, data_access); load(features) takes the pair from features.input_data_match
and raises ValueError when it is None. Subclasses are module-level and never final readers.
"""

from typing import Any
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class HoistBareInputData(BaseInputData):
    """Minimal BaseInputData subclass implementing nothing; drives the base init_reader directly."""


class HoistSentinelReader(BaseInputData):
    """Reader class carried inside the pair for the happy-path pin."""


class TestInitReaderIsHoistedToBase:
    """After the hoist, the three reader families no longer override init_reader."""

    @pytest.mark.parametrize("family", [ReadDB, ReadFile, ReadDocument])
    def test_family_does_not_override_init_reader(self, family: type[BaseInputData]) -> None:
        # init_reader is a plain instance method; accessing it on the class object yields
        # the plain function, so identity comparison works without __func__ unwrapping.
        assert family.init_reader is BaseInputData.init_reader

    @pytest.mark.parametrize("family", [ReadDB, ReadFile, ReadDocument])
    def test_family_dict_has_no_init_reader(self, family: type[BaseInputData]) -> None:
        assert "init_reader" not in family.__dict__


class TestBaseInitReaderIsConcrete:
    """BaseInputData.init_reader becomes the single concrete implementation."""

    def test_happy_path_returns_reader_instance_and_data_access(self) -> None:
        data_access: dict[str, Any] = {"dsn": "x"}
        reader, returned_data_access = HoistBareInputData().init_reader((HoistSentinelReader, data_access))
        assert isinstance(reader, HoistSentinelReader)
        assert returned_data_access is data_access

    def test_load_without_match_raises_value_error(self) -> None:
        features = FeatureSet()
        features.add(Feature("hoist_unmatched_col"))
        with pytest.raises(ValueError):
            HoistBareInputData().load(features)

    def test_load_takes_the_pair_from_the_feature_set_match(self) -> None:
        feature = Feature("hoist_matched_col")
        feature.input_data_match = (HoistSentinelReader, "access")
        features = FeatureSet()
        features.add(feature)
        with patch.object(BaseInputData, "_load_data_via_hook", return_value="loaded") as hook:
            assert HoistBareInputData().load(features) == "loaded"
        reader, data_access, _ = hook.call_args.args
        assert isinstance(reader, HoistSentinelReader)
        assert data_access == "access"
