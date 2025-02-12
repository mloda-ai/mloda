from typing import Union
from unittest.mock import patch
import pytest

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork


from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.api.request import mlodaAPI
from mloda_core.core.engine import Engine
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link
from tests.test_core.test_setup.test_graph_builder import BaseTestGraphFeatureGroup3
from tests.test_core.test_setup.test_link_resolver import BaseLinkTestFeatureGroup1
from mloda_core.abstract_plugins.components.utils import get_all_subclasses


class TestmlodaAPI:
    @pytest.fixture
    def features(self) -> list[str]:
        return ["some_feature"]

    def init_with_no_params(self, features: list[Union[str, Feature]]) -> mlodaAPI:
        api_request = mlodaAPI(features)

        assert isinstance(api_request.features, Features)
        assert isinstance(api_request.compute_framework, set)
        assert api_request.links is None
        return api_request

    def test_init_with_all_params(self) -> None:
        features = [Feature("BaseTestGraphFeatureGroup3")]

        compute_fws = [fw.get_class_name() for fw in get_all_subclasses(ComputeFrameWork)]
        links = {
            Link.inner(
                (BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                (BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
            )
        }

        api_request = mlodaAPI(features, compute_fws, links)  # type: ignore
        assert isinstance(api_request.features, Features)
        assert len(api_request.compute_framework) == len(get_all_subclasses(ComputeFrameWork))
        assert api_request.links is not None
        assert len(api_request.links) == 1

    def test_setup_engine(self, features: list[Union[str, Feature]]) -> None:
        with patch("mloda_core.core.engine.Engine.create_setup_execution_plan"):
            mloda_api = self.init_with_no_params(features)
            assert isinstance(mloda_api.engine, Engine)
