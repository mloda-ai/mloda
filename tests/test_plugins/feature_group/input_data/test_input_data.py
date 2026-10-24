import importlib
import os
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components import declaration_surface
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import GeneratedFeatureGroup
from mloda.user import mloda

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature  # noqa: F401


class InputDataTestFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {
            "InputDataTestFeatureGroup_id",
            "InputDataTestFeatureGroup_V1",
            "InputDataTestFeatureGroup_V2",
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "InputDataTestFeatureGroup_id": [12, 2, 3],
            "InputDataTestFeatureGroup_V1": [1, 2, 3],
            "InputDataTestFeatureGroup_V2": [1, 2, 3],
        }


class TestInputData:
    file_path = f"{os.path.dirname(os.path.abspath(__file__))}/creditcard_2023.csv"

    feature_names = "id,V1,V2"
    feature_list = feature_names.split(",")

    def test_data_creator(self) -> Any:
        features = [f"InputDataTestFeatureGroup_{f}" for f in self.feature_list]
        result = mloda.run_all(features, compute_frameworks=["PyArrowTable"])
        assert result[0].to_pydict() == {
            "InputDataTestFeatureGroup_id": [12, 2, 3],
            "InputDataTestFeatureGroup_V1": [1, 2, 3],
            "InputDataTestFeatureGroup_V2": [1, 2, 3],
        }

    def test_api_data(self) -> None:
        features = [f"ApiDataTestFeatureGroup_{f}" for f in self.feature_list]

        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
            api_data={
                "TestApiData": {
                    "ApiDataTestFeatureGroup_id": [12, 2, 3],
                    "ApiDataTestFeatureGroup_V1": [1, 2, 3],
                    "ApiDataTestFeatureGroup_V2": [1, 2, 3],
                }
            },
        )

        assert result[0].to_pydict() == {
            "ApiDataTestFeatureGroup_id": [12, 2, 3],
            "ApiDataTestFeatureGroup_V1": [1, 2, 3],
            "ApiDataTestFeatureGroup_V2": [1, 2, 3],
        }

    def test_agg_creator_and_api_data(self) -> None:
        _agg_creator = Feature(
            name="sum_of_",
            options={
                "sum": ("InputDataTestFeatureGroup_id", "InputDataTestFeatureGroup_V1"),
            },
        )
        _agg_api_input_feature = Feature(
            name="sum_of_",
            options={
                "sum": ("ApiDataTestFeatureGroup_V1", "ApiDataTestFeatureGroup_V2"),
            },
        )

        result = mloda.run_all(
            [_agg_creator, _agg_api_input_feature],
            compute_frameworks=["PyArrowTable"],
            api_data={
                "TestApiData": {
                    "ApiDataTestFeatureGroup_id": [12, 2, 3],
                    "ApiDataTestFeatureGroup_V1": [1, 2, 3],
                    "ApiDataTestFeatureGroup_V2": [1, 2, 3],
                }
            },
        )

        for res in result:
            assert (
                res.to_pydict()
                == {
                    "SumFeature_InputDataTestFeatureGroup_idInputDataTestFeatureGroup_V1": [23, 23, 23],
                }
            ) or (
                res.to_pydict()
                == {
                    "SumFeature_ApiDataTestFeatureGroup_V1ApiDataTestFeatureGroup_V2": [12, 12, 12],
                }
            )


class _BareInputData(BaseInputData):
    """A subclass that adds nothing: no hook of its own."""


_REMOVED_BASE_INPUT_DATA_MEMBERS = [
    "READER_OPTIONS",
    "load_data",
    "load",
    "_load_data_via_hook",
    "init_reader",
    "feature_scope_data_access",
    "global_scope_data_access",
    "match_data_access",
    "_record_unowned_pin",
    "add_base_input_data_to_options",
    "deal_with_base_input_data_name_as_cls_or_str",
    "wrap_feature_scoped_access",
    "_wrap_scoped_access",
    "reader_option",
    "reader_option_default",
    "reader_option_specs",
    "declared_reader_option_keys",
    "data_access_identity",
    "describe_columns",
    "count_rows",
    "declared_attributes",
]

_REMOVED_MODULE_SYMBOLS = [
    ("mloda.core.abstract_plugins.components.input_data.reader_probe", "probe_reader"),
    ("mloda.core.abstract_plugins.components.match_rejection", "INPUT_DATA_OWNED_STAGE"),
    ("mloda.core.abstract_plugins.components.match_rejection", "restamp_match_rejections_since"),
    ("mloda.core.abstract_plugins.components.declared_attributes", "current_declaration_requirement"),
    ("mloda.core.abstract_plugins.components.declared_attributes", "declaration_requirement_scope"),
    ("mloda.core.abstract_plugins.components.credential_scrub", "_URI_PATTERN"),
    ("mloda.provider", "DataCreator"),
]


class TestBaseInputDataIsOnlyAMarker:
    """BaseInputData keeps no reader path: data comes from a SourceFeatureGroup, or a GeneratedFeatureGroup."""

    @pytest.mark.parametrize("member", _REMOVED_BASE_INPUT_DATA_MEMBERS)
    def test_the_legacy_reader_member_is_gone(self, member: str) -> None:
        assert not hasattr(BaseInputData, member)

    @pytest.mark.parametrize(("module", "symbol"), _REMOVED_MODULE_SYMBOLS)
    def test_the_legacy_module_symbol_is_gone(self, module: str, symbol: str) -> None:
        assert not hasattr(importlib.import_module(module), symbol)

    def test_the_reader_declaration_surface_is_gone(self) -> None:
        assert not hasattr(declaration_surface, "DeclarationSurface")

    def test_a_bare_subclass_matches_nothing_on_its_own(self) -> None:
        options = Options({_BareInputData.__name__: "bare_access", "BaseInputData": (_BareInputData, "bare_access")})
        access = DataAccessCollection(files={"bare_access.csv"}, connections={"bare_handle": "bare_access"})

        assert _BareInputData().matches("bare_input_feature", options, access) is False
        assert _BareInputData().matches("bare_input_feature", Options(), None) is False

    def test_a_group_returning_a_bare_subclass_matches_nothing(self) -> None:
        class BareInputDataGroup(FeatureGroup):
            @classmethod
            def input_data(cls) -> BaseInputData | None:
                return _BareInputData()

        access = DataAccessCollection(files={"bare_access.csv"})

        result = IdentifyFeatureGroupClass.evaluate(
            Feature("bare_input_feature"), {BareInputDataGroup: {PyArrowTable}}, None, access
        )

        assert result.identified == {}

    @pytest.mark.parametrize(
        ("hook", "value"),
        [
            ("match_subclass_data_access", classmethod(lambda cls, data_access, feature_names, options=None: None)),
            ("load_data", classmethod(lambda cls, data_access, features: None)),
            ("READER_OPTIONS", {}),
        ],
    )
    def test_defining_a_legacy_hook_raises_naming_the_source_feature_group(self, hook: str, value: Any) -> None:
        with pytest.raises(TypeError, match="SourceFeatureGroup") as excinfo:
            type("LegacyHookInputData", (BaseInputData,), {hook: value})

        assert hook in str(excinfo.value)

    @pytest.mark.parametrize(
        ("hook", "value"),
        [
            ("init_reader", classmethod(lambda cls, data_access: None)),
            ("match_data_access", classmethod(lambda cls, data_access, feature_names, options=None: None)),
        ],
    )
    def test_defining_init_reader_or_match_data_access_raises(self, hook: str, value: Any) -> None:
        with pytest.raises(TypeError, match="SourceFeatureGroup") as excinfo:
            type("LegacyHookInputData2", (BaseInputData,), {hook: value})

        assert hook in str(excinfo.value)

    def test_a_mixin_defining_a_legacy_hook_raises(self) -> None:
        class LoadMixin:
            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> None:
                return None

        with pytest.raises(TypeError, match="SourceFeatureGroup") as excinfo:
            type("MixedInputData", (LoadMixin, BaseInputData), {})

        assert "load_data" in str(excinfo.value)

    def test_a_subclass_without_a_legacy_hook_still_defines(self) -> None:
        defined = type("PlainMarkerInputData", (BaseInputData,), {})

        assert issubclass(defined, BaseInputData)
